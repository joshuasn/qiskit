// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Lower: emission circuit (IR2) → template circuit, sampling graph and parameter table.
//!
//! Each collect box becomes a *synth template*, the parametric fragment whose angles the graph
//! fills in; the absorbed gates in its body fold into those angles rather than reaching the
//! template. `Emit` instructions disappear. Content boxes are kept, since what is left in one after
//! absorption is exactly the content that could not be absorbed, and the annotations and duration
//! they carry are meant for the consumer of the template.
//!
//! **Parameters are minted here and nowhere earlier**, so every pass that changes the number or
//! width of collectors must already have run.
//!
//! Nothing here mutates its input. Both readers traverse in `topological_op_nodes` order, which is
//! what lets [`build_sampling_graph`] pair its collectors with the template's parameter ranges by
//! position. Inside a collector body that order must not be reported as circuit order; see
//! [`Collect::steps`](crate::sampling_graph::Collect::steps).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::{Symbol, Value};

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::operations::StandardInstruction;

use super::utils::{IntoPyResult, block_body, collect_annotation, emission_spec, is_emission};
use crate::annotated_circuit::SynthesizerType;
use crate::distributions::DistributionTable;
use crate::emission_circuit::Emit;
use crate::parameters::ParameterTable;
use crate::partition::Partition;
use crate::sampling_graph::{
    AbsorbedGate, AbsorbedParam, Collect, CollectStep, Direction, Edge, Emission, LocalEmission,
    Measure, Node, NodeKind, Propagate, SamplingGraph,
};
use crate::virtual_type::{VirtualType, propagates};

/// How many angles a synthesizer needs per qubit; both are three-angle Euler decompositions.
const PARAMS_PER_QUBIT: usize = 3;

/// Where one collector's angles live in the template's parameter vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectorParams {
    /// The collector's qubits, ascending.
    pub qubits: Vec<usize>,
    pub synthesizer: SynthesizerType,
    /// Indices into the template's parameter vector, `qubits.len() * PARAMS_PER_QUBIT` of them,
    /// grouped per qubit in `qubits` order.
    pub param_indices: Vec<usize>,
}

/// Build the template circuit for an emission circuit.
///
/// Returns the template plus one [`CollectorParams`] per collector, in circuit order.
pub fn build_template(dag: &DAGCircuit) -> PyResult<(CircuitData, Vec<CollectorParams>)> {
    let mut out = CircuitData::with_capacity(
        dag.num_qubits() as u32,
        dag.num_clbits() as u32,
        dag.num_ops(),
        Param::Float(0.0),
    )
    .into_py_result()?;
    let mut collectors = Vec::new();
    let mut next_param = 0usize;

    // The identity frame: at the top level a scope-local qubit is already a circuit qubit, and the
    // output frame and the global frame coincide.
    let identity: Vec<usize> = (0..dag.num_qubits()).collect();
    write_scope(
        dag,
        &mut out,
        &identity,
        &identity,
        &mut collectors,
        &mut next_param,
    )?;
    Ok((out, collectors))
}

/// One collector as seen from Python: qubits, synthesizer name, parameter indices.
type CollectorSummary = (Vec<usize>, String, Vec<usize>);

/// Python-facing entry point: returns the template and the collector parameter map.
#[pyfunction]
#[pyo3(name = "build_template")]
pub fn py_build_template(dag: &DAGCircuit) -> PyResult<(PyCircuitData, Vec<CollectorSummary>)> {
    let (template, collectors) = build_template(dag)?;
    let summary = collectors
        .into_iter()
        .map(|c| {
            let synth = match c.synthesizer {
                SynthesizerType::RzSx => "rzsx".to_string(),
                SynthesizerType::RzRx => "rzrx".to_string(),
            };
            (c.qubits, synth, c.param_indices)
        })
        .collect();
    Ok((PyCircuitData { inner: template }, summary))
}

/// Lower an emission circuit into its three artifacts, all read off the same IR2 circuit: template
/// circuit, sampling graph, parameter table.
#[pyfunction]
#[pyo3(name = "lower")]
pub fn py_lower(
    dag: &DAGCircuit,
    table: &DistributionTable,
) -> PyResult<(PyCircuitData, SamplingGraph, ParameterTable)> {
    let (template, collectors) = build_template(dag)?;
    let (graph, parameters) = build_sampling_graph(dag, table, &collectors)?;
    Ok((PyCircuitData { inner: template }, graph, parameters))
}

/// Emit one scope's worth of template content.
///
/// `frame` maps scope-local qubits to indices in `out`, which is the enclosing box's body when this
/// scope is nested; `global` maps them to circuit qubits, which is the frame a [`CollectorParams`] is
/// always reported in. At the top level the two coincide.
fn write_scope(
    src: &DAGCircuit,
    out: &mut CircuitData,
    frame: &[usize],
    global: &[usize],
    collectors: &mut Vec<CollectorParams>,
    next_param: &mut usize,
) -> PyResult<()> {
    // Topological order, which is the order parameters are minted in and hence the order
    // `CollectorParams` are reported in. `flatten` walks the same order, so the two line up.
    for node in src.topological_op_nodes(false) {
        let inst = src.dag()[node].unwrap_operation();

        // A collector becomes the parametric fragment its angles drive.
        if let Some(spec) = collect_annotation(inst) {
            let locals = src.qargs_interner().get(inst.qubits);
            let written: Vec<usize> = locals.iter().map(|q| frame[q.index()]).collect();
            let qubits: Vec<usize> = locals.iter().map(|q| global[q.index()]).collect();
            let count = qubits.len() * PARAMS_PER_QUBIT;
            let param_indices: Vec<usize> = (*next_param..*next_param + count).collect();
            *next_param += count;

            write_synth_template(out, spec.synthesizer(), &written, &param_indices)?;
            collectors.push(CollectorParams {
                qubits,
                synthesizer: spec.synthesizer(),
                param_indices,
            });
            continue;
        }

        // Emissions are markers for the sampling graph; they are not executable.
        if is_emission(inst) {
            continue;
        }

        // A content box is kept, not flattened: its annotations and duration are the point of it, and
        // after absorption its body holds exactly what could not be absorbed — so the box marks the
        // hard content rather than merely having contained it. Recursing writes its body in the box's
        // own frame, nested collectors included.
        if let Some(body) = plain_box_body(src, inst)? {
            let locals = src.qargs_interner().get(inst.qubits);
            let width = locals.len();
            let cargs = src.cargs_interner().get(inst.clbits).to_vec();
            let inner_global: Vec<usize> = locals.iter().map(|q| global[q.index()]).collect();
            let inner_frame: Vec<usize> = (0..width).collect();
            let mut inner_out = CircuitData::with_capacity(
                width as u32,
                cargs.len() as u32,
                body.num_ops(),
                Param::Float(0.0),
            )
            .into_py_result()?;
            write_scope(
                body,
                &mut inner_out,
                &inner_frame,
                &inner_global,
                collectors,
                next_param,
            )?;
            // The op carries the box's annotations, duration and widths; only its body is rebuilt.
            let qargs: Vec<Qubit> = locals
                .iter()
                .map(|q| Qubit(frame[q.index()] as u32))
                .collect();
            let block = out.add_block(inner_out);
            out.push_packed_operation(
                inst.op.clone(),
                Some(qiskit_circuit::instruction::Parameters::Blocks(vec![block])),
                &qargs,
                &cargs,
            )
            .into_py_result()?;
            continue;
        }

        // Everything else is real content, remapped into the template's frame.
        let qargs: Vec<Qubit> = src
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|q| Qubit(frame[q.index()] as u32))
            .collect();
        let cargs = src.cargs_interner().get(inst.clbits).to_vec();
        copy_with_qargs(inst, out, &qargs, &cargs)?;
    }
    Ok(())
}

/// Write the parametric fragment for one collector, on each of its qubits.
///
/// `RzSx` is `rz sx rz sx rz` and `RzRx` is `rz rx rz`; both take three angles.
fn write_synth_template(
    out: &mut CircuitData,
    synthesizer: SynthesizerType,
    qubits: &[usize],
    param_indices: &[usize],
) -> PyResult<()> {
    for (position, qubit) in qubits.iter().enumerate() {
        let angles: Vec<Param> = param_indices
            [position * PARAMS_PER_QUBIT..(position + 1) * PARAMS_PER_QUBIT]
            .iter()
            .map(|index| fresh_parameter(*index))
            .collect();
        let target = [Qubit(*qubit as u32)];
        let sequence: Vec<(StandardGate, Option<&Param>)> = match synthesizer {
            SynthesizerType::RzSx => vec![
                (StandardGate::RZ, Some(&angles[0])),
                (StandardGate::SX, None),
                (StandardGate::RZ, Some(&angles[1])),
                (StandardGate::SX, None),
                (StandardGate::RZ, Some(&angles[2])),
            ],
            SynthesizerType::RzRx => vec![
                (StandardGate::RZ, Some(&angles[0])),
                (StandardGate::RX, Some(&angles[1])),
                (StandardGate::RZ, Some(&angles[2])),
            ],
        };
        for (gate, angle) in sequence {
            let params: Vec<Param> = angle.into_iter().cloned().collect();
            out.push_standard_gate(gate, &params, &target)
                .into_py_result()?;
        }
    }
    Ok(())
}

/// Mint a template parameter named `p{index:04}`, zero-padded so lexicographic order is numeric
/// order.
fn fresh_parameter(index: usize) -> Param {
    // A fresh uuid each time, so two runs' parameters share names but are not equal objects.
    let symbol = Symbol::standalone(format!("p{index:04}"), None);
    Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(symbol)))
}

/// The body of an unannotated box, or `None` if this is not one.
fn plain_box_body<'a>(
    src: &'a DAGCircuit,
    inst: &PackedInstruction,
) -> PyResult<Option<&'a DAGCircuit>> {
    match inst.op.view() {
        OperationRef::ControlFlow(cf)
            if matches!(
                cf.control_flow,
                qiskit_circuit::operations::ControlFlow::Box { .. }
            ) =>
        {
            block_body(src, inst)
        }
        _ => Ok(None),
    }
}

/// Copy an instruction into the template with explicitly remapped bits.
///
/// Errors on a block-carrying instruction, which should be unreachable.
fn copy_with_qargs(
    inst: &PackedInstruction,
    out: &mut CircuitData,
    qargs: &[Qubit],
    cargs: &[qiskit_circuit::Clbit],
) -> PyResult<()> {
    if !inst.blocks_view().is_empty() {
        return Err(PyValueError::new_err(format!(
            "cannot lower '{}' into a template: it carries a body but is not a `box`",
            inst.op.name()
        )));
    }
    let params = (!inst.params_view().is_empty()).then(|| {
        qiskit_circuit::instruction::Parameters::Params(
            inst.params_view().iter().cloned().collect(),
        )
    });
    out.push_packed_operation(inst.op.clone(), params, qargs, cargs)
        .into_py_result()
}

// --- Sampling graph construction ----------------------------------------------------------------
//
// The template says *what to execute*; the graph says *how to compute the angles*.

/// One collector, flattened out of the circuit.
struct CollectorInfo {
    qubits: Vec<usize>,
    /// How those qubits group into subsystems, by index into `qubits`.
    partition: Partition,
    synthesizer: SynthesizerType,
    param_indices: Vec<usize>,
    /// Everything this collector composes, in the order `flatten` read it out of the body.
    steps: Vec<CollectStep>,
}

impl CollectorInfo {
    /// The absorbed gates alone, for an enclosing emission crossing this collector: it conjugates
    /// by the gates and ignores what the collector consumes.
    fn gates(&self) -> impl Iterator<Item = &AbsorbedGate> {
        crate::sampling_graph::collect_step_gates(&self.steps)
    }
}

/// The circuit as a flat sequence, which is what makes the propagation walk a simple scan.
enum Event {
    Emission(Emit, Vec<usize>),
    Collector(usize),
    Gate(StandardGate, Vec<usize>),
    Measure(Vec<usize>, Vec<usize>),
    Reset(Vec<usize>),
    /// A real operation with no virtual effect, kept so positions line up with the template.
    Opaque,
}

/// What identifies one conjugation node: a gate occurrence together with the flow crossing it.
///
/// The occurrence is `(event position, offset)`, the offset being the position within a collector's
/// absorbed run and zero for a gate that stands on its own.
type GateKey = (usize, usize, Direction, VirtualType);

/// Build the sampling graph for an emission circuit, and the parameter table it refers into.
///
/// `collectors` must come from [`build_template`] over the same circuit.
pub fn build_sampling_graph(
    dag: &DAGCircuit,
    table: &DistributionTable,
    collectors: &[CollectorParams],
) -> PyResult<(SamplingGraph, ParameterTable)> {
    let mut events = Vec::new();
    let mut infos = Vec::new();
    let mut parameters = ParameterTable::new();
    let identity: Vec<usize> = (0..dag.num_qubits()).collect();
    flatten(dag, &identity, &mut events, &mut infos, &mut parameters)?;

    if infos.len() != collectors.len() {
        return Err(PyValueError::new_err(format!(
            "the template found {} collectors but the graph walk found {}; they must be built from \
             the same circuit",
            collectors.len(),
            infos.len()
        )));
    }
    for (info, params) in infos.iter_mut().zip(collectors) {
        info.param_indices = params.param_indices.clone();
    }

    let mut sg = SamplingGraph::new();

    // Sinks first, so an emission's walk always has a node to terminate at.
    let mut collector_nodes = Vec::with_capacity(infos.len());
    for info in &infos {
        collector_nodes.push(sg.graph.add_node(Node::new(
            info.qubits.clone(),
            info.partition.clone(),
            NodeKind::Collect(Collect {
                synthesizer: info.synthesizer,
                param_indices: info.param_indices.clone(),
                steps: info.steps.clone(),
            }),
        )));
    }

    // One Propagate node per *conjugation*, created lazily and shared by the emissions for which it
    // is the same conjugation. Direction and virtual type are in the key because they change what
    // the node computes, so sharing across them would fuse operations that cannot be evaluated as
    // one.
    let mut gate_nodes: HashMap<GateKey, NodeIndex> = HashMap::new();
    let mut emission_nodes: HashMap<usize, NodeIndex> = HashMap::new();

    for (position, event) in events.iter().enumerate() {
        match event {
            Event::Emission(spec, qubits) => {
                let node = sg.graph.add_node(Node::new(
                    qubits.clone(),
                    spec.partition.clone(),
                    emission_kind(spec, table)?,
                ));
                emission_nodes.insert(position, node);
            }
            Event::Measure(qubits, clbits) => {
                sg.graph.add_node(Node::singletons(
                    qubits.clone(),
                    NodeKind::Measure(Measure {
                        clbit_indices: clbits.clone(),
                    }),
                ));
            }
            Event::Reset(qubits) => {
                sg.graph
                    .add_node(Node::singletons(qubits.clone(), NodeKind::Reset));
            }
            _ => {}
        }
    }

    // Walk each emission to its target collector, wiring the conjugation chain in between.
    // Target resolution is purely positional: scan from the emission in its travel direction to
    // find the nearest compatible collector.
    for (position, event) in events.iter().enumerate() {
        let Event::Emission(spec, qubits) = event else {
            continue;
        };
        let source = emission_nodes[&position];
        // A local emission is resolved in place inside its collector's body, so it never reaches
        // the top-level event list; anything here is still travelling.
        let direction = spec.direction.expect(
            "a local emission never surfaces as a top-level Event::Emission — it lives inside its \
             collector's body",
        );
        // Unreachable in well-formed IR2: build writes both of a box's collectors, so an emission
        // always has a compatible collector ahead of it. Reaching this means either the pairing was
        // broken between the two passes or a hand-built circuit has an emission nothing can collect,
        // which would otherwise show up as a randomization that is never undone — so it is reported
        // rather than skipped.
        let target = scan_for_collector(&events, position, direction, spec, qubits, &infos, table)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "emission on qubits {qubits:?} travelling {direction:?} has no compatible \
                     collector ahead of it; its randomization could not be undone",
                ))
            })?;
        walk_emission(
            &mut sg,
            &events,
            position,
            spec,
            qubits,
            source,
            target,
            collector_nodes[target],
            &infos,
            &mut gate_nodes,
            table,
        )?;
    }
    Ok((sg, parameters))
}

/// Scan from `start` in `direction` for the first collector that can take this emission.
///
/// Proximity decides, filtered by [`compatible`]. A collector that declines is simply crossed — its
/// absorbed gates conjugate the emission on the way past — so an emission travels until it finds one
/// that can take it, out of the box it started in and on through whatever it passes. Reaching the end
/// of the circuit is the error case.
#[allow(clippy::too_many_arguments)]
fn scan_for_collector(
    events: &[Event],
    start: usize,
    direction: Direction,
    spec: &Emit,
    qubits: &[usize],
    infos: &[CollectorInfo],
    table: &DistributionTable,
) -> Option<usize> {
    let range: Box<dyn Iterator<Item = usize>> = match direction {
        Direction::Right => Box::new((start + 1)..events.len()),
        Direction::Left => Box::new((0..start).rev()),
    };
    for i in range {
        if let Event::Collector(index) = &events[i]
            && compatible(&infos[*index], spec, qubits, table)
        {
            return Some(*index);
        }
    }
    None
}

/// Whether this collector can take this emission.
///
/// **This is the seam where "whose emission is this" is decided, and it is deliberately incomplete.**
/// Two conditions are in place:
///
/// - It covers every qubit the emission acts on. A collector that covers only part of an emission
///   could not synthesize the whole of what was emitted. The emission's qubits come from the walk
///   rather than from the spec: a spec groups its *own qargs* by index and is shared by every
///   placement of it, so it cannot know which wires it landed on.
/// - Its synthesizer accepts the emission's virtual type, so the value it would have to produce is one
///   it can express.
///
/// Nothing here asks which annotated box the emission came from — position and these two conditions
/// are all of it. So a collector nested inside an enclosing box will take that box's propagating
/// emission if it happens to be the first one the walk reaches, terminating the enclosing
/// randomization at the inner dressing with none of the enclosing content in between. That is
/// invisible to a round-trip test, since the circuit still evaluates to the same unitary.
///
/// **TO DO: make this a type question.** The intended shape is that an emission carries a type a
/// collector either accepts or declines — a basis change becoming a distinct type rather than a Pauli
/// that looks like any other, an inner twirl marked as unable to collect it — so that a collector that
/// should not have it declines, the emission propagates on, and it reaches its own collector by
/// walking rather than by consulting an id. Until then a nested twirl of the same group is collected
/// early, and `test_sampling_graph.py::TestNestedPropagation` pins that provisional behaviour so the
/// change of rule shows up as a test change rather than silently.
fn compatible(
    info: &CollectorInfo,
    spec: &Emit,
    qubits: &[usize],
    table: &DistributionTable,
) -> bool {
    qubits.iter().all(|q| info.qubits.contains(q))
        && info.synthesizer.accepts(spec.virtual_type(table))
}

/// Read one absorbed gate parameter, interning it only if it is genuinely symbolic.
///
/// A fully bound expression folds to [`AbsorbedParam::Bound`]. A [`Param::Obj`] is refused
/// outright.
fn absorbed_param(table: &mut ParameterTable, param: &Param) -> PyResult<AbsorbedParam> {
    match param {
        Param::Float(value) => Ok(AbsorbedParam::Bound(*value)),
        Param::ParameterExpression(expr) if expr.num_symbols() == 0 => {
            match expr.try_to_value(true).into_py_result()? {
                Value::Real(value) => Ok(AbsorbedParam::Bound(value)),
                Value::Int(value) => Ok(AbsorbedParam::Bound(value as f64)),
                // Reported rather than silently projected onto the real axis: a complex angle is
                // not something a rotation can be given, so it means the circuit was already wrong.
                Value::Complex(value) => Err(PyValueError::new_err(format!(
                    "cannot absorb a gate whose angle evaluates to the complex value {value}"
                ))),
            }
        }
        Param::ParameterExpression(expr) => Ok(AbsorbedParam::Symbolic(table.intern(expr.clone()))),
        Param::Obj(_) => Err(PyValueError::new_err(
            "cannot absorb a gate whose parameter is an opaque Python object: the sampling graph is \
             read without the GIL, so it cannot carry one",
        )),
    }
}

/// Flatten a scope into events, inlining hard boxes and reducing each collector to one event.
fn flatten(
    src: &DAGCircuit,
    frame: &[usize],
    events: &mut Vec<Event>,
    infos: &mut Vec<CollectorInfo>,
    parameters: &mut ParameterTable,
) -> PyResult<()> {
    // The same order `write_scope` uses, so the collectors line up with the template's ranges.
    for node in src.topological_op_nodes(false) {
        let inst = src.dag()[node].unwrap_operation();
        let qubits: Vec<usize> = src
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|q| frame[q.index()])
            .collect();

        if let Some(spec) = collect_annotation(inst) {
            // This read yields *a* linear extension of the body, not the order the absorption walk
            // appended in. Harmless — a local emission spans every qubit it covers, so it stays a
            // barrier — but it must not be reported as circuit order. See `Collect::steps`.
            let mut steps = Vec::new();
            if let Some(body) = block_body(src, inst)? {
                for node in body.topological_op_nodes(false) {
                    let gate = body.dag()[node].unwrap_operation();
                    if let OperationRef::StandardGate(standard) = gate.op.view() {
                        // The angles have to come along: this is the only place they are read, and
                        // the collector's body does not reach the template.
                        let params = gate
                            .params_view()
                            .iter()
                            .map(|param| absorbed_param(parameters, param))
                            .collect::<PyResult<Vec<_>>>()?;
                        steps.push(CollectStep::Gate(AbsorbedGate {
                            gate: standard,
                            qubits: body
                                .qargs_interner()
                                .get(gate.qubits)
                                .iter()
                                .map(|q| qubits[q.index()])
                                .collect(),
                            params,
                        }));
                        continue;
                    }
                    let local = emission_spec(gate).expect(
                        "a collector body holds only absorbed gates and absorbed local emissions",
                    );
                    steps.push(CollectStep::Local(LocalEmission {
                        qubits: body
                            .qargs_interner()
                            .get(gate.qubits)
                            .iter()
                            .map(|q| qubits[q.index()])
                            .collect(),
                        partition: local.partition.clone(),
                        parts: local.parts.clone(),
                    }));
                }
            }
            events.push(Event::Collector(infos.len()));
            infos.push(CollectorInfo {
                partition: spec.partition.clone(),
                qubits,
                synthesizer: spec.synthesizer(),
                param_indices: Vec::new(),
                steps,
            });
            continue;
        }

        if let Some(spec) = emission_spec(inst) {
            events.push(Event::Emission(spec, qubits));
            continue;
        }

        // A hard box is a grouping: inline it so its gates sit on the same spine.
        if let Some(body) = plain_box_body(src, inst)? {
            flatten(body, &qubits, events, infos, parameters)?;
            continue;
        }

        events.push(match inst.op.view() {
            OperationRef::StandardGate(gate) => Event::Gate(gate, qubits),
            OperationRef::StandardInstruction(StandardInstruction::Measure) => Event::Measure(
                qubits,
                src.cargs_interner()
                    .get(inst.clbits)
                    .iter()
                    .map(|c| c.index())
                    .collect(),
            ),
            OperationRef::StandardInstruction(StandardInstruction::Reset) => Event::Reset(qubits),
            _ => Event::Opaque,
        });
    }
    Ok(())
}

/// Wire one emission's path: every gate between it and its collector, chained per qubit.
#[allow(clippy::too_many_arguments)]
fn walk_emission(
    sg: &mut SamplingGraph,
    events: &[Event],
    from: usize,
    spec: &Emit,
    emission_qubits: &[usize],
    source: NodeIndex,
    target_index: usize,
    target_node: NodeIndex,
    infos: &[CollectorInfo],
    gate_nodes: &mut HashMap<GateKey, NodeIndex>,
    table: &DistributionTable,
) -> PyResult<()> {
    let qubits: HashSet<usize> = emission_qubits.iter().copied().collect();
    let mut frontier: HashMap<usize, NodeIndex> = qubits.iter().map(|q| (*q, source)).collect();
    let direction = spec.direction.expect(
        "a local emission never surfaces as a top-level Event::Emission — it lives inside its \
         collector's body",
    );
    let virtual_type = spec.virtual_type(table);

    // Walking in the emission's own direction is what makes propagation derivable rather than
    // recorded.
    let indices: Vec<usize> = match direction {
        Direction::Right => (from + 1..events.len()).collect(),
        Direction::Left => (0..from).rev().collect(),
    };

    for index in indices {
        match &events[index] {
            Event::Collector(collector) if *collector == target_index => break,
            Event::Collector(collector) => {
                // A foreign collector's absorbed gates are still real gates on this emission's
                // path, so they conjugate it, even though that collector also multiplies them into
                // its own layer.
                let absorbed: Vec<&AbsorbedGate> = infos[*collector].gates().collect();
                let order: Vec<usize> = match direction {
                    Direction::Right => (0..absorbed.len()).collect(),
                    Direction::Left => (0..absorbed.len()).rev().collect(),
                };
                for offset in order {
                    let gate = &absorbed[offset];
                    chain(
                        sg,
                        &mut frontier,
                        &qubits,
                        direction,
                        gate_nodes,
                        (index, offset),
                        gate.gate,
                        &gate.qubits,
                        virtual_type,
                    )?;
                }
            }
            Event::Gate(gate, gate_qubits) => chain(
                sg,
                &mut frontier,
                &qubits,
                direction,
                gate_nodes,
                (index, 0),
                *gate,
                gate_qubits,
                virtual_type,
            )?,
            _ => {}
        }
    }

    // Whatever each wire's virtual state ended up as is what the collector synthesizes.
    let ends: HashSet<NodeIndex> = frontier.values().copied().collect();
    for end in ends {
        sg.graph.add_edge(end, target_node, Edge::new());
    }
    Ok(())
}

/// Add or reuse the node for one gate and advance the frontier over its qubits.
#[allow(clippy::too_many_arguments)]
fn chain(
    sg: &mut SamplingGraph,
    frontier: &mut HashMap<usize, NodeIndex>,
    tracked: &HashSet<usize>,
    direction: Direction,
    gate_nodes: &mut HashMap<GateKey, NodeIndex>,
    occurrence: (usize, usize),
    gate: StandardGate,
    gate_qubits: &[usize],
    virtual_type: VirtualType,
) -> PyResult<()> {
    if !gate_qubits.iter().any(|q| tracked.contains(q)) {
        return Ok(());
    }
    // Refuse rather than emit a node that cannot be evaluated: conjugating this virtual type by
    // this gate leaves its group, so there is no rule to apply.
    if !propagates(virtual_type, gate) {
        return Err(PyValueError::new_err(format!(
            "cannot propagate a {} virtual gate through '{}': no propagation rule exists for that \
             combination, so the randomization could not be undone. Only Cliffords (and RZZ) admit \
             Pauli and local-C1 propagation; a local U2 element admits single-qubit gates only.",
            virtual_type.name(),
            gate.name(),
        )));
    }
    let key = (occurrence.0, occurrence.1, direction, virtual_type);
    let node = *gate_nodes.entry(key).or_insert_with(|| {
        // One joint subsystem: a conjugation by a multi-qubit gate mixes its qubits, so they can
        // only be evaluated together.
        sg.graph.add_node(Node::joint(
            gate_qubits.to_vec(),
            NodeKind::Propagate(Propagate { gate, direction }),
        ))
    });
    let predecessors: HashSet<NodeIndex> = gate_qubits
        .iter()
        .filter_map(|q| frontier.get(q).copied())
        .collect();
    for predecessor in predecessors {
        sg.graph.add_edge(predecessor, node, Edge::new());
    }
    for q in gate_qubits.iter().filter(|q| tracked.contains(*q)) {
        frontier.insert(*q, node);
    }
    Ok(())
}

/// The graph node for one emission, resolved from the table entry its `dist` key points at.
fn emission_kind(spec: &Emit, table: &DistributionTable) -> PyResult<NodeKind> {
    let entry = table.get(spec.dist()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "emission (dist={}) references a missing table entry",
            spec.dist().0
        ))
    })?;
    Ok(NodeKind::Emission(Emission {
        key: spec.dist(),
        direction: spec.direction.expect(
            "a local emission never surfaces as a top-level Event::Emission — it lives inside its \
             collector's body",
        ),
        virtual_type: entry.virtual_type(),
    }))
}
