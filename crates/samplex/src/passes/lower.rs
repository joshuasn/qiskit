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

//! Lower: emission circuit (IR2) → template circuit.
//!
//! Each collect box becomes a *synth template* — the fixed parametric fragment whose angles the
//! sampling graph will fill in. The absorbed gates in its body are discarded, because they are folded
//! into those angles rather than executed separately; that is the whole point of having absorbed them.
//! `Emit` instructions are markers and disappear. Hard boxes were only a grouping, so their content is
//! flattened out.
//!
//! **Parameters are minted here and nowhere earlier.** Merging changes how many collectors exist and
//! how wide they are, so a label assigned before `merge_collectors` would be invalidated by it — on the
//! notebook circuit, six collectors and 48 parameters become four and 36. The ordering constraint is
//! asymmetric: lowering *unmerged* IR2 is correct, just suboptimal (more dressing layers than needed);
//! what is invalid is merging after lowering. So merging stays a switchable optimization.
//!
//! Alongside the template this returns each collector's parameter range, which is what the sampling
//! graph's `Collect` nodes need in order to know which angles they are responsible for.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;

use hashbrown::{HashMap, HashSet};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::operations::StandardInstruction;

use super::utils::{
    IntoPyResult, block_body, collect_annotation, copy_through, emission_spec, is_emission,
};
use crate::annotated_circuit::SynthesizerType;
use crate::distributions::{DistEntry, DistributionTable};
use crate::emission_circuit::{CollectItem, EmitSource, EmitSpec};
use crate::partition::Partition;
use crate::virtual_flow_graph::{
    AbsorbedGate, Collect, CollectStep, Direction, Edge, Emission, Measure, Node, NodeKind,
    Propagate, VirtualFlowGraph,
};
use crate::virtual_type::{VirtualType, propagates};

/// How many angles a synthesizer needs per qubit. Both supported synthesizers are three-angle Euler
/// decompositions.
const PARAMS_PER_QUBIT: usize = 3;

/// Where one collector's angles live in the template's parameter vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectorParams {
    /// The collector's qubits, ascending.
    pub qubits: Vec<usize>,
    pub synthesizer: SynthesizerType,
    /// The emissions this collector consumes, as recorded on its annotation.
    pub collects: Vec<u32>,
    /// Indices into the template's parameter vector, `qubits.len() * PARAMS_PER_QUBIT` of them,
    /// grouped per qubit in `qubits` order.
    pub param_indices: Vec<usize>,
}

/// Build the template circuit for an emission circuit.
///
/// Returns the template plus one [`CollectorParams`] per collector, in circuit order.
pub fn build_template(
    py: Python,
    circuit: &CircuitData,
) -> PyResult<(CircuitData, Vec<CollectorParams>)> {
    let mut out = CircuitData::with_capacity(
        circuit.num_qubits() as u32,
        circuit.num_clbits() as u32,
        circuit.len(),
        Param::Float(0.0),
    )
    .into_py_result()?;
    let mut collectors = Vec::new();
    let mut next_param = 0usize;

    // The identity frame: at the top level a scope-local qubit is already a circuit qubit.
    let identity: Vec<usize> = (0..circuit.num_qubits()).collect();
    write_scope(
        py,
        circuit,
        &mut out,
        &identity,
        &mut collectors,
        &mut next_param,
    )?;
    Ok((out, collectors))
}

/// One collector as seen from Python: qubits, synthesizer name, emissions collected, parameter indices.
type CollectorSummary = (Vec<usize>, String, Vec<u32>, Vec<usize>);

/// Python-facing entry point: returns the template and the collector parameter map.
#[pyfunction]
#[pyo3(name = "build_template")]
pub fn py_build_template(
    py: Python,
    circuit: &PyCircuitData,
) -> PyResult<(PyCircuitData, Vec<CollectorSummary>)> {
    let (template, collectors) = build_template(py, &circuit.inner)?;
    let summary = collectors
        .into_iter()
        .map(|c| {
            let synth = match c.synthesizer {
                SynthesizerType::RzSx => "rzsx".to_string(),
                SynthesizerType::RzRx => "rzrx".to_string(),
            };
            (c.qubits, synth, c.collects, c.param_indices)
        })
        .collect();
    Ok((PyCircuitData { inner: template }, summary))
}

/// Lower an emission circuit into its executable pair: template circuit and sampling graph.
///
/// Both are read off the same IR2 circuit, so the graph's parameter ranges are exactly the ones the
/// template minted.
#[pyfunction]
#[pyo3(name = "lower")]
pub fn py_lower(
    py: Python,
    circuit: &PyCircuitData,
    table: &DistributionTable,
) -> PyResult<(PyCircuitData, VirtualFlowGraph)> {
    let (template, collectors) = build_template(py, &circuit.inner)?;
    let graph = build_sampling_graph(py, &circuit.inner, table, &collectors)?;
    Ok((PyCircuitData { inner: template }, graph))
}

/// Emit one scope's worth of template content. `frame` maps scope-local qubits to circuit qubits.
fn write_scope(
    py: Python,
    src: &CircuitData,
    out: &mut CircuitData,
    frame: &[usize],
    collectors: &mut Vec<CollectorParams>,
    next_param: &mut usize,
) -> PyResult<()> {
    for inst in src.data() {
        // A collector becomes the parametric fragment its angles drive.
        if let Some(spec) = collect_annotation(py, inst) {
            let qubits: Vec<usize> = src
                .qargs_interner()
                .get(inst.qubits)
                .iter()
                .map(|q| frame[q.index()])
                .collect();
            let count = qubits.len() * PARAMS_PER_QUBIT;
            let param_indices: Vec<usize> = (*next_param..*next_param + count).collect();
            *next_param += count;

            write_synth_template(out, spec.synthesizer, &qubits, &param_indices)?;
            collectors.push(CollectorParams {
                qubits,
                synthesizer: spec.synthesizer,
                collects: spec.incoming_ids(),
                param_indices,
            });
            continue;
        }

        // Emissions are markers for the sampling graph; they are not executable.
        if is_emission(py, inst) {
            continue;
        }

        // A hard box was a grouping, so flatten it — recursing so nested collectors are lowered too.
        if let Some(body) = plain_box_body(src, inst)? {
            let inner: Vec<usize> = src
                .qargs_interner()
                .get(inst.qubits)
                .iter()
                .map(|q| frame[q.index()])
                .collect();
            write_scope(py, body, out, &inner, collectors, next_param)?;
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
        copy_with_qargs(src, inst, out, &qargs, &cargs)?;
    }
    Ok(())
}

/// Write the parametric fragment for one collector, on each of its qubits.
///
/// `RzSx` is `rz sx rz sx rz` and `RzRx` is `rz rx rz` — both three angles, matching samplomatic's
/// synthesizers.
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

/// Mint a template parameter.
///
/// Names are zero-padded so that lexicographic order matches numeric order, matching samplomatic's
/// `ParamIter`. Each carries a fresh uuid, so parameters from two runs share names and indices but are
/// not equal objects — the same semantics as Python's `Parameter`.
fn fresh_parameter(index: usize) -> Param {
    let symbol = Symbol::standalone(format!("p{index:04}"), None);
    Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(symbol)))
}

/// The body of an unannotated box, or `None` if this is not one.
fn plain_box_body<'a>(
    src: &'a CircuitData,
    inst: &PackedInstruction,
) -> PyResult<Option<&'a CircuitData>> {
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

/// Copy an instruction with explicitly remapped bits. Flattening means qargs change, so
/// [`copy_through`] cannot be reused directly.
fn copy_with_qargs(
    src: &CircuitData,
    inst: &PackedInstruction,
    out: &mut CircuitData,
    qargs: &[Qubit],
    cargs: &[qiskit_circuit::Clbit],
) -> PyResult<()> {
    if inst.blocks_view().is_empty() {
        let params = (!inst.params_view().is_empty()).then(|| {
            qiskit_circuit::instruction::Parameters::Params(
                inst.params_view().iter().cloned().collect(),
            )
        });
        return out
            .push_packed_operation(inst.op.clone(), params, qargs, cargs)
            .into_py_result();
    }
    // Control flow other than `box` never reaches here — build rejects it.
    copy_through(src, inst, out, None)
}

// --- Sampling graph construction ----------------------------------------------------------------
//
// The template says *what to execute*; the graph says *how to compute the angles*. Both are read off
// the same IR2 circuit, so they agree by construction rather than by convention.

/// One collector, flattened out of the circuit.
struct CollectorInfo {
    qubits: Vec<usize>,
    synthesizer: SynthesizerType,
    collects: Vec<u32>,
    param_indices: Vec<usize>,
    /// Everything this collector composes, in circuit order — emissions and absorbed gates interleaved.
    steps: Vec<CollectStep>,
}

impl CollectorInfo {
    /// The absorbed gates alone, in circuit order. Used by an *enclosing* emission crossing this
    /// collector, which conjugates by the gates and ignores what the collector consumes.
    fn gates(&self) -> impl Iterator<Item = &AbsorbedGate> {
        crate::virtual_flow_graph::collect_step_gates(&self.steps)
    }
}

/// The circuit as a flat sequence, which is what makes the propagation walk a simple scan.
enum Event {
    Emission(EmitSpec),
    Collector(usize),
    Gate(StandardGate, Vec<usize>),
    Measure(Vec<usize>, Vec<usize>),
    Reset(Vec<usize>),
    /// A real operation with no virtual effect — a barrier, say. It still blocks nothing, but it is
    /// kept so positions line up with the template.
    Opaque,
}

/// What identifies one conjugation node: a gate occurrence together with the flow crossing it.
///
/// The occurrence is `(event position, offset)`, the offset being the position within a collector's
/// absorbed run — zero for a gate that stands on its own.
type GateKey = (usize, usize, Direction, VirtualType);

/// Build the sampling graph for an emission circuit.
///
/// `collectors` comes from [`build_template`], so the graph's `Collect` nodes carry exactly the
/// parameter ranges the template minted.
pub fn build_sampling_graph(
    py: Python,
    circuit: &CircuitData,
    table: &DistributionTable,
    collectors: &[CollectorParams],
) -> PyResult<VirtualFlowGraph> {
    let mut events = Vec::new();
    let mut infos = Vec::new();
    let identity: Vec<usize> = (0..circuit.num_qubits()).collect();
    flatten(py, circuit, &identity, &mut events, &mut infos)?;

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

    let mut vfg = VirtualFlowGraph::new();

    // Sinks first, so an emission's walk always has a node to terminate at.
    let mut collector_nodes = Vec::with_capacity(infos.len());
    for info in &infos {
        collector_nodes.push(vfg.graph.add_node(Node {
            partition: Partition::from_elements(info.qubits.iter().copied()),
            kind: NodeKind::Collect(Collect {
                synthesizer: info.synthesizer,
                param_indices: info.param_indices.clone(),
                steps: info.steps.clone(),
            }),
        }));
    }

    // One Propagate node per *conjugation*, created lazily and shared by the emissions for which it is
    // the same conjugation. The key is the gate occurrence — (event position, offset within a
    // collector's absorbed run) — plus the handedness and virtual type of the flow crossing it.
    //
    // Direction and type belong in the key because they change what the node computes: conjugating a
    // Pauli by CX leftward and rightward are different operations, as are conjugating a Pauli and a
    // local C1 by the same gate. Sharing across them would fuse operations that cannot be evaluated as
    // one. Both cases are reachable — an outer right-walking factor and an inner left-walking factor
    // cross the same nested hard gates in opposite directions.
    let mut gate_nodes: HashMap<GateKey, NodeIndex> = HashMap::new();
    let mut emission_nodes: HashMap<u32, NodeIndex> = HashMap::new();

    for (position, event) in events.iter().enumerate() {
        match event {
            Event::Emission(spec) => {
                let node = vfg.graph.add_node(Node {
                    partition: spec.partition.clone(),
                    kind: emission_kind(spec, table)?,
                });
                emission_nodes.insert(spec.id, node);
                let _ = position;
            }
            Event::Measure(qubits, clbits) => {
                vfg.graph.add_node(Node {
                    partition: Partition::from_elements(qubits.iter().copied()),
                    kind: NodeKind::Measure(Measure {
                        clbit_indices: clbits.clone(),
                    }),
                });
            }
            Event::Reset(qubits) => {
                vfg.graph.add_node(Node {
                    partition: Partition::from_elements(qubits.iter().copied()),
                    kind: NodeKind::Reset,
                });
            }
            _ => {}
        }
    }

    // Now walk each emission to the collector that names it, wiring the conjugations in between.
    // Emissions with no collector (standalone, awaiting walk_emissions) are skipped.
    for (position, event) in events.iter().enumerate() {
        let Event::Emission(spec) = event else {
            continue;
        };
        let source = emission_nodes[&spec.id];
        // First try: find via Incoming(id) on a collector (absorb_emissions has run).
        // Fallback: scan for nearest collector in direction (unoptimized path).
        let target = infos
            .iter()
            .position(|info| info.collects.contains(&spec.id))
            .or_else(|| {
                scan_for_nearest_collector(&events, position, spec.direction, &infos)
            });
        let Some(target) = target else {
            continue;
        };
        walk_emission(
            &mut vfg,
            &events,
            position,
            spec,
            source,
            target,
            collector_nodes[target],
            &infos,
            &mut gate_nodes,
        )?;
    }
    Ok(vfg)
}

/// Scan from position `start` in `direction` through the events list to find the nearest collector.
/// Used as a fallback when absorb_emissions hasn't run (unoptimized path).
fn scan_for_nearest_collector(
    events: &[Event],
    start: usize,
    direction: Direction,
    _infos: &[CollectorInfo],
) -> Option<usize> {
    let range: Box<dyn Iterator<Item = usize>> = match direction {
        Direction::Right => Box::new((start + 1)..events.len()),
        Direction::Left => Box::new((0..start).rev()),
    };
    for i in range {
        if let Event::Collector(idx) = &events[i] {
            return Some(*idx);
        }
    }
    None
}

/// Flatten a scope into events, inlining hard boxes and reducing each collector to one event.
fn flatten(
    py: Python,
    src: &CircuitData,
    frame: &[usize],
    events: &mut Vec<Event>,
    infos: &mut Vec<CollectorInfo>,
) -> PyResult<()> {
    for inst in src.data() {
        let qubits: Vec<usize> = src
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|q| frame[q.index()])
            .collect();

        if let Some(spec) = collect_annotation(py, inst) {
            // A collector's body is its absorbed gates; they stay with it rather than becoming
            // separate events, because the collector owns them.
            let mut gates = Vec::new();
            if let Some(body) = block_body(src, inst)? {
                for gate in body.data() {
                    if let OperationRef::StandardGate(standard) = gate.op.view() {
                        gates.push(AbsorbedGate {
                            gate: standard,
                            qubits: body
                                .qargs_interner()
                                .get(gate.qubits)
                                .iter()
                                .map(|q| qubits[q.index()])
                                .collect(),
                        });
                    }
                }
            }
            // The annotation's `Gates` counts say where the body sits among the emissions, so this is
            // where the two halves are woven back together. A mismatch means the annotation and the
            // body disagree, which no pass should be able to produce.
            if gates.len() != spec.gate_count() {
                return Err(PyValueError::new_err(format!(
                    "a collector's annotation accounts for {} absorbed gates but its body holds {}",
                    spec.gate_count(),
                    gates.len()
                )));
            }
            let mut cursor = 0usize;
            let mut steps = Vec::with_capacity(spec.items.len());
            for item in &spec.items {
                match item {
                    CollectItem::Emission(local) => {
                        steps.push(CollectStep::Local(local.clone()));
                    }
                    CollectItem::Incoming(id) => steps.push(CollectStep::Incoming(*id)),
                    CollectItem::Gates(count) => {
                        for gate in &gates[cursor..cursor + count] {
                            steps.push(CollectStep::Gate(gate.clone()));
                        }
                        cursor += count;
                    }
                }
            }
            events.push(Event::Collector(infos.len()));
            infos.push(CollectorInfo {
                qubits,
                synthesizer: spec.synthesizer,
                collects: spec.incoming_ids(),
                param_indices: Vec::new(),
                steps,
            });
            continue;
        }

        if let Some(spec) = emission_spec(py, inst) {
            events.push(Event::Emission(spec));
            continue;
        }

        // A hard box is a grouping: inline it so its gates sit on the same spine.
        if let Some(body) = plain_box_body(src, inst)? {
            flatten(py, body, &qubits, events, infos)?;
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
    vfg: &mut VirtualFlowGraph,
    events: &[Event],
    from: usize,
    spec: &EmitSpec,
    source: NodeIndex,
    target_index: usize,
    target_node: NodeIndex,
    infos: &[CollectorInfo],
    gate_nodes: &mut HashMap<GateKey, NodeIndex>,
) -> PyResult<()> {
    let qubits: HashSet<usize> = spec.partition.all_elements().iter().copied().collect();
    let mut frontier: HashMap<usize, NodeIndex> = qubits.iter().map(|q| (*q, source)).collect();

    // Walking in the emission's own direction is what makes propagation derivable rather than
    // recorded: everything crossed on the way to its collector conjugates it.
    let indices: Vec<usize> = match spec.direction {
        Direction::Right => (from + 1..events.len()).collect(),
        Direction::Left => (0..from).rev().collect(),
    };

    for index in indices {
        match &events[index] {
            Event::Collector(collector) if *collector == target_index => break,
            Event::Collector(collector) => {
                // A *foreign* collector's absorbed gates are still real gates on this emission's
                // path, so they conjugate it — even though that collector owns them for its own
                // multiplication. The two roles are independent.
                let absorbed: Vec<&AbsorbedGate> = infos[*collector].gates().collect();
                let order: Vec<usize> = match spec.direction {
                    Direction::Right => (0..absorbed.len()).collect(),
                    Direction::Left => (0..absorbed.len()).rev().collect(),
                };
                for offset in order {
                    let gate = &absorbed[offset];
                    chain(
                        vfg,
                        &mut frontier,
                        &qubits,
                        spec.direction,
                        gate_nodes,
                        (index, offset),
                        gate.gate,
                        &gate.qubits,
                        spec.virtual_type,
                    )?;
                }
            }
            Event::Gate(gate, gate_qubits) => chain(
                vfg,
                &mut frontier,
                &qubits,
                spec.direction,
                gate_nodes,
                (index, 0),
                *gate,
                gate_qubits,
                spec.virtual_type,
            )?,
            _ => {}
        }
    }

    // Whatever each wire's virtual state ended up as is what the collector synthesizes.
    let ends: HashSet<NodeIndex> = frontier.values().copied().collect();
    for end in ends {
        vfg.graph.add_edge(end, target_node, Edge::new());
    }
    Ok(())
}

/// Add or reuse the node for one gate and advance the frontier over its qubits.
#[allow(clippy::too_many_arguments)]
fn chain(
    vfg: &mut VirtualFlowGraph,
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
    // Refuse rather than emit a node that cannot be evaluated: conjugating this virtual type by this
    // gate leaves its group, so there is no rule to apply.
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
        vfg.graph.add_node(Node {
            partition: Partition::with_parts(std::iter::once(
                gate_qubits.to_vec().into_boxed_slice(),
            ))
            .unwrap(),
            kind: NodeKind::Propagate(Propagate { gate, direction }),
        })
    });
    let predecessors: HashSet<NodeIndex> = gate_qubits
        .iter()
        .filter_map(|q| frontier.get(q).copied())
        .collect();
    for predecessor in predecessors {
        vfg.graph.add_edge(predecessor, node, Edge::new());
    }
    for q in gate_qubits.iter().filter(|q| tracked.contains(*q)) {
        frontier.insert(*q, node);
    }
    Ok(())
}

/// The graph node for one emission.
///
/// A direct mapping, now that twirls, basis changes and noise injections are one node kind: the table
/// entry travels onto the node as-is and its discriminant *is* the source tag. All this has left to do
/// is check that the tag IR2 recorded and the entry it points at still agree, which they only would
/// not if IR2 were built inconsistently.
fn emission_kind(spec: &EmitSpec, table: &DistributionTable) -> PyResult<NodeKind> {
    let entry = table.get(spec.dist).ok_or_else(|| {
        PyValueError::new_err(format!(
            "emission #{} references a missing table entry",
            spec.id
        ))
    })?;
    let agrees = matches!(
        (spec.source, entry),
        (EmitSource::Twirl, DistEntry::Distribution(_))
            | (EmitSource::ChangeBasis, DistEntry::Basis { .. })
            | (EmitSource::InjectNoise, DistEntry::Noise { .. })
    );
    if !agrees {
        return Err(PyValueError::new_err(format!(
            "emission #{} does not match its table entry",
            spec.id
        )));
    }
    Ok(NodeKind::Emission(Emission {
        id: spec.id,
        entry: entry.clone(),
        direction: spec.direction,
        virtual_type: spec.virtual_type,
    }))
}
