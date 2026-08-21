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
//! Nothing here mutates its input. Two readers walk the same emission circuit — one writing the
//! template, one reading the spine — and each records the [`Site`] of every collector it sees, so
//! [`build_sampling_graph`] pairs a graph collector with its template parameter range by identity
//! rather than by the order the two walks happened to arrive in. A collector one reader sees and the
//! other does not is then a failed lookup rather than a shifted range. Inside a collector body the
//! traversal order must not be reported as circuit order; see
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

use hashbrown::HashMap;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::operations::StandardInstruction;

use crate::annotated_circuit::SynthesizerType;
use crate::distributions::DistributionTable;
use crate::emission_circuit::Emit;
use crate::emission_circuit_navigation::{
    Site, block_body, collect_annotation, emission_spec, is_box, is_emission,
};
use crate::error::IntoPyResult;
use crate::parameters::ParameterTable;
use crate::sampling_graph::{
    AbsorbedGate, AbsorbedParam, Collect, CollectStep, Emission, LocalEmission, Measure, Node,
    NodeKind, SamplingGraph,
};
use crate::spine::{self, Spine};

/// How many angles a synthesizer needs per qubit; both are three-angle Euler decompositions.
const PARAMS_PER_QUBIT: usize = 3;

/// Where one collector's angles live in the template's parameter vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectorParams {
    /// The collect box these angles were minted for. This is the join key: it is what lets
    /// [`build_sampling_graph`] find the range belonging to a given graph collector without relying
    /// on the two walks reporting their collectors in the same order.
    pub site: Site,
    /// The collector's qubits, ascending.
    pub qubits: Vec<usize>,
    pub synthesizer: SynthesizerType,
    /// Indices into the template's parameter vector, `qubits.len() * PARAMS_PER_QUBIT` of them,
    /// grouped per qubit in `qubits` order.
    pub param_indices: Vec<usize>,
}

/// Build the template circuit for an emission circuit.
///
/// Returns the template plus one [`CollectorParams`] per collector, each naming the collect box it
/// came from. The order is circuit order, but nothing downstream depends on that: the site is what
/// identifies a range.
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
        &[],
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
/// always reported in. At the top level the two coincide. `scope` is the path of box nodes descended
/// through to reach `src`, which is what lets a collector here be named by its [`Site`].
fn write_scope(
    src: &DAGCircuit,
    out: &mut CircuitData,
    scope: &[NodeIndex],
    frame: &[usize],
    global: &[usize],
    collectors: &mut Vec<CollectorParams>,
    next_param: &mut usize,
) -> PyResult<()> {
    // Topological order, which is the order parameters are minted in. `flatten` walks the same order,
    // but neither walk relies on that any more: each collector is reported under its own site.
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
                site: Site {
                    scope: scope.to_vec(),
                    node,
                },
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
                &descend(scope, node),
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

/// The scope one level down: the path to `src` extended by the box node being entered.
///
/// Both readers descend through exactly the same boxes, so the paths they build agree, and hence so
/// do the [`Site`]s they name their collectors by.
fn descend(scope: &[NodeIndex], node: NodeIndex) -> Vec<NodeIndex> {
    let mut inner = Vec::with_capacity(scope.len() + 1);
    inner.extend_from_slice(scope);
    inner.push(node);
    inner
}

/// The body of an unannotated box, or `None` if this is not one.
fn plain_box_body<'a>(
    src: &'a DAGCircuit,
    inst: &PackedInstruction,
) -> PyResult<Option<&'a DAGCircuit>> {
    if is_box(inst) {
        block_body(src, inst)
    } else {
        Ok(None)
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

/// Build the sampling graph for an emission circuit, and the parameter table it refers into.
///
/// `collectors` must come from [`build_template`] over the same circuit. Each is matched to the graph
/// collector standing at the same [`Site`], so the two walks agreeing on *which* collect boxes exist
/// is what is required of them, not agreeing on the order they report them in.
pub fn build_sampling_graph(
    dag: &DAGCircuit,
    table: &DistributionTable,
    collectors: &[CollectorParams],
) -> PyResult<(SamplingGraph, ParameterTable)> {
    let mut spine = Spine::default();
    let mut parameters = ParameterTable::new();
    let identity: Vec<usize> = (0..dag.num_qubits()).collect();
    flatten(dag, &[], &identity, &mut spine, &mut parameters)?;
    attach_param_indices(&mut spine.collectors, collectors)?;

    let mut sg = SamplingGraph::new();

    // Sinks first, so an emission's walk always has a node to terminate at.
    let mut collector_nodes = Vec::with_capacity(spine.collectors.len());
    for info in &spine.collectors {
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
    let mut gate_nodes: HashMap<spine::GateKey, NodeIndex> = HashMap::new();
    let mut emission_nodes: HashMap<usize, NodeIndex> = HashMap::new();

    for (position, item) in spine.items.iter().enumerate() {
        match item {
            spine::Item::Emission(spec, qubits) => {
                let node = sg.graph.add_node(Node::new(
                    qubits.clone(),
                    spec.partition.clone(),
                    emission_kind(spec, table)?,
                ));
                emission_nodes.insert(position, node);
            }
            spine::Item::Measure(qubits, clbits) => {
                sg.graph.add_node(Node::singletons(
                    qubits.clone(),
                    NodeKind::Measure(Measure {
                        clbit_indices: clbits.clone(),
                    }),
                ));
            }
            spine::Item::Reset(qubits) => {
                sg.graph
                    .add_node(Node::singletons(qubits.clone(), NodeKind::Reset));
            }
            _ => {}
        }
    }

    // Walk each emission to its target collector, wiring the conjugation chain in between.
    // Target resolution is purely positional: scan from the emission in its travel direction to
    // find the nearest compatible collector.
    for (position, item) in spine.items.iter().enumerate() {
        let spine::Item::Emission(spec, qubits) = item else {
            continue;
        };
        let source = emission_nodes[&position];
        // A local emission is resolved in place inside its collector's body, so it never reaches
        // the top-level spine; anything here is still travelling.
        let direction = spec.direction.expect(
            "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
             collector's body",
        );
        // Unreachable in well-formed IR2: build writes both of a box's collectors, so an emission
        // always has a compatible collector ahead of it. Reaching this means either the pairing was
        // broken between the two passes or a hand-built circuit has an emission nothing can collect,
        // which would otherwise show up as a randomization that is never undone — so it is reported
        // rather than skipped.
        let target = spine
            .resolve_collector(position, direction, spec, qubits, table)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "emission on qubits {qubits:?} travelling {direction:?} has no compatible \
                     collector ahead of it; its randomization could not be undone",
                ))
            })?;
        spine.propagate(
            &mut sg,
            position,
            spec,
            qubits,
            source,
            target,
            collector_nodes[target],
            &mut gate_nodes,
            table,
        )?;
    }
    Ok((sg, parameters))
}

/// Give each graph collector the parameter range the template minted for the same collect box.
///
/// The join is on [`Site`] and not on position. The two readings of the emission circuit are
/// deliberately different walks — the template keeps a content box and recurses into it, this side
/// inlines a hard box onto the spine — so a shared arrival order is a convention neither walk is in a
/// position to check, and comparing counts catches a collector that went missing but not one that
/// moved. Keyed on identity a move is simply resolved, and the failure that remains is a collect box
/// only one walk saw, which is reported: the alternative is a range landing on the wrong synth
/// template, which mis-randomizes a circuit that still executes and still round-trips.
fn attach_param_indices(
    collectors: &mut [spine::Collector],
    params: &[CollectorParams],
) -> PyResult<()> {
    let mut by_site: HashMap<&Site, &CollectorParams> = HashMap::with_capacity(params.len());
    for entry in params {
        // A site names one collect box, so two ranges under one site means a walk visited a node
        // twice — the join would have no defined answer, so it is refused rather than resolved.
        if by_site.insert(&entry.site, entry).is_some() {
            return Err(PyValueError::new_err(format!(
                "the template reported two parameter ranges for the collector at {:?}",
                entry.site
            )));
        }
    }
    for info in collectors.iter_mut() {
        let entry = by_site.remove(&info.site).ok_or_else(|| {
            PyValueError::new_err(format!(
                "the graph walk found a collector at {:?} that the template did not; the two must be \
                 built from the same circuit",
                info.site
            ))
        })?;
        info.param_indices = entry.param_indices.clone();
    }
    // The other direction: parameters minted for a collector no graph collector claimed. Those angles
    // would be in the template with nothing computing them.
    if let Some(site) = by_site.keys().next() {
        return Err(PyValueError::new_err(format!(
            "the template minted parameters for {} collector(s) the graph walk did not find, one at \
             {site:?}; the two must be built from the same circuit",
            by_site.len()
        )));
    }
    Ok(())
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

/// Read a scope onto the spine, inlining hard boxes and reducing each collector to one position.
///
/// This is the adapter between IR2 and [`Spine`], and the only place the DAG is read: everything the
/// propagation walk needs is flat data by the time it sees it.
fn flatten(
    src: &DAGCircuit,
    scope: &[NodeIndex],
    frame: &[usize],
    spine: &mut Spine,
    parameters: &mut ParameterTable,
) -> PyResult<()> {
    // `scope` is the path of box nodes descended through to reach `src`, so that a collector read out
    // here carries the same site `write_scope` gave it. It is the join key, and it is the reason this
    // walk and the template's are free to differ in every other respect — as they already do, this one
    // inlining a hard box where the other keeps it.
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
            spine.push_collector(spine::Collector {
                site: Site {
                    scope: scope.to_vec(),
                    node,
                },
                partition: spec.partition.clone(),
                qubits,
                synthesizer: spec.synthesizer(),
                param_indices: Vec::new(),
                steps,
            });
            continue;
        }

        if let Some(spec) = emission_spec(inst) {
            spine.items.push(spine::Item::Emission(spec, qubits));
            continue;
        }

        // A hard box is a grouping: inline it so its gates sit on the same spine.
        if let Some(body) = plain_box_body(src, inst)? {
            flatten(body, &descend(scope, node), &qubits, spine, parameters)?;
            continue;
        }

        spine.items.push(match inst.op.view() {
            OperationRef::StandardGate(gate) => spine::Item::Gate(gate, qubits),
            OperationRef::StandardInstruction(StandardInstruction::Measure) => {
                spine::Item::Measure(
                    qubits,
                    src.cargs_interner()
                        .get(inst.clbits)
                        .iter()
                        .map(|c| c.index())
                        .collect(),
                )
            }
            OperationRef::StandardInstruction(StandardInstruction::Reset) => {
                spine::Item::Reset(qubits)
            }
            _ => spine::Item::Opaque,
        });
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
            "a local emission never surfaces as a top-level Item::Emission — it lives inside its \
             collector's body",
        ),
        virtual_type: entry.virtual_type(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use hashbrown::HashSet;
    use qiskit_circuit::annotation::Annotation;

    use super::super::build::build;
    use crate::annotated_circuit::{DistributionType, Dressing, Twirl};
    use crate::emission_circuit_navigation::{append, new_dag_body, write_box};
    use crate::partition::Partition;

    // --- The join, on the two sides built by hand ------------------------------------------------
    //
    // None of these needs a circuit: a `Site` is a path of node indices and both sides of the join are
    // plain data, so what the join does with them can be pinned directly. Errors are only ever checked
    // with `is_err` — reading a `PyErr`'s message would normalize it, which does need Python.

    /// A site at `node`, reached through the boxes `scope` names.
    fn site(scope: &[usize], node: usize) -> Site {
        Site {
            scope: scope.iter().map(|index| NodeIndex::new(*index)).collect(),
            node: NodeIndex::new(node),
        }
    }

    /// One collector as the graph walk reports it, with no range attached yet.
    fn graph_collector(site: Site) -> spine::Collector {
        spine::Collector {
            site,
            qubits: vec![0],
            partition: Partition::singletons(1),
            synthesizer: SynthesizerType::RzSx,
            param_indices: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// One collector as the template reports it, carrying the range it minted.
    fn template_params(site: Site, param_indices: Vec<usize>) -> CollectorParams {
        CollectorParams {
            site,
            qubits: vec![0],
            synthesizer: SynthesizerType::RzSx,
            param_indices,
        }
    }

    #[test]
    fn test_a_collector_keeps_its_own_range_when_the_walks_disagree_on_order() {
        // This is the divergence a count comparison cannot see: the same number of collectors on both
        // sides, in opposite orders. Positionally the second collector would take the first one's
        // angles, mis-randomizing a circuit that still executes. Keyed on the site there is nothing to
        // get wrong — the reorder is resolved rather than carried through.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 7), vec![3, 4, 5]),
        ];
        let mut collectors = vec![graph_collector(site(&[], 7)), graph_collector(site(&[], 1))];
        attach_param_indices(&mut collectors, &template).unwrap();
        assert_eq!(collectors[0].param_indices, vec![3, 4, 5]);
        assert_eq!(collectors[1].param_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_the_same_node_index_in_two_scopes_is_two_collectors() {
        // Node indices are per-scope, so identity is the whole path and not its last step. A join keyed
        // on the node alone would fuse a nested collector with a top-level one.
        let template = vec![
            template_params(site(&[], 3), vec![0, 1, 2]),
            template_params(site(&[4], 3), vec![3, 4, 5]),
        ];
        let mut collectors = vec![
            graph_collector(site(&[4], 3)),
            graph_collector(site(&[], 3)),
        ];
        attach_param_indices(&mut collectors, &template).unwrap();
        assert_eq!(collectors[0].param_indices, vec![3, 4, 5]);
        assert_eq!(collectors[1].param_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_a_collector_only_the_graph_walk_found_is_refused() {
        // Nothing minted its angles, so there is no honest range to give it. Reported, never skipped: a
        // collector left with an empty range would synthesize from parameters the template does not
        // have.
        let template = vec![template_params(site(&[], 1), vec![0, 1, 2])];
        let mut collectors = vec![graph_collector(site(&[], 1)), graph_collector(site(&[], 9))];
        assert!(attach_param_indices(&mut collectors, &template).is_err());
    }

    #[test]
    fn test_a_collector_only_the_template_found_is_refused() {
        // The other direction: angles standing in the template with nothing in the graph computing
        // them.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 9), vec![3, 4, 5]),
        ];
        let mut collectors = vec![graph_collector(site(&[], 1))];
        assert!(attach_param_indices(&mut collectors, &template).is_err());
    }

    #[test]
    fn test_two_ranges_for_one_site_is_refused() {
        // A site names one collect box, so the join would have no defined answer here.
        let template = vec![
            template_params(site(&[], 1), vec![0, 1, 2]),
            template_params(site(&[], 1), vec![3, 4, 5]),
        ];
        let mut collectors = vec![graph_collector(site(&[], 1))];
        assert!(attach_param_indices(&mut collectors, &template).is_err());
    }

    // --- The two walks over a real emission circuit ----------------------------------------------

    fn twirl() -> Arc<dyn Annotation> {
        Arc::new(Twirl {
            distribution: DistributionType::UniformPauli,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    /// A twirled box holding a gate and a second twirled box, so lowering has to cross a boundary.
    ///
    /// Nesting is the case worth building, because it is where the two walks genuinely differ:
    /// `write_scope` keeps the content box and recurses into it, `flatten` inlines it onto the spine.
    fn nested_twirl() -> DAGCircuit {
        Python::initialize();
        let mut inner = new_dag_body(1, 0, 1).unwrap().into_builder();
        append(&mut inner, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();

        let mut outer = new_dag_body(1, 0, 2).unwrap().into_builder();
        append(&mut outer, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();
        write_box(
            &mut outer,
            inner.build(),
            vec![twirl()],
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();

        let mut out = new_dag_body(1, 0, 1).unwrap().into_builder();
        write_box(
            &mut out,
            outer.build(),
            vec![twirl()],
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();
        out.build()
    }

    #[test]
    fn test_both_walks_name_the_same_collectors_across_a_box_boundary() {
        // The claim the join rests on: the two readings descend through the same boxes, so a collector
        // has one site whichever walk found it.
        let (dag, _table) = build(&nested_twirl()).unwrap();
        let (_template, params) = build_template(&dag).unwrap();

        let mut spine = Spine::default();
        let mut parameters = ParameterTable::new();
        let identity: Vec<usize> = (0..dag.num_qubits()).collect();
        flatten(&dag, &[], &identity, &mut spine, &mut parameters).unwrap();

        let template_sites: HashSet<&Site> = params.iter().map(|c| &c.site).collect();
        let graph_sites: HashSet<&Site> = spine.collectors.iter().map(|c| &c.site).collect();
        assert_eq!(template_sites.len(), params.len(), "sites are unique");
        assert_eq!(template_sites, graph_sites);
        assert_eq!(
            params.len(),
            4,
            "build collects on both edges of both twirled boxes"
        );
        assert_eq!(
            template_sites
                .iter()
                .filter(|site| !site.scope.is_empty())
                .count(),
            2,
            "the inner twirl's pair is named through the content box it sits in"
        );
    }

    #[test]
    fn test_lowering_a_nested_circuit_gives_every_collector_its_own_range() {
        // End to end over the same circuit: the ranges the template minted are exactly the ranges the
        // graph's collect nodes ask for.
        let (dag, table) = build(&nested_twirl()).unwrap();
        let (_template, params) = build_template(&dag).unwrap();
        let (graph, _parameters) = build_sampling_graph(&dag, &table, &params).unwrap();

        let mut minted: Vec<&Vec<usize>> = params.iter().map(|c| &c.param_indices).collect();
        let mut bound: Vec<&Vec<usize>> = graph
            .graph
            .node_weights()
            .filter_map(|node| match &node.kind {
                NodeKind::Collect(collect) => Some(&collect.param_indices),
                _ => None,
            })
            .collect();
        minted.sort();
        bound.sort();
        assert_eq!(minted, bound);
        assert!(
            bound.iter().all(|range| range.len() == PARAMS_PER_QUBIT),
            "one qubit each, so three angles each: {bound:?}"
        );
    }
}
