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
//! template, one reading the track — and each records the [`Site`] of every collector it sees, so
//! [`build_sampling_graph`] pairs a graph collector with its template parameter range by identity
//! rather than by the order the two walks happened to arrive in. A collector one reader sees and the
//! other does not is then a failed lookup rather than a shifted range. Inside a collector body the
//! traversal order must not be reported as circuit order; see
//! [`Collect::steps`](crate::sampling_graph::Collect::steps).

use std::sync::Arc;

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::{Symbol, Value};

use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::operations::StandardInstruction;

use crate::annotated_circuit::SynthesizerType;
use crate::distributions::DistributionTable;
use crate::emission_circuit_navigation::{
    Site, block_body, collect_annotation, emission_spec, is_box, is_emission,
};
use crate::error::{Result, SamplexError};
use crate::parameters::ParameterTable;
use crate::sampling_graph::{
    AbsorbedGate, AbsorbedParam, CollectStep, LocalEmission, SamplingGraph,
};
use crate::track::{Collector, CollectorParams, Item, SamplingGraphBuilder, Track};

/// How many angles a synthesizer needs per qubit; both are three-angle Euler decompositions.
const PARAMS_PER_QUBIT: usize = 3;

/// Build the template circuit for an emission circuit.
///
/// Returns the template plus one [`CollectorParams`] per collector, each naming the collect box it
/// came from. The order is circuit order, but nothing downstream depends on that: the site is what
/// identifies a range.
pub fn build_template(dag: &DAGCircuit) -> Result<(CircuitData, Vec<CollectorParams>)> {
    let mut out = CircuitData::with_capacity(
        dag.num_qubits() as u32,
        dag.num_clbits() as u32,
        dag.num_ops(),
        Param::Float(0.0),
    )?;
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
) -> Result<()> {
    // Topological order, which is the order parameters are minted in. `flatten` walks the same order,
    // but neither walk relies on that any more: each collector is reported under its own site.
    for node in src.topological_op_nodes(false) {
        let inst = src.dag()[node].unwrap_operation();

        // A collector becomes the parametric fragment its angles drive.
        if let Some(collector) = collect_annotation(inst) {
            let locals = src.qargs_interner().get(inst.qubits);
            let written: Vec<usize> = locals.iter().map(|q| frame[q.index()]).collect();
            let qubits: Vec<usize> = locals.iter().map(|q| global[q.index()]).collect();
            let count = qubits.len() * PARAMS_PER_QUBIT;
            let param_indices: Vec<usize> = (*next_param..*next_param + count).collect();
            *next_param += count;

            write_synth_template(out, collector.synthesizer(), &written, &param_indices)?;
            collectors.push(CollectorParams {
                site: Site {
                    scope: scope.to_vec(),
                    node,
                },
                qubits,
                synthesizer: collector.synthesizer(),
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
            )?;
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
            )?;
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
) -> Result<()> {
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
            out.push_standard_gate(gate, &params, &target)?;
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
) -> Result<Option<&'a DAGCircuit>> {
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
) -> Result<()> {
    if !inst.blocks_view().is_empty() {
        return Err(SamplexError::BodyOnNonBox(inst.op.name().to_string()));
    }
    let params = (!inst.params_view().is_empty()).then(|| {
        qiskit_circuit::instruction::Parameters::Params(
            inst.params_view().iter().cloned().collect(),
        )
    });
    out.push_packed_operation(inst.op.clone(), params, qargs, cargs)?;
    Ok(())
}

// --- Sampling graph construction ----------------------------------------------------------------
//
// The template says *what to execute*; the graph says *how to compute the angles*.

/// Build the sampling graph for an emission circuit, and the parameter table it refers into.
///
/// `collectors` must come from [`build_template`] over the same circuit. Each is matched to the graph
/// collector standing at the same [`Site`], so the two walks agreeing on *which* collect boxes exist
/// is what is required of them, not agreeing on the order they report them in.
///
/// An adapter and nothing else: [`flatten`] reads the circuit into a [`Track`] and
/// [`SamplingGraphBuilder`] takes it from there over flat data. The parameter table comes back
/// untouched by the graph — it is minted by the reading, since the reading is what sees the absorbed
/// angles.
pub fn build_sampling_graph(
    dag: &DAGCircuit,
    table: &DistributionTable,
    collectors: &[CollectorParams],
) -> Result<(SamplingGraph, ParameterTable)> {
    let mut track = Track::default();
    let mut parameters = ParameterTable::new();
    let identity: Vec<usize> = (0..dag.num_qubits()).collect();
    flatten(dag, &[], &identity, &mut track, &mut parameters)?;
    track.attach_param_indices(collectors)?;
    Ok((SamplingGraphBuilder::new(track, table).build()?, parameters))
}

/// Read one absorbed gate parameter, interning it only if it is genuinely symbolic.
///
/// A fully bound expression folds to [`AbsorbedParam::Bound`]. A [`Param::Obj`] is refused
/// outright.
fn absorbed_param(table: &mut ParameterTable, param: &Param) -> Result<AbsorbedParam> {
    match param {
        Param::Float(value) => Ok(AbsorbedParam::Bound(*value)),
        Param::ParameterExpression(expr) if expr.num_symbols() == 0 => {
            match expr.try_to_value(true)? {
                Value::Real(value) => Ok(AbsorbedParam::Bound(value)),
                Value::Int(value) => Ok(AbsorbedParam::Bound(value as f64)),
                // Reported rather than silently projected onto the real axis: a complex angle is
                // not something a rotation can be given, so it means the circuit was already wrong.
                Value::Complex(value) => Err(SamplexError::ComplexAbsorbedAngle(value)),
            }
        }
        Param::ParameterExpression(expr) => Ok(AbsorbedParam::Symbolic(table.intern(expr.clone()))),
        Param::Obj(_) => Err(SamplexError::OpaqueAbsorbedParameter),
    }
}

/// Read a scope onto the track, dissolving every box but a collector and reducing each collector to
/// one position.
///
/// This is the adapter between IR2 and [`Track`], and the only place the DAG is read: everything the
/// propagation walk needs is flat data by the time it sees it.
fn flatten(
    src: &DAGCircuit,
    scope: &[NodeIndex],
    frame: &[usize],
    track: &mut Track,
    parameters: &mut ParameterTable,
) -> Result<()> {
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

        if let Some(collector) = collect_annotation(inst) {
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
                            .collect::<Result<Vec<_>>>()?;
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
            track.push_collector(Collector {
                site: Site {
                    scope: scope.to_vec(),
                    node,
                },
                partition: collector.partition.clone(),
                qubits,
                synthesizer: collector.synthesizer(),
                param_indices: Vec::new(),
                steps,
            });
            continue;
        }

        if let Some(emission) = emission_spec(inst) {
            track.push_item(Item::Emission(emission, qubits));
            continue;
        }

        // Any box left by here is a grouping and nothing more — a collector and an emission have
        // both already been handled, so this catches a content box as well as a hard one. Dissolve
        // it, so its gates sit on the same track as the gates that surrounded it.
        if let Some(body) = plain_box_body(src, inst)? {
            flatten(body, &descend(scope, node), &qubits, track, parameters)?;
            continue;
        }

        track.push_item(match inst.op.view() {
            OperationRef::StandardGate(gate) => Item::Gate(gate, qubits),
            OperationRef::StandardInstruction(StandardInstruction::Measure) => Item::Measure(
                qubits,
                src.cargs_interner()
                    .get(inst.clbits)
                    .iter()
                    .map(|c| c.index())
                    .collect(),
            ),
            OperationRef::StandardInstruction(StandardInstruction::Reset) => Item::Reset(qubits),
            _ => Item::Opaque,
        });
    }
    Ok(())
}
