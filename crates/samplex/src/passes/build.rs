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

//! Build pass: annotated circuit (IR1) → emission circuit (IR2).
//!
//! Turns each annotated box into `Emit` instructions plus the collect boxes that consume them, and a
//! box holding the gates virtual state is conjugated by. Also produces the [`DistributionTable`] those
//! emissions reference.
//!
//! **This pass is purely local.** Every annotated box yields exactly two collect boxes, one per side,
//! consuming only its own emissions and absorbing only its own easy gates. There is no cross-box
//! state: no qubit-to-collector map, no detach logic, no shared collectors. Widening those collectors
//! and fusing adjacent ones is `merge_collectors`' job. The consequence here is that a collect box is
//! *complete* the moment it is emitted, so this is a single forward sweep with no deferred buffer,
//! appending to a [`DAGCircuitBuilder`] rather than revisiting anything it has written.
//!
//! Parameters are deliberately absent: merging changes how many collectors exist and how wide they
//! are, so labelling happens in the IR2 → IR3 lowering. See `SAMPLEX_IR_DESIGN.md`.

use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeIndex};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    ControlFlow, ControlFlowInstruction, ControlFlowView, Operation, OperationRef, Param,
    PyInstruction, PyOpKind,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};

use crate::annotated_circuit::{
    BasisOrigin, BoxAnnotation, Dressing, ResolvedBox, SynthesizerType, extract_annotation,
    resolve_annotations,
};
use crate::distributions::{DistEntry, DistKey, DistributionTable};
use crate::emission_circuit::{
    Collect, CollectPart, CollectSpec, Emit, EmitPart, EmitSource, EmitSpec,
};
use crate::partition::Partition;
use crate::virtual_flow_graph::Direction;
use crate::virtual_type::VirtualType;

use super::utils::{IntoPyResult, append, new_dag_body};

/// The synthesizer assumed when a box's annotations do not name one.
///
/// Only `InjectLocalClifford` leaves this open — it has no `decomposition` field. Every other
/// annotation defaults to `rzsx`, so that is the assumption here too.
const DEFAULT_SYNTHESIZER: SynthesizerType = SynthesizerType::RzSx;

/// How deeply an emission nests inside its box, `0` being immediately against the hard content.
///
/// This is the ordering the annotation vocabulary implies, and it is what fixes both the spine order and
/// each collector's composition order:
///
/// - A **twirl** *is* the easy/hard boundary, so its pair is innermost.
/// - An **injection** — noise, or a local Clifford — happens *to the hard content*, so it sits just
///   outside the twirl point. `InjectLocalClifford` belongs here rather than with `ChangeBasis` despite
///   resolving to the same `ResolvedBasis`; that is what [`BasisOrigin`] records.
/// - A **basis change** applies to the box as a whole, so it is outermost — outside even the easy gates
///   the dressing absorbed.
///
/// For a left-dressed box with all of them, the spine reads
/// `collector, basis start, [easy gates], injections before + twirl, hard, injections after, basis end,
/// collector`.
const DEPTH_TWIRL: u8 = 0;
const DEPTH_INJECTION: u8 = 1;
const DEPTH_BASIS: u8 = 2;

/// An emission together with where it goes on its box's spine.
struct Placed {
    spec: EmitSpec,
    /// Which side of the hard box it is written on.
    edge: Direction,
    /// Distance from the hard content; see the `DEPTH_*` constants.
    depth: u8,
}

/// How a walked scope's qubits map outward.
///
/// A scope is always walked alongside an output circuit of the same width, so a scope-local index is
/// also an index into that output. `global` is what the emissions record, since the sampling graph
/// works in the circuit's own frame rather than any box's.
struct Scope<'a> {
    /// Scope-local qubit → index in the output being written.
    qubits: &'a [usize],
    /// Scope-local qubit → global circuit qubit.
    global: &'a [usize],
    /// Scope-local clbit → index in the output being written.
    clbits: &'a [usize],
}

impl Scope<'_> {
    fn out_qubits(&self, locals: &[Qubit]) -> PyResult<Vec<Qubit>> {
        locals
            .iter()
            .map(|q| {
                self.qubits
                    .get(q.index())
                    .map(|&i| Qubit(i as u32))
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("qubit {} out of scope", q.index()))
                    })
            })
            .collect()
    }

    fn out_clbits(&self, locals: &[Clbit]) -> PyResult<Vec<Clbit>> {
        locals
            .iter()
            .map(|c| {
                self.clbits
                    .get(c.index())
                    .map(|&i| Clbit(i as u32))
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("clbit {} out of scope", c.index()))
                    })
            })
            .collect()
    }

    fn global_qubits(&self, locals: &[Qubit]) -> PyResult<Vec<usize>> {
        locals
            .iter()
            .map(|q| {
                self.global.get(q.index()).copied().ok_or_else(|| {
                    PyValueError::new_err(format!("qubit {} out of scope", q.index()))
                })
            })
            .collect()
    }
}

struct Build {
    table: DistributionTable,
    draw_counts: HashMap<DistKey, u32>,
    /// The next unused box id. Counts *emitting* boxes only, in the order they are lowered.
    next_box_id: u32,
}

/// Build the emission circuit for an annotated circuit.
#[pyfunction]
#[pyo3(name = "build_lowered")]
pub fn py_build(py: Python, dag: &DAGCircuit) -> PyResult<(DAGCircuit, DistributionTable)> {
    build(py, dag)
}

/// Build the emission circuit for an annotated circuit.
pub fn build(py: Python, dag: &DAGCircuit) -> PyResult<(DAGCircuit, DistributionTable)> {
    crate::emission_circuit::ensure_registered(py)?;
    let num_qubits = dag.num_qubits();
    let num_clbits = dag.num_clbits();
    let identity_q: Vec<usize> = (0..num_qubits).collect();
    let identity_c: Vec<usize> = (0..num_clbits).collect();

    // IR2 is over the same wires as IR1, so the output keeps the input's bits and registers; only
    // the instructions change. Blocks are dropped because every body here is built fresh.
    let mut out = dag
        .copy_empty_like_with_capacity(dag.num_ops(), 0, VarsMode::Alike, BlocksMode::Drop)
        .into_builder();
    let mut build = Build {
        table: DistributionTable::new(),
        draw_counts: HashMap::new(),
        next_box_id: 0,
    };
    let scope = Scope {
        qubits: &identity_q,
        global: &identity_q,
        clbits: &identity_c,
    };
    build.walk(py, dag, &mut out, &scope)?;
    for (key, count) in &build.draw_counts {
        build.table.set_draw_count(*key, *count);
    }
    Ok((out.build(), build.table))
}

impl Build {
    /// Claim an id for one emitting box, naming the pairing between its emissions and its collectors.
    fn alloc_box_id(&mut self) -> u32 {
        let id = self.next_box_id;
        self.next_box_id += 1;
        id
    }

    /// Allocate `count` consecutive draw slots for `dist`, returning the start index.
    fn alloc_draws(&mut self, dist: DistKey, count: u32) -> u32 {
        let next = self.draw_counts.entry(dist).or_insert(0);
        let start = *next;
        *next += count;
        start
    }

    /// Emit the IR2 form of every op in `dag` into `out`.
    fn walk(
        &mut self,
        py: Python,
        dag: &DAGCircuit,
        out: &mut DAGCircuitBuilder,
        scope: &Scope,
    ) -> PyResult<()> {
        for node in dag.topological_op_nodes(false) {
            let inst = dag.dag()[node].unwrap_operation();
            match inst.op.view() {
                OperationRef::ControlFlow(cf) => {
                    if !matches!(cf.control_flow, ControlFlow::Box { .. }) {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported control flow in a samplex circuit: '{}'. Only `box` is \
                             supported; see the control-flow section of SAMPLEX_IR_DESIGN.md.",
                            cf.name()
                        )));
                    }
                    self.walk_box(py, dag, inst, out, scope)?;
                }
                _ => copy_instruction(dag, inst, out, scope)?,
            }
        }
        Ok(())
    }

    /// Lower one annotated box, or flatten it if it emits nothing.
    fn walk_box(
        &mut self,
        py: Python,
        dag: &DAGCircuit,
        inst: &PackedInstruction,
        out: &mut DAGCircuitBuilder,
        scope: &Scope,
    ) -> PyResult<()> {
        let annotations = box_annotations(py, inst)?;
        let resolved = resolve_annotations(&annotations).into_py_result()?;

        let locals = dag.qargs_interner().get(inst.qubits);
        let out_qargs = scope.out_qubits(locals)?;
        let global = scope.global_qubits(locals)?;
        let body = match dag.try_view_control_flow(inst) {
            Some(ControlFlowView::Box { body, .. }) => body,
            _ => return Err(PyValueError::new_err("box instruction is missing its body")),
        };
        let body_clbits: Vec<usize> = dag
            .cargs_interner()
            .get(inst.clbits)
            .iter()
            .map(|c| c.index())
            .collect();
        let out_cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;

        // A box that emits nothing — unannotated, or `Tag` only — is a transparent wrapper. Flatten
        // it: walk its body straight into the current output, remapped through this box's qargs.
        if !resolved.is_emitting() {
            let flat_q: Vec<usize> = out_qargs.iter().map(|q| q.index()).collect();
            let flat_c: Vec<usize> = out_cargs.iter().map(|c| c.index()).collect();
            let inner = Scope {
                qubits: &flat_q,
                global: &global,
                clbits: &flat_c,
            };
            return self.walk(py, body, out, &inner);
        }

        // One id for this box, stamped on every emission it produces and on both of its collectors.
        // That pairing is what a later pass checks instead of trusting adjacency.
        let box_id = self.alloc_box_id();
        let dressing = resolved.dressing.unwrap_or(Dressing::Left);
        let emissions = self.build_emissions(&resolved, &global, dressing, box_id);
        let synthesizer = resolved.synthesizer.unwrap_or(DEFAULT_SYNTHESIZER);
        let partition = Partition::from_elements(global.iter().copied());
        let collect_parts: Vec<CollectPart> = (0..partition.len())
            .map(|_| CollectPart { synthesizer })
            .collect();

        // The body splits into gates the dressing can absorb and everything else. Nested annotated
        // boxes are always "everything else", and are lowered recursively into the hard box so that
        // an outer emission crossing them sees their real gates.
        let width = locals.len();
        let inner_global = global.clone();
        let inner_identity_q: Vec<usize> = (0..width).collect();
        let inner_identity_c: Vec<usize> = (0..body_clbits.len()).collect();
        let inner = Scope {
            qubits: &inner_identity_q,
            global: &inner_global,
            clbits: &inner_identity_c,
        };

        // Classified before anything is written, because whether there *is* hard content decides where
        // the propagating emissions go, and they are written into the hard body ahead of its gates.
        let (easy_nodes, hard_nodes) = classify_body(body, resolved.dressing);
        let has_hard_content = !hard_nodes.is_empty();

        let mut easy_builder = new_body(width, body_clbits.len(), easy_nodes.len())?.into_builder();
        for node in easy_nodes {
            copy_instruction(
                body,
                body.dag()[node].unwrap_operation(),
                &mut easy_builder,
                &inner,
            )?;
        }
        let easy = easy_builder.build();

        // Collectors start empty — the absorb_dressing pass populates them by walking the spine.
        let empty_body = new_body(width, body_clbits.len(), 0)?;
        let left = CollectSpec {
            owned: vec![box_id],
            partition: partition.clone(),
            parts: collect_parts.clone(),
        };
        let right = CollectSpec {
            owned: vec![box_id],
            partition: partition.clone(),
            parts: collect_parts,
        };

        // Partition emissions into groups for each edge. Within each edge, the spine order is:
        //   outer emissions (depth >= DEPTH_BASIS) | easy gates | inner emissions (local, depth < DEPTH_BASIS)
        //   | propagating emissions (facing away)
        // The dressing side carries the easy gates; the opposite side has no gates.
        let is_outer = |p: &&Placed| p.depth >= DEPTH_BASIS;
        let is_local = |p: &&Placed, side: Direction| {
            let faces_collector = match side {
                Direction::Left => p.spec.direction == Some(Direction::Left),
                Direction::Right => p.spec.direction == Some(Direction::Right),
            };
            faces_collector || p.depth >= DEPTH_BASIS
        };

        let sorted = |select: &dyn Fn(&&Placed) -> bool, side: Direction| -> Vec<&Placed> {
            let mut group: Vec<&Placed> = emissions.iter().filter(|p| select(p)).collect();
            match side {
                Direction::Left => group.sort_by_key(|p| std::cmp::Reverse(p.depth)),
                Direction::Right => group.sort_by_key(|p| p.depth),
            }
            group
        };

        // A scope that maps body-local qubits to the output circuit qubits.
        let body_to_out: Vec<usize> = out_qargs.iter().map(|q| q.index()).collect();
        let body_scope = Scope {
            qubits: &body_to_out,
            global: &global,
            clbits: scope.clbits,
        };

        let left_propagating = sorted(
            &|p| p.edge == Direction::Left && !is_local(p, Direction::Left),
            Direction::Left,
        );
        let right_propagating = sorted(
            &|p| p.edge == Direction::Right && !is_local(p, Direction::Right),
            Direction::Right,
        );

        // Build the hard body, with the propagating emissions *inside* it at the edge each starts
        // from — front for the ones travelling right, back for the ones travelling left. That is where
        // they belong: the hard content is exactly what they are conjugated by on the way to the far
        // collector, so writing them outside it would put the box boundary between an emission and the
        // gates it has to cross. It also means nothing has to move them later.
        let mut hard_builder = new_body(width, body_clbits.len(), hard_nodes.len())?.into_builder();
        if has_hard_content {
            write_emissions(py, &mut hard_builder, &left_propagating, &inner)?;
            for node in hard_nodes {
                let inst = body.dag()[node].unwrap_operation();
                match inst.op.view() {
                    OperationRef::ControlFlow(cf) => {
                        if !matches!(cf.control_flow, ControlFlow::Box { .. }) {
                            return Err(PyValueError::new_err(format!(
                                "Unsupported control flow in a samplex circuit: '{}'.",
                                cf.name()
                            )));
                        }
                        // A nested annotated box is lowered in place, so its collect boxes and
                        // emissions land inside this hard box. An outer emission's walk crosses them
                        // — including the gates the inner dressing absorbed, which are still real
                        // gates in the inner collector's body at this stage.
                        self.walk_box(py, body, inst, &mut hard_builder, &inner)?;
                    }
                    _ => copy_instruction(body, inst, &mut hard_builder, &inner)?,
                }
            }
            write_emissions(py, &mut hard_builder, &right_propagating, &inner)?;
        }
        let hard = hard_builder.build();

        // Write left edge: collector, outer emissions, easy gates (if left-dressed), inner
        // emissions.
        write_collect(py, out, left, empty_body.clone(), &out_qargs, &out_cargs)?;
        let left_outer = sorted(
            &|p| p.edge == Direction::Left && is_outer(p),
            Direction::Left,
        );
        write_emissions(py, out, &left_outer, scope)?;
        if matches!(dressing, Dressing::Left) {
            write_easy_gates(out, &easy, &body_scope)?;
        }
        let left_inner = sorted(
            &|p| p.edge == Direction::Left && !is_outer(p) && is_local(p, Direction::Left),
            Direction::Left,
        );
        write_emissions(py, out, &left_inner, scope)?;
        // With no hard content there is no hard box to sit inside, and nothing for the emission to be
        // conjugated by either — so it stays on the spine, where its collector absorbs it as local.
        if !has_hard_content {
            write_emissions(py, out, &left_propagating, scope)?;
        }

        // Hard box.
        write_hard_box(out, hard, &out_qargs, &out_cargs)?;

        // Write right edge: inner emissions, easy gates (if right-dressed), outer emissions,
        // collector.
        if !has_hard_content {
            write_emissions(py, out, &right_propagating, scope)?;
        }
        let right_inner = sorted(
            &|p| p.edge == Direction::Right && !is_outer(p) && is_local(p, Direction::Right),
            Direction::Right,
        );
        write_emissions(py, out, &right_inner, scope)?;
        if matches!(dressing, Dressing::Right) {
            write_easy_gates(out, &easy, &body_scope)?;
        }
        let right_outer = sorted(
            &|p| p.edge == Direction::Right && is_outer(p),
            Direction::Right,
        );
        write_emissions(py, out, &right_outer, scope)?;
        write_collect(
            py,
            out,
            right,
            new_body(width, body_clbits.len(), 0)?,
            &out_qargs,
            &out_cargs,
        )?;
        Ok(())
    }

    /// Turn a resolved box into its emissions, each tagged with where on the spine it belongs.
    ///
    /// A `Twirl` yields **two** — the inverse pair — sharing one table key with opposite directions;
    /// the inversion is implied by the direction rather than recorded. Its pair goes on the *dressing*
    /// edge, because that edge is the twirl point and the easy/hard split is defined relative to it.
    ///
    /// A basis change or noise injection yields one, on the edge its own `placement` / `site` names —
    /// **not** the dressing edge. When the two differ the hard box would otherwise sit between the
    /// emission and the collector consuming it, so the propagation walk would conjugate it by content it
    /// is meant to sit outside of. None of these ever propagate through hard content.
    fn build_emissions(
        &mut self,
        resolved: &ResolvedBox,
        qubits: &[usize],
        dressing: Dressing,
        box_id: u32,
    ) -> Vec<Placed> {
        let partition = Partition::from_elements(qubits.iter().copied());
        let num_parts = partition.len();
        let dressing_edge = match dressing {
            Dressing::Left => Direction::Left,
            Dressing::Right => Direction::Right,
        };
        let mut emissions = Vec::new();

        if let Some(twirl) = &resolved.twirl {
            let dist = self
                .table
                .intern(DistEntry::Distribution(twirl.distribution));
            let virtual_type = twirl.distribution.virtual_type();
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            for direction in [Direction::Left, Direction::Right] {
                let adjoint = direction != dressing_edge;
                let parts = (0..num_parts)
                    .map(|i| EmitPart {
                        dist,
                        virtual_type,
                        draw: draw_base + i as u32,
                        adjoint,
                    })
                    .collect();
                emissions.push(Placed {
                    spec: EmitSpec {
                        box_id,
                        source: EmitSource::Twirl,
                        direction: Some(direction),
                        partition: partition.clone(),
                        parts,
                    },
                    edge: dressing_edge,
                    depth: DEPTH_TWIRL,
                });
            }
        }
        if let Some(basis) = &resolved.change_basis {
            let dist = self.table.intern(DistEntry::Basis {
                mode: basis.mode,
                ref_id: basis.ref_id.clone(),
            });
            let direction: Direction = basis.placement.into();
            let virtual_type = basis.mode.virtual_type();
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            let parts = (0..num_parts)
                .map(|i| EmitPart {
                    dist,
                    virtual_type,
                    draw: draw_base + i as u32,
                    adjoint: false,
                })
                .collect();
            emissions.push(Placed {
                spec: EmitSpec {
                    box_id,
                    source: EmitSource::ChangeBasis,
                    direction: Some(direction),
                    partition: partition.clone(),
                    parts,
                },
                edge: direction,
                depth: match basis.origin {
                    BasisOrigin::ChangeBasis => DEPTH_BASIS,
                    BasisOrigin::InjectLocalClifford => DEPTH_INJECTION,
                },
            });
        }
        if let Some(noise) = &resolved.inject_noise {
            let dist = self.table.intern(DistEntry::Noise {
                reference: noise.reference.clone(),
                modifier: noise.modifier.clone(),
            });
            let direction: Direction = noise.site.into();
            let virtual_type = VirtualType::Pauli;
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            let parts = (0..num_parts)
                .map(|i| EmitPart {
                    dist,
                    virtual_type,
                    draw: draw_base + i as u32,
                    adjoint: false,
                })
                .collect();
            emissions.push(Placed {
                spec: EmitSpec {
                    box_id,
                    source: EmitSource::InjectNoise,
                    direction: Some(direction),
                    partition: partition.clone(),
                    parts,
                },
                edge: direction,
                depth: DEPTH_INJECTION,
            });
        }
        emissions
    }
}

/// Sweep a box body from the dressing edge, splitting absorbable gates from the rest.
///
/// Classification only — it decides *which* nodes are easy and which are hard, and writes nothing. The
/// caller needs that answer before it starts writing, because whether the box has hard content at all
/// decides where its propagating emissions go.
///
/// **Per qubit, not one latch for the whole body.** A single-qubit gate on a wire that no multi-qubit
/// gate has touched is still at the dressing edge *on its own wire*, so it commutes out and folds into
/// the dressing even when it sits after an entangler elsewhere in the body.
///
/// Poisoning over a topological order is exactly DAG ancestry: a gate is absorbable iff every one of
/// its ancestors was absorbed, and since absorbed gates all move to the dressing edge keeping their
/// relative order, such a gate can move there too. Poison spreads transitively, so
/// `cx(0,1); cx(1,2); s(2)` correctly leaves the `s` as content.
fn classify_body(
    body: &DAGCircuit,
    dressing: Option<Dressing>,
) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    let nodes: Vec<_> = body.topological_op_nodes(false).collect();
    // Sweeping from the right means visiting in reverse, then restoring circuit order.
    let right = matches!(dressing, Some(Dressing::Right));
    let order: Vec<_> = if right {
        nodes.iter().rev().copied().collect()
    } else {
        nodes.clone()
    };

    // No dressing at all means nothing is absorbable, which poisoning every wire expresses.
    let dressed = dressing.is_some();
    let mut poisoned: HashSet<usize> = HashSet::new();
    let mut easy_nodes = Vec::new();
    let mut hard_nodes = Vec::new();
    for node in order {
        let inst = body.dag()[node].unwrap_operation();
        let qargs = body.qargs_interner().get(inst.qubits);
        let absorbable =
            dressed && is_absorbable(body, inst) && !poisoned.contains(&qargs[0].index());
        if absorbable {
            easy_nodes.push(node);
        } else {
            poisoned.extend(qargs.iter().map(|q| q.index()));
            hard_nodes.push(node);
        }
    }
    if right {
        easy_nodes.reverse();
        hard_nodes.reverse();
    }
    (easy_nodes, hard_nodes)
}

/// Whether the dressing can absorb this instruction: a single-qubit standard gate.
fn is_absorbable(dag: &DAGCircuit, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && dag.qargs_interner().get(inst.qubits).len() == 1
}

/// The `BoxAnnotation`s on a box instruction. Annotations that are not ours are ignored.
fn box_annotations(py: Python, inst: &PackedInstruction) -> PyResult<Vec<BoxAnnotation>> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return Ok(Vec::new());
    };
    let ControlFlow::Box { annotations, .. } = &cf.control_flow else {
        return Ok(Vec::new());
    };
    Ok(annotations
        .iter()
        .filter_map(|a| extract_annotation(a.bind(py)).ok())
        .collect())
}

fn new_body(num_qubits: usize, num_clbits: usize, capacity: usize) -> PyResult<DAGCircuit> {
    new_dag_body(num_qubits, num_clbits, capacity)
}

/// Copy an instruction verbatim into `out`, remapping its bits through `scope`.
fn copy_instruction(
    dag: &DAGCircuit,
    inst: &PackedInstruction,
    out: &mut DAGCircuitBuilder,
    scope: &Scope,
) -> PyResult<()> {
    let qargs = scope.out_qubits(dag.qargs_interner().get(inst.qubits))?;
    let cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;
    let params: Option<Parameters<_>> = (!inst.params_view().is_empty()).then(|| {
        Parameters::Params(
            inst.params_view()
                .iter()
                .cloned()
                .collect::<SmallVec<[Param; 3]>>(),
        )
    });
    append(out, inst.op.clone(), params, &qargs, &cargs)
}

/// Write the `Emit` instructions belonging to one edge of a box, in the order given.
fn write_emissions(
    py: Python,
    out: &mut DAGCircuitBuilder,
    emissions: &[&Placed],
    scope: &Scope,
) -> PyResult<()> {
    for spec in emissions.iter().map(|placed| &placed.spec) {
        // The spec's partition is global; the qargs must be in the output's frame.
        let qargs: Vec<Qubit> = spec
            .qubits()
            .iter()
            .map(|g| {
                scope
                    .global
                    .iter()
                    .position(|x| x == g)
                    .and_then(|local| scope.qubits.get(local).copied())
                    .map(|i| Qubit(i as u32))
                    .ok_or_else(|| PyValueError::new_err(format!("qubit {g} not in scope")))
            })
            .collect::<PyResult<_>>()?;
        let emit = Py::new(py, Emit::new(spec.clone()))?;
        let op = PackedOperation::from(PyInstruction {
            kind: PyOpKind::Operation,
            qubits: qargs.len() as u32,
            clbits: 0,
            params: 0,
            op_name: spec.source.name().to_string(),
            ob: emit.into_any(),
        });
        append(out, op, None, &qargs, &[])?;
    }
    Ok(())
}

/// Write the easy (absorbed) gates directly onto the spine.
///
/// Each gate is copied from `easy` (a body-local DAG) into `out`, remapping its qubits through
/// `scope` so they land on the correct output wires. Every absorbable gate is single-qubit, so
/// topological order preserves each wire's run, which is the only order that carries meaning.
fn write_easy_gates(out: &mut DAGCircuitBuilder, easy: &DAGCircuit, scope: &Scope) -> PyResult<()> {
    for node in easy.topological_op_nodes(false) {
        let inst = easy.dag()[node].unwrap_operation();
        let qargs: Vec<Qubit> = easy
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|q| Qubit(scope.qubits[q.index()] as u32))
            .collect();
        let params: Option<Parameters<_>> = (!inst.params_view().is_empty()).then(|| {
            Parameters::Params(
                inst.params_view()
                    .iter()
                    .cloned()
                    .collect::<SmallVec<[Param; 3]>>(),
            )
        });
        append(out, inst.op.clone(), params, &qargs, &[])?;
    }
    Ok(())
}

/// Write a collect box. Skipped entirely when it would collect nothing and absorb nothing.
fn write_collect(
    py: Python,
    out: &mut DAGCircuitBuilder,
    spec: CollectSpec,
    body: DAGCircuit,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    let annotation = Py::new(py, (Collect::new_from_spec(spec), PyAnnotation))?;
    write_box(out, body, vec![annotation.into_any()], qargs, cargs)
}

/// Write the box holding the gates virtual state is conjugated by. Skipped when empty — with
/// propagation derived from placement, a gateless box carries no information.
fn write_hard_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    if body.num_ops() == 0 {
        return Ok(());
    }
    write_box(out, body, Vec::new(), qargs, cargs)
}

fn write_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Py<PyAny>>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: None,
            annotations,
        },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    }));
    let block = out.add_block(body);
    append(out, op, Some(Parameters::Blocks(vec![block])), qargs, cargs)
}
