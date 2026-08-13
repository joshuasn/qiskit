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
//! Turns each annotated box into **three boxes** — collect, content, collect — with the box's
//! `Emit` instructions on the spine between them. The samplex annotations are factored out into
//! those emissions and collectors; everything else about the box (annotations we do not act on, its
//! duration, its body) stays on the content box. Also produces the [`DistributionTable`] those
//! emissions reference.
//!
//! The body is **not** split here. Which of its gates fold into a dressing is decided by walking, in
//! `absorb_dressing`, so what remains in a content box afterwards is exactly what could not be
//! absorbed.
//!
//! **This pass is purely local.** Every annotated box yields exactly two collect boxes, one per
//! side, with no cross-box state; widening them and fusing adjacent ones is `merge_collectors`'
//! job. So this is a single forward sweep appending to a [`DAGCircuitBuilder`], never revisiting
//! what it has written.
//!
//! Parameters are deliberately absent; they are minted during lowering.

use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qiskit_circuit::annotation::PyAnnotation;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeIndex};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    BoxDuration, ControlFlow, ControlFlowInstruction, ControlFlowView, Operation, OperationRef,
    Param,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};

use crate::annotated_circuit::{
    BasisOrigin, BoxAnnotation, Dressing, ResolvedBox, SynthesizerType, extract_annotation,
    resolve_annotations,
};
use crate::distributions::{DistEntry, DistKey, DistributionTable};
use crate::emission_circuit::{Collect, CollectPart, CollectSpec, EmitPart, EmitSpec};
use crate::partition::Partition;
use crate::virtual_flow_graph::Direction;

use super::utils::{IntoPyResult, append, new_dag_body};

/// The synthesizer assumed when a box's annotations do not name one.
///
/// Unreachable as things stand: `InjectLocalClifford` is the only annotation that names no synthesizer
/// and it cannot stand without a `Twirl`, which does. Kept as the stated default rather than an
/// `expect`, so adding an annotation that names none is a silent sensible choice rather than a panic.
const DEFAULT_SYNTHESIZER: SynthesizerType = SynthesizerType::RzSx;

/// How deeply an emission nests inside its box, `0` being immediately against the content box.
///
/// Fixes both the spine order and each collector's composition order. `InjectLocalClifford` counts
/// as an injection despite resolving to a `ResolvedBasis`, which is what [`BasisOrigin`] records.
const DEPTH_TWIRL: u8 = 0;
const DEPTH_INJECTION: u8 = 1;
const DEPTH_BASIS: u8 = 2;

/// The emissions that go inside a box's content, in the order they are written there.
///
/// Everything nearer the content than the dressing is in here, which is every emission except the
/// ones naming the box's own edge. The two propagating groups sit against the content, with the
/// facing ones outside them — an emission's `depth` orders each group within itself.
struct ContentEmissions<'a> {
    left_facing: Vec<&'a Placed>,
    left_propagating: Vec<&'a Placed>,
    right_propagating: Vec<&'a Placed>,
    right_facing: Vec<&'a Placed>,
}

impl ContentEmissions<'_> {
    /// For a box that emits nothing, so its body has no twirl point in it.
    fn none() -> Self {
        ContentEmissions {
            left_facing: Vec::new(),
            left_propagating: Vec::new(),
            right_propagating: Vec::new(),
            right_facing: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.left_facing.len()
            + self.left_propagating.len()
            + self.right_propagating.len()
            + self.right_facing.len()
    }
}

/// An emission together with where it goes relative to its box's content.
struct Placed {
    spec: EmitSpec,
    /// Which side of the content box it is written on.
    edge: Direction,
    /// Distance from the content box; see the `DEPTH_*` constants.
    depth: u8,
}

/// How a walked scope's qubits map outward. Emissions record `global`.
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
}

/// Build the emission circuit for an annotated circuit.
#[pyfunction]
#[pyo3(name = "build_lowered")]
pub fn py_build(py: Python, dag: &DAGCircuit) -> PyResult<(DAGCircuit, DistributionTable)> {
    build(py, dag)
}

/// Build the emission circuit for an annotated circuit.
pub fn build(py: Python, dag: &DAGCircuit) -> PyResult<(DAGCircuit, DistributionTable)> {
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
                             supported.",
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
        let (annotations, foreign) = box_annotations(py, inst)?;
        let resolved = resolve_annotations(&annotations).into_py_result()?;
        let duration = box_duration(inst);

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

        let width = locals.len();

        // A box that emits nothing is not a samplex box but user content. With nothing on it worth
        // keeping it is a transparent wrapper — unannotated or `Tag` only — so its body walks straight
        // into the current output. With annotations we do not own, or a duration, the box stays:
        // dissolving it would discard them.
        if !resolved.is_emitting() {
            if foreign.is_empty() && duration.is_none() {
                let flat_q: Vec<usize> = out_qargs.iter().map(|q| q.index()).collect();
                let flat_c: Vec<usize> = out_cargs.iter().map(|c| c.index()).collect();
                let inner = Scope {
                    qubits: &flat_q,
                    global: &global,
                    clbits: &flat_c,
                };
                return self.walk(py, body, out, &inner);
            }
            // Nothing is emitted, so there is no twirl point to place and nothing to classify.
            let content = self.content_body(
                py,
                body,
                width,
                body_clbits.len(),
                &global,
                None,
                &ContentEmissions::none(),
            )?;
            return write_content_box(out, content, foreign, duration, &out_qargs, &out_cargs);
        }

        let dressing = resolved.dressing.unwrap_or(Dressing::Left);
        let emissions = self.build_emissions(&resolved, global.len(), dressing);
        let synthesizer = resolved.synthesizer.unwrap_or(DEFAULT_SYNTHESIZER);
        // One subsystem per qubit: nothing here samples a box's qubits jointly yet.
        let partition = Partition::singletons(global.len());
        let collect_parts: Vec<CollectPart> = (0..partition.len())
            .map(|_| CollectPart { synthesizer })
            .collect();

        // Collectors start empty — the absorb_dressing pass populates them by walking the spine.
        let empty_body = new_body(width, body_clbits.len(), 0)?;
        let left = CollectSpec {
            partition: partition.clone(),
            parts: collect_parts.clone(),
        };
        let right = CollectSpec {
            partition: partition.clone(),
            parts: collect_parts,
        };

        // Partition emissions into groups for each edge. Only the *outer* ones stay on the spine: a
        // basis change names the box's own edge, so that is where it belongs. Everything nearer the
        // content than the dressing goes inside the content box, in this order per edge:
        //   easy run | inner emissions (facing the collector) | propagating (facing away) | content
        // An emission's `depth` is its distance from the content, which is what fixes both that order
        // and each collector's composition order.
        //
        // **A twirl's two halves must sit together, at the twirl point.** They are one pair around one
        // point, so splitting them across the box boundary would make the near half compose on the far
        // side of gates the far half is conjugated by — the pair would no longer be inverses of one
        // draw about the same point.
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

        let inside = ContentEmissions {
            left_facing: sorted(
                &|p| p.edge == Direction::Left && !is_outer(p) && is_local(p, Direction::Left),
                Direction::Left,
            ),
            left_propagating: sorted(
                &|p| p.edge == Direction::Left && !is_local(p, Direction::Left),
                Direction::Left,
            ),
            right_propagating: sorted(
                &|p| p.edge == Direction::Right && !is_local(p, Direction::Right),
                Direction::Right,
            ),
            right_facing: sorted(
                &|p| p.edge == Direction::Right && !is_outer(p) && is_local(p, Direction::Right),
                Direction::Right,
            ),
        };

        // The body goes into the content box whole — nothing is hoisted onto the spine — with the
        // twirl point and its emissions written *inside* it, and nested annotated boxes lowered in
        // place so an outer emission crossing this box sees their real gates.
        let content = self.content_body(
            py,
            body,
            width,
            body_clbits.len(),
            &global,
            resolved.dressing,
            &inside,
        )?;

        // Write the left edge: the collector, then the emissions that name the box's own edge. The
        // rest are inside the content box, at the twirl point.
        write_collect(py, out, left, empty_body.clone(), &out_qargs, &out_cargs)?;
        let left_outer = sorted(
            &|p| p.edge == Direction::Left && is_outer(p),
            Direction::Left,
        );
        write_emissions(out, &left_outer, &out_qargs)?;

        write_content_box(out, content, foreign, duration, &out_qargs, &out_cargs)?;

        // Write the right edge, mirrored.
        let right_outer = sorted(
            &|p| p.edge == Direction::Right && is_outer(p),
            Direction::Right,
        );
        write_emissions(out, &right_outer, &out_qargs)?;
        write_collect(py, out, right, empty_body, &out_qargs, &out_cargs)?;
        Ok(())
    }

    /// Walk a box's body into a fresh body DAG, with the propagating emissions at the twirl point.
    ///
    /// `global` maps a body-local qubit to its circuit qubit, which is what the emissions inside
    /// record.
    ///
    /// **Where the propagating emissions go is the twirl point, and it is not the box boundary.** The
    /// absorbable run is swept to the dressing edge and they are written just after it, so those gates
    /// are on the near side of the twirl point and multiply into the dressing. Writing them at the
    /// boundary instead would leave every gate in the body to be *crossed*, and a non-Clifford cannot
    /// be crossed by a Pauli at all — an `rz` in a twirled box would stop being expressible.
    ///
    /// Sweeping is sound because absorbability is DAG ancestry: a gate is absorbable only if all of
    /// its ancestors are, so it can move to the dressing edge keeping its relative order. Which of
    /// those gates actually get folded in is still `absorb_dressing`'s decision, taken by walking into
    /// this box; this only decides what is on which side of the twirl point.
    #[allow(clippy::too_many_arguments)]
    fn content_body(
        &mut self,
        py: Python,
        body: &DAGCircuit,
        width: usize,
        num_clbits: usize,
        global: &[usize],
        dressing: Option<Dressing>,
        inside: &ContentEmissions,
    ) -> PyResult<DAGCircuit> {
        let identity_q: Vec<usize> = (0..width).collect();
        let identity_c: Vec<usize> = (0..num_clbits).collect();
        let inner = Scope {
            qubits: &identity_q,
            global,
            clbits: &identity_c,
        };
        let (easy_nodes, hard_nodes) = classify_body(body, dressing);
        let mut builder =
            new_body(width, num_clbits, body.num_ops() + inside.len())?.into_builder();
        // A body is exactly as wide as its box, so inside it the box's qubits *are* `0..width`. An
        // emission's own partition indexes its qargs, so these are what it is written on.
        let body_qargs: Vec<Qubit> = (0..width as u32).map(Qubit).collect();

        // A right dressing sweeps the absorbable run to the other end, so it is a suffix there.
        let easy_first = !matches!(dressing, Some(Dressing::Right));
        if easy_first {
            self.copy_nodes(py, body, &easy_nodes, &mut builder, &inner)?;
        }
        write_emissions(&mut builder, &inside.left_facing, &body_qargs)?;
        write_emissions(&mut builder, &inside.left_propagating, &body_qargs)?;
        self.copy_nodes(py, body, &hard_nodes, &mut builder, &inner)?;
        write_emissions(&mut builder, &inside.right_propagating, &body_qargs)?;
        write_emissions(&mut builder, &inside.right_facing, &body_qargs)?;
        if !easy_first {
            self.copy_nodes(py, body, &easy_nodes, &mut builder, &inner)?;
        }
        Ok(builder.build())
    }

    /// Copy a run of a body's nodes, lowering a nested annotated box in place.
    fn copy_nodes(
        &mut self,
        py: Python,
        body: &DAGCircuit,
        nodes: &[NodeIndex],
        out: &mut DAGCircuitBuilder,
        inner: &Scope,
    ) -> PyResult<()> {
        for node in nodes {
            let inst = body.dag()[*node].unwrap_operation();
            match inst.op.view() {
                OperationRef::ControlFlow(cf) => {
                    if !matches!(cf.control_flow, ControlFlow::Box { .. }) {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported control flow in a samplex circuit: '{}'.",
                            cf.name()
                        )));
                    }
                    // A nested annotated box is lowered in place, so its collect boxes and emissions
                    // land inside this content box, where an outer emission's walk crosses them.
                    self.walk_box(py, body, inst, out, inner)?;
                }
                _ => copy_instruction(body, inst, out, inner)?,
            }
        }
        Ok(())
    }

    /// Turn a resolved box into its emissions, each tagged with where on the spine it belongs.
    ///
    /// A `Twirl` yields two — the inverse pair, sharing one table key and its draw slots, with
    /// opposite directions — on the *dressing* edge. A basis change or noise injection yields one,
    /// on the edge its own `placement` / `site` names and **not** the dressing edge.
    fn build_emissions(
        &mut self,
        resolved: &ResolvedBox,
        width: usize,
        dressing: Dressing,
    ) -> Vec<Placed> {
        // Every emission of a box covers the box's full width, one subsystem per qubit.
        let partition = Partition::singletons(width);
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
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            for direction in [Direction::Left, Direction::Right] {
                let adjoint = direction != dressing_edge;
                let parts = (0..num_parts)
                    .map(|i| EmitPart {
                        dist,
                        draw: draw_base + i as u32,
                        adjoint,
                    })
                    .collect();
                emissions.push(Placed {
                    spec: EmitSpec {
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
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            let parts = (0..num_parts)
                .map(|i| EmitPart {
                    dist,
                    draw: draw_base + i as u32,
                    adjoint: false,
                })
                .collect();
            emissions.push(Placed {
                spec: EmitSpec {
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
            let draw_base = self.alloc_draws(dist, num_parts as u32);
            let parts = (0..num_parts)
                .map(|i| EmitPart {
                    dist,
                    draw: draw_base + i as u32,
                    adjoint: false,
                })
                .collect();
            emissions.push(Placed {
                spec: EmitSpec {
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

/// Sweep a box body from the dressing edge, splitting the absorbable run from the rest.
///
/// Classification only, writing nothing. It no longer decides which box a gate goes in — everything
/// stays in the content box — only which side of the twirl point it lands on. Absorbability is **per
/// qubit**, not one latch for the whole body.
fn classify_body(
    body: &DAGCircuit,
    dressing: Option<Dressing>,
) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    // Poisoning over a topological order is DAG ancestry: a gate is absorbable iff all of its
    // ancestors were, and since absorbable gates all move to the dressing edge keeping their relative
    // order, such a gate can move there too. So a single-qubit gate on an untouched wire folds into
    // the dressing even if an entangler sits before it elsewhere in the body, while poison
    // spreading transitively leaves the `s` in `cx(0,1); cx(1,2); s(2)` as content.
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

/// Whether a dressing could absorb this instruction: a single-qubit standard gate.
fn is_absorbable(dag: &DAGCircuit, inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::StandardGate(_))
        && dag.qargs_interner().get(inst.qubits).len() == 1
}

/// A box's annotations, split into the ones we act on and the ones we only carry.
///
/// The second half is not ours to interpret, so it rides along on the content box rather than being
/// dropped: whatever it means to the consumer, it still means it about this box's content.
fn box_annotations(
    py: Python,
    inst: &PackedInstruction,
) -> PyResult<(Vec<BoxAnnotation>, Vec<Py<PyAny>>)> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let ControlFlow::Box { annotations, .. } = &cf.control_flow else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut ours = Vec::new();
    let mut foreign = Vec::new();
    for annotation in annotations {
        match extract_annotation(annotation.bind(py)) {
            Ok(parsed) => ours.push(parsed),
            Err(_) => foreign.push(annotation.clone_ref(py)),
        }
    }
    Ok((ours, foreign))
}

/// A box's duration, which the content box carries for the same reason as a foreign annotation.
fn box_duration(inst: &PackedInstruction) -> Option<BoxDuration> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return None;
    };
    let ControlFlow::Box { duration, .. } = &cf.control_flow else {
        return None;
    };
    duration.clone()
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

/// Write the emissions belonging to one edge of a box, in the order given, on `qargs`.
///
/// Every emission of a box covers that box's full width, so they all land on the same wires — the
/// box's own qargs, in whichever frame is being written into. That is also the frame the specs'
/// partitions index into, which is what keeps them meaningful wherever the emission ends up.
fn write_emissions(
    out: &mut DAGCircuitBuilder,
    emissions: &[&Placed],
    qargs: &[Qubit],
) -> PyResult<()> {
    for spec in emissions.iter().map(|placed| &placed.spec) {
        if spec.partition.num_qubits() != qargs.len() {
            return Err(PyValueError::new_err(format!(
                "an emission on {} qubits cannot be written on {} of them",
                spec.partition.num_qubits(),
                qargs.len(),
            )));
        }
        let op = PackedOperation::from_custom_operation(Box::new(spec.clone()));
        append(out, op, None, qargs, &[])?;
    }
    Ok(())
}

/// Write a collect box, with an empty body for `absorb_dressing` to fill in.
fn write_collect(
    py: Python,
    out: &mut DAGCircuitBuilder,
    spec: CollectSpec,
    body: DAGCircuit,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    let annotation = Py::new(py, (Collect::new_from_spec(spec), PyAnnotation))?;
    write_box(out, body, vec![annotation.into_any()], None, qargs, cargs)
}

/// Write the box holding this box's content, and whatever else rode along on it.
///
/// Written even when the body is empty, and *especially* then: after absorption a content box holds
/// exactly what could not be absorbed, so an empty one is the statement that nothing here is hard.
fn write_content_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Py<PyAny>>,
    duration: Option<BoxDuration>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    write_box(out, body, annotations, duration, qargs, cargs)
}

fn write_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Py<PyAny>>,
    duration: Option<BoxDuration>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> PyResult<()> {
    let op = PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration,
            annotations,
        },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    }));
    let block = out.add_block(body);
    append(out, op, Some(Parameters::Blocks(vec![block])), qargs, cargs)
}
