// This code is a Qiskit project.
//
// (C) Copyright IBM 2025, 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Getting around an emission circuit: addressing an instruction inside nested boxes, reading what a
//! box declares, walking a wire, and rewriting what a walk found.
//!
//! An emission circuit is a nest of boxes, so a bare `NodeIndex` does not identify an instruction —
//! it only does once you already know which `DAGCircuit` it belongs to, which a walk crossing box
//! boundaries no longer does. [`Site`] is the address that does identify one, and every other item
//! here either produces a site, reads through one, or rewrites at one.
//!
//! Three IR2 invariants live here rather than in each pass's prose, so that a pass gets them by
//! calling rather than by remembering:
//!
//! - **A walk descends into content, never into a collector.** A collect box is a barrier: its body
//!   holds only what was already absorbed into it, so there is nothing in there for anyone else to
//!   reach. [`WireCursor::advance`] and [`collectors`] both apply this, and neither takes it as an
//!   argument.
//! - **A walk descends but never ascends.** A cursor climbs back out of a box it descended into, and
//!   stops at the end of the scope it started in. That asymmetry is load-bearing: a collector inside
//!   a box must not reach out and take an enclosing box's emission, which would undo that box's
//!   randomization with none of its content in between. It is also what makes [`Site::deeper_than`]
//!   enough to tell whether a site a walk reached is inside a given scope — depth alone decides,
//!   because a reached site can never be shallower.
//! - **A collect box carries exactly one annotation and no duration.** [`collect_op`] mints them that
//!   way and [`collect_annotation`] reads them back, and those two are the whole of it.
//!
//! What a walk finds is offered as data as well as by address. A [`Sighting`] is what one position
//! looks like to a rule, and [`WireSights`] and [`EmissionTally`] are the two readings of a body that
//! the IR2 rules decide on. That seam is the point: reading a circuit needs the circuit, deciding does
//! not, so a rule stated over sightings can be put to a case written as a literal instead of as a
//! nested circuit.
//!
//! Nothing here holds a `Python` token. A native annotation is a Rust value, so asking what a box
//! declares is a `TypeId` comparison rather than an attribute lookup.

use std::sync::Arc;

use rustworkx_core::petgraph::Direction as PetDirection;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;

use qiskit_circuit::annotation::Annotation;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    BoxDuration, ControlFlow, ControlFlowInstruction, Operation, OperationRef,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Block, Clbit, Qubit};

use crate::emission_circuit::{Collect, Emit};
use crate::error::{Result, SamplexError};
use crate::sampling_graph::Direction;

// --- What an instruction declares ---------------------------------------------------------------

/// Whether an instruction is a `box`.
pub fn is_box(inst: &PackedInstruction) -> bool {
    matches!(inst.op.view(), OperationRef::ControlFlow(cf) if matches!(cf.control_flow, ControlFlow::Box { .. }))
}

/// Whether this instruction is a collect box.
///
/// The borrowing half of [`collect_annotation`], for the many walk sites that only need to know
/// whether to descend, not what the collector holds.
pub fn is_collector(inst: &PackedInstruction) -> bool {
    box_collect(inst).is_some()
}

/// The [`Collect`] on this instruction, if it is a collector.
pub fn collect_annotation(inst: &PackedInstruction) -> Option<Collect> {
    box_collect(inst).cloned()
}

/// Borrow the collect annotation off a box's annotations.
///
/// The crate's only read of a box's annotations for its own vocabulary, and the reason none of the
/// IR2 walks need a `Python` token: a native annotation is a Rust value, so asking what a box
/// declares is a `TypeId` comparison rather than an attribute lookup.
fn box_collect(inst: &PackedInstruction) -> Option<&Collect> {
    let OperationRef::ControlFlow(cf) = inst.op.view() else {
        return None;
    };
    let ControlFlow::Box { annotations, .. } = &cf.control_flow else {
        return None;
    };
    annotations
        .iter()
        .find_map(|a| a.as_ref().downcast_ref::<Collect>())
}

/// Whether this instruction is an emission.
pub fn is_emission(inst: &PackedInstruction) -> bool {
    matches!(
        inst.op.view(),
        OperationRef::CustomOperation(op) if op.downcast_ref::<Emit>().is_some()
    )
}

/// The [`Emit`] on this instruction, if it is an emission.
pub fn emission_spec(inst: &PackedInstruction) -> Option<Emit> {
    match inst.op.view() {
        OperationRef::CustomOperation(op) => op.downcast_ref::<Emit>().cloned(),
        _ => None,
    }
}

/// The single body of a box instruction.
pub fn block_body<'a>(
    src: &'a DAGCircuit,
    inst: &PackedInstruction,
) -> Result<Option<&'a DAGCircuit>> {
    match inst.blocks_view() {
        [] => Ok(None),
        [block] => Ok(Some(&src.blocks()[*block])),
        _ => Err(SamplexError::BoxWithoutOneBody),
    }
}

// --- Addressing an instruction inside the nest --------------------------------------------------

/// The address of one instruction in a nested circuit: the box nodes descended through, outermost
/// first, then the node itself within that innermost scope.
///
/// A scope is a path of node indices rather than a `DAGCircuit` reference so that a site outlives the
/// borrow that produced it, which is what lets a pass plan against the whole circuit and rewrite it
/// afterwards.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Site {
    pub scope: Vec<NodeIndex>,
    pub node: NodeIndex,
}

impl Site {
    /// The DAG of the scope this site lives in.
    ///
    /// For the flat work a pass still does within one scope; anything that crosses a box boundary
    /// goes through a [`WireCursor`] instead.
    pub fn scope_dag<'a>(&self, root: &'a DAGCircuit) -> Result<&'a DAGCircuit> {
        scope_dag(root, &self.scope)
    }

    /// The instruction this site names.
    pub fn instruction<'a>(&self, root: &'a DAGCircuit) -> Result<&'a PackedInstruction> {
        Ok(self.scope_dag(root)?.dag()[self.node].unwrap_operation())
    }

    /// The body of the box this site names, if it has one.
    pub fn body<'a>(&self, root: &'a DAGCircuit) -> Result<Option<&'a DAGCircuit>> {
        let dag = self.scope_dag(root)?;
        block_body(dag, dag.dag()[self.node].unwrap_operation())
    }

    /// The [`Collect`] this site declares, if it is a collector.
    pub fn collector(&self, root: &DAGCircuit) -> Result<Option<Collect>> {
        Ok(collect_annotation(self.instruction(root)?))
    }

    /// The wires this site covers, in its own scope's frame.
    pub fn qubits(&self, root: &DAGCircuit) -> Result<Vec<Qubit>> {
        let dag = self.scope_dag(root)?;
        let inst = dag.dag()[self.node].unwrap_operation();
        Ok(dag.qargs_interner().get(inst.qubits).to_vec())
    }

    /// The wires this site covers, lifted into the frame of an enclosing scope.
    ///
    /// Content keeps its position in the frame of whatever takes it over, so content taken out of a
    /// nested body has to be lifted through every box between. `base` must name a scope this site
    /// sits at or below — which a site a walk reached always does, since a cursor never ascends — and
    /// anything else is an error rather than a silent remapping.
    pub fn qubits_in(&self, root: &DAGCircuit, base: &[NodeIndex]) -> Result<Vec<Qubit>> {
        if base.len() > self.scope.len() {
            return Err(SamplexError::LiftIntoDeeperScope);
        }
        lift_wires(
            scope_dag(root, base)?,
            &self.scope[base.len()..],
            &self.qubits(root)?,
        )
    }

    /// How many classical wires this site covers.
    pub fn num_clbits(&self, root: &DAGCircuit) -> Result<usize> {
        let dag = self.scope_dag(root)?;
        let inst = dag.dag()[self.node].unwrap_operation();
        Ok(dag.cargs_interner().get(inst.clbits).len())
    }

    /// Whether this site sits strictly inside the scope `base` names.
    ///
    /// Depth is the whole test, because a cursor never ascends: a site one reached is at or below the
    /// scope the walk started in, so it is inside exactly when it is deeper.
    pub fn deeper_than(&self, base: &[NodeIndex]) -> bool {
        self.scope.len() > base.len()
    }

    /// Replace the operation and body at this site.
    ///
    /// The body is added as a block of the site's own scope, so what the instruction was catching on
    /// its wires it goes on catching.
    pub fn substitute(
        &self,
        root: &mut DAGCircuit,
        op: PackedOperation,
        body: DAGCircuit,
    ) -> Result<()> {
        let dag = scope_dag_mut(root, &self.scope)?;
        let block = dag.add_block(body);
        dag.substitute_op(self.node, op, Some(Parameters::Blocks(vec![block])), None)?;
        Ok(())
    }

    /// Delete the instruction at this site from the scope it lives in.
    ///
    /// Every other site stays valid across a removal: a scope is a `StableDiGraph`, so the indices a
    /// plan is carrying do not shift under it.
    pub fn remove(&self, root: &mut DAGCircuit) -> Result<()> {
        scope_dag_mut(root, &self.scope)?.remove_op_node(self.node);
        Ok(())
    }
}

/// The DAG of the scope a path names.
pub fn scope_dag<'a>(root: &'a DAGCircuit, scope: &[NodeIndex]) -> Result<&'a DAGCircuit> {
    let mut dag = root;
    for node in scope {
        let inst = dag.dag()[*node].unwrap_operation();
        dag = block_body(dag, inst)?.ok_or(SamplexError::ScopeWithoutBody)?;
    }
    Ok(dag)
}

/// The DAG of the scope a path names, for writing.
///
/// Private: descending mutably is only ever wanted in order to rewrite at a site, and
/// [`Site::substitute`] and [`Site::remove`] are the two ways a pass does that.
fn scope_dag_mut<'a>(root: &'a mut DAGCircuit, scope: &[NodeIndex]) -> Result<&'a mut DAGCircuit> {
    let mut dag = root;
    for node in scope {
        let block = match dag.dag()[*node].unwrap_operation().blocks_view() {
            [block] => *block,
            _ => return Err(SamplexError::ScopeWithoutOneBody),
        };
        dag = dag.view_block_mut(block);
    }
    Ok(dag)
}

/// Map wires from the scope `path` names up into the scope the path starts in.
///
/// Private: a caller that has a [`Site`] wants [`Site::qubits_in`], which knows that the path to lift
/// through is the tail of the site's own scope. Getting that slice wrong is the whole hazard here.
fn lift_wires(root: &DAGCircuit, path: &[NodeIndex], wires: &[Qubit]) -> Result<Vec<Qubit>> {
    // Each box's qargs, in the frame of the scope containing it.
    let mut frames: Vec<Vec<Qubit>> = Vec::with_capacity(path.len());
    let mut dag = root;
    for node in path {
        let inst = dag.dag()[*node].unwrap_operation();
        frames.push(dag.qargs_interner().get(inst.qubits).to_vec());
        dag = block_body(dag, inst)?.ok_or(SamplexError::ScopeWithoutBody)?;
    }
    let mut lifted = wires.to_vec();
    for frame in frames.iter().rev() {
        for wire in &mut lifted {
            *wire = *frame
                .get(wire.index())
                .ok_or(SamplexError::ContentOffItsBoxWires)?;
        }
    }
    Ok(lifted)
}

// --- Sweeping every scope -----------------------------------------------------------------------

/// Which end of the nest a sweep reports first.
///
/// Both orders report every collector exactly once and take topological order within a scope; they
/// differ only in whether a scope comes before or after the scopes nested inside it. A pass picks the
/// one its rule needs: innermost-first gives a collector inside a box first refusal on the content in
/// there, outermost-first lets a collector be judged against the one nested below it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScopeOrder {
    Innermost,
    Outermost,
}

/// Every collector in the circuit, addressed by site.
///
/// Descends through content boxes and never into a collector, so a collector's own body is not swept:
/// what is in there already belongs to it.
pub fn collectors(root: &DAGCircuit, order: ScopeOrder) -> Result<Vec<Site>> {
    let mut sites = Vec::new();
    sweep_scope(root, &mut Vec::new(), order, &mut sites)?;
    Ok(sites)
}

fn sweep_scope(
    root: &DAGCircuit,
    path: &mut Vec<NodeIndex>,
    order: ScopeOrder,
    out: &mut Vec<Site>,
) -> Result<()> {
    // Materialized because the recursion re-borrows `root` to reach each nested scope.
    let nodes: Vec<NodeIndex> = scope_dag(root, path)?.topological_op_nodes(false).collect();
    if order == ScopeOrder::Innermost {
        sweep_bodies(root, path, order, &nodes, out)?;
    }
    for node in &nodes {
        if is_collector(scope_dag(root, path)?.dag()[*node].unwrap_operation()) {
            out.push(Site {
                scope: path.clone(),
                node: *node,
            });
        }
    }
    if order == ScopeOrder::Outermost {
        sweep_bodies(root, path, order, &nodes, out)?;
    }
    Ok(())
}

fn sweep_bodies(
    root: &DAGCircuit,
    path: &mut Vec<NodeIndex>,
    order: ScopeOrder,
    nodes: &[NodeIndex],
    out: &mut Vec<Site>,
) -> Result<()> {
    for node in nodes {
        let inst = scope_dag(root, path)?.dag()[*node].unwrap_operation();
        if !is_box(inst) || is_collector(inst) {
            continue;
        }
        path.push(*node);
        sweep_scope(root, path, order, out)?;
        path.pop();
    }
    Ok(())
}

// --- Walking a wire -----------------------------------------------------------------------------

/// The next operation node along one wire, or `None` at the end of it.
///
/// Reaching the wire's output node counts as the end. This scope only: a box on the wire is the node
/// reported, not something stepped into.
///
/// Private: adjacency along a wire is a step, and the two things a pass wants are a walk that crosses
/// box boundaries — [`WireCursor`] — and a reading of what this scope's wire holds — [`WireSights`].
/// Both are built from this, and neither leaves a caller to re-derive the other.
fn next_on_wire(
    dag: &DAGCircuit,
    from: NodeIndex,
    qubit: Qubit,
    direction: Direction,
) -> Option<NodeIndex> {
    // Per-wire, unlike `quantum_successors`, which pools every wire of a node together. "What does
    // this qubit see next" is the whole adjacency notion the IR2 passes need.
    let (search, wire) = match direction {
        Direction::Right => (PetDirection::Outgoing, Wire::Qubit(qubit)),
        Direction::Left => (PetDirection::Incoming, Wire::Qubit(qubit)),
    };
    let next = dag
        .dag()
        .edges_directed(from, search)
        .find(|edge| *edge.weight() == wire)
        .map(|edge| match direction {
            Direction::Right => edge.target(),
            Direction::Left => edge.source(),
        })?;
    matches!(dag.dag()[next], NodeType::Operation(_)).then_some(next)
}

/// A per-wire walk position that can descend into a box body and climb back out.
///
/// One wire, one cursor: a walk over several wires advances each independently, so a wire blocked
/// inside a box stops on its own while another goes on descending. A cursor climbs back out only of
/// boxes it descended into — it never ascends above the scope it started in, which is what keeps a
/// collector inside a box from reaching content outside it.
#[derive(Clone, Debug)]
pub struct WireCursor {
    /// The scope the walk started in. Never climbed out of, which is the whole of "descends, never
    /// ascends".
    base: Vec<NodeIndex>,
    /// The boxes descended through since the start, each with this wire's index in the scope
    /// *containing* that box.
    descent: Vec<(NodeIndex, Qubit)>,
    /// The node the cursor sits on, in the scope its path names. A wire's boundary node while
    /// mid-descent, which is why this is only ever used as a starting point for the next step.
    node: NodeIndex,
    /// This wire's index in that same scope.
    qubit: Qubit,
}

impl WireCursor {
    /// A cursor sitting at `site`, walking along one of the wires it covers.
    ///
    /// The site's scope becomes the walk's base, so the cursor will never report anything outside it.
    pub fn at(site: &Site, qubit: Qubit) -> Self {
        WireCursor {
            base: site.scope.clone(),
            descent: Vec::new(),
            node: site.node,
            qubit,
        }
    }

    /// Where the cursor is now.
    pub fn site(&self) -> Site {
        Site {
            scope: self.path(),
            node: self.node,
        }
    }

    /// The scope the cursor is currently in.
    fn path(&self) -> Vec<NodeIndex> {
        let mut path = self.base.clone();
        path.extend(self.descent.iter().map(|(node, _)| *node));
        path
    }

    /// The site this wire sees next, and the cursor that reaches it, without moving.
    ///
    /// A peek, not a step: the caller takes the probe only if it takes what the probe found, which is
    /// how a walk tests a candidate it is still free to refuse.
    pub fn peek(&self, root: &DAGCircuit, direction: Direction) -> Result<Option<(Self, Site)>> {
        let mut probe = self.clone();
        Ok(probe.advance(root, direction)?.map(|site| (probe, site)))
    }

    /// Advance one operation along this wire, descending into any content box and climbing back out at
    /// the end of a body.
    ///
    /// Returns the site reached, or `None` once the wire has run out in the scope the walk started in.
    /// A collect box is reported rather than descended into — it is a barrier, and that is not the
    /// caller's choice to make. A content box the cursor descends into and finds nothing on this wire
    /// is passed straight through: there is nothing there to reorder against, so it is not a barrier.
    pub fn advance(&mut self, root: &DAGCircuit, direction: Direction) -> Result<Option<Site>> {
        loop {
            let dag = scope_dag(root, &self.path())?;
            match next_on_wire(dag, self.node, self.qubit, direction) {
                Some(next) => {
                    let inst = dag.dag()[next].unwrap_operation();
                    if !is_box(inst) || is_collector(inst) {
                        self.node = next;
                        return Ok(Some(self.site()));
                    }
                    let body =
                        block_body(dag, inst)?.ok_or(SamplexError::DescentIntoBodylessBox)?;
                    // The walk only offers nodes adjacent along this wire, so a box reached this way
                    // covers it.
                    let local = dag
                        .qargs_interner()
                        .get(inst.qubits)
                        .iter()
                        .position(|q| *q == self.qubit)
                        .map(|index| Qubit(index as u32))
                        .expect("a box reached along a wire covers that wire");
                    // Enter at the far end of the body from the direction of travel, so the first
                    // step inside lands on the operation nearest the boundary just crossed.
                    let boundary = match direction {
                        Direction::Right => body.qubit_io_map()[local.index()][0],
                        Direction::Left => body.qubit_io_map()[local.index()][1],
                    };
                    self.descent.push((next, self.qubit));
                    self.node = boundary;
                    self.qubit = local;
                }
                // End of the wire in this scope: climb out of the box it was inside, or stop.
                None => match self.descent.pop() {
                    Some((box_node, outer)) => {
                        self.node = box_node;
                        self.qubit = outer;
                    }
                    None => return Ok(None),
                },
            }
        }
    }
}

// --- What a walk sees ---------------------------------------------------------------------------

/// What stands at one position, as an IR2 rule sees it.
///
/// The whole of what the absorption and escape rules ever ask about an instruction, and deliberately no
/// more: with the question narrowed to this, a rule is a function of plain data rather than of a
/// `DAGCircuit`, so stating the case one decides costs a literal rather than a nested circuit and an
/// interpreter to build it in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sighting {
    /// A gate that can belong to an easy run: a single-qubit standard gate, which is the only kind a
    /// dressing multiplies in.
    EasyGate,
    /// An emission, and the direction it is still travelling. `None` once it has resolved in place,
    /// owned by the collector body it sits in rather than propagating towards one.
    Emission(Option<Direction>),
    /// A collect box.
    CollectBox,
    /// Hard content: anything else. No collector absorbs it, and nothing is moved across it.
    HardContent,
}

impl Sighting {
    /// What this instruction is.
    ///
    /// Takes no DAG, which is what keeps the rules downstream free of one: a node's qargs and its
    /// operation always agree on how many wires it covers, so an easy gate's width comes off the
    /// operation rather than out of an interner.
    pub fn of(inst: &PackedInstruction) -> Self {
        match inst.op.view() {
            OperationRef::CustomOperation(op) => match op.downcast_ref::<Emit>() {
                Some(emit) => Sighting::Emission(emit.direction),
                None => Sighting::HardContent,
            },
            OperationRef::StandardGate(gate) if gate.num_qubits() == 1 => Sighting::EasyGate,
            OperationRef::ControlFlow(_) if is_collector(inst) => Sighting::CollectBox,
            _ => Sighting::HardContent,
        }
    }
}

/// What one wire sees ahead of a position, in the order it sees it.
///
/// This scope only: a box is one sighting rather than something stepped into, so a rule reading these
/// judges the run of content it is standing in and never something nested inside it. [`WireCursor`] is
/// the walk that crosses box boundaries; this is the reading that stays where it is.
///
/// Lazy, because every rule over it stops at the first sighting that settles the question, and the rest
/// of the wire is then nobody's business.
pub struct WireSights<'a> {
    dag: &'a DAGCircuit,
    at: NodeIndex,
    wire: Qubit,
    direction: Direction,
}

impl<'a> WireSights<'a> {
    /// What `wire` sees ahead of the node `from`, travelling `direction`.
    pub fn ahead(dag: &'a DAGCircuit, from: NodeIndex, wire: Qubit, direction: Direction) -> Self {
        WireSights {
            dag,
            at: from,
            wire,
            direction,
        }
    }
}

impl Iterator for WireSights<'_> {
    type Item = Sighting;

    fn next(&mut self) -> Option<Sighting> {
        // The end of the wire ends the sequence and keeps ending it: from the last operation,
        // `next_on_wire` reports the same nothing every time.
        self.at = next_on_wire(self.dag, self.at, self.wire, self.direction)?;
        Some(Sighting::of(self.dag.dag()[self.at].unwrap_operation()))
    }
}

/// How many emissions a body holds, by the direction each is still travelling.
///
/// Two rules count emissions and neither can use the other's count, so the difference between them is
/// the two readings here rather than an argument: absorption asks whether the body it is folding out of
/// has a twirl point of its own, and an escape asks how many emissions would arrive at the collector it
/// is folding into, wherever in the box they were written. Once counted, both are these same three
/// numbers, and the rules over them need no circuit.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EmissionTally {
    left: usize,
    right: usize,
    /// Resolved in place, so travelling nowhere and arriving at no collector.
    resolved: usize,
}

impl EmissionTally {
    /// The emissions written directly in this body.
    ///
    /// This scope only, deliberately. A nested box's emissions are fenced by that box's own collectors
    /// — build writes them before its emissions on either side — so they resolve inside it and never
    /// reach a gate in the enclosing body. Counting them anyway would make a `ChangeBasis` box wrapping
    /// a twirled one look occupied by the inner box's twirl point.
    pub fn here(dag: &DAGCircuit) -> Self {
        dag.op_nodes(true)
            .map(|(_, inst)| Sighting::of(inst))
            .collect()
    }

    /// Every emission in this body and in every body nested inside it, at any depth.
    ///
    /// A collect box is descended into as well, unlike everywhere else in this module: what one holds
    /// has resolved in place, so it adds to no direction and needs no exception made for it.
    pub fn subtree(dag: &DAGCircuit) -> Result<Self> {
        let mut tally = Self::default();
        for (_, inst) in dag.op_nodes(true) {
            if let Sighting::Emission(direction) = Sighting::of(inst) {
                tally.count(direction);
                continue;
            }
            if let Some(body) = block_body(dag, inst)? {
                tally.add(Self::subtree(body)?);
            }
        }
        Ok(tally)
    }

    /// How many are still travelling in `direction`, and so still looking for a collector that way.
    pub fn towards(&self, direction: Direction) -> usize {
        match direction {
            Direction::Left => self.left,
            Direction::Right => self.right,
        }
    }

    /// Whether the body holds any emission at all, travelling or resolved — that is, whether it has a
    /// twirl point in it.
    pub fn any(&self) -> bool {
        self.left + self.right + self.resolved > 0
    }

    fn count(&mut self, direction: Option<Direction>) {
        match direction {
            Some(Direction::Left) => self.left += 1,
            Some(Direction::Right) => self.right += 1,
            None => self.resolved += 1,
        }
    }

    /// Add a nested body's tally to this one. An emission arrives where it arrives regardless of how
    /// many boxes it was written inside.
    fn add(&mut self, nested: Self) {
        self.left += nested.left;
        self.right += nested.right;
        self.resolved += nested.resolved;
    }
}

impl FromIterator<Sighting> for EmissionTally {
    fn from_iter<I: IntoIterator<Item = Sighting>>(sightings: I) -> Self {
        let mut tally = Self::default();
        for sighting in sightings {
            if let Sighting::Emission(direction) = sighting {
                tally.count(direction);
            }
        }
        tally
    }
}

// --- Writing ------------------------------------------------------------------------------------

/// Create an empty `DAGCircuit` body with the given dimensions and anonymous wires.
pub fn new_dag_body(num_qubits: usize, num_clbits: usize, capacity: usize) -> Result<DAGCircuit> {
    // Anonymous because a box body's qubits are positional, addressed only through the box's qargs,
    // so there is nothing outside for them to be identified with. `with_capacity` reserves space
    // but registers no wires, hence the explicit adds.
    let mut body =
        DAGCircuit::with_capacity(num_qubits, num_clbits, None, Some(capacity), None, None);
    for _ in 0..num_qubits {
        body.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
    }
    for _ in 0..num_clbits {
        body.add_clbit_unchecked(ShareableClbit::new_anonymous())?;
    }
    Ok(body)
}

/// Append an operation to the back of a DAG under construction.
pub fn append(
    out: &mut DAGCircuitBuilder,
    op: PackedOperation,
    params: Option<Parameters<Block>>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
    // Exists to keep `apply_operation_back`'s `cache_pygates` argument in one place: everything
    // samplex appends is built from a `PackedOperation`, never a live Python object, so there is
    // never a cached gate to pass. `CircuitData::push_packed_operation` is the same convenience on
    // the flat side.
    out.apply_operation_back(
        op,
        qargs,
        cargs,
        params,
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;
    Ok(())
}

/// Append a copy of an instruction, on the wires given rather than on its own.
///
/// Moving an instruction somewhere else is always this pair of steps — clone the operation, carry its
/// parameters across — and every caller wants both, so neither is offered separately.
pub fn append_instruction(
    out: &mut DAGCircuitBuilder,
    inst: &PackedInstruction,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
    let params = (!inst.params_view().is_empty())
        .then(|| Parameters::Params(inst.params_view().iter().cloned().collect()));
    append(out, inst.op.clone(), params, qargs, cargs)
}

/// Append a `box` with this body and these annotations.
///
/// The one place a box is written, so the one place its annotations become `Arc<dyn Annotation>`. Its
/// counterpart is [`collect_annotation`]: together they are the whole of how samplex puts a
/// declaration on a box and gets it back, and neither needs a `Python` token to do it.
pub fn write_box(
    out: &mut DAGCircuitBuilder,
    body: DAGCircuit,
    annotations: Vec<Arc<dyn Annotation>>,
    duration: Option<BoxDuration>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
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

/// A collect box of the given width, carrying this annotation.
///
/// Any other annotation and any duration on the box being replaced are dropped, which is sound only
/// because `build::write_collect` is the sole minter of collect boxes and gives them exactly one
/// annotation and no duration. A collect box that ever needs to carry a second annotation has to
/// revisit this.
pub fn collect_op(annotation: Collect, num_qubits: usize, num_clbits: usize) -> PackedOperation {
    PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: None,
            annotations: vec![Arc::new(annotation)],
        },
        num_qubits: num_qubits as u32,
        num_clbits: num_clbits as u32,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;

    use crate::annotated_circuit::{DistributionType, Dressing, SynthesizerType, Twirl};
    use crate::distributions::DistKey;
    use crate::emission_circuit::{CollectPart, EmitPart};
    use crate::partition::Partition;
    use qiskit_circuit::annotation::Annotation;
    use qiskit_circuit::operations::StandardGate;

    /// An annotation from outside samplex's vocabulary.
    ///
    /// It never materializes into Python here, which is the point: these readers only compare
    /// `TypeId`s, so a foreign annotation costs them nothing and needs no interpreter.
    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }

        fn create_py_annotation(&self, _py: Python) -> PyResult<Py<PyAny>> {
            unimplemented!("these tests never cross into Python")
        }
    }

    /// A one-qubit `box` carrying exactly these annotations.
    ///
    /// The interner keys are the default empty-slice ones and the block list is empty: these readers
    /// only look at `inst.op`, so anything else would be fixture that no assertion depends on.
    fn box_instruction(annotations: Vec<Arc<dyn Annotation>>) -> PackedInstruction {
        PackedInstruction::from_control_flow(
            ControlFlowInstruction {
                control_flow: ControlFlow::Box {
                    duration: None,
                    annotations,
                },
                num_qubits: 1,
                num_clbits: 0,
            },
            Vec::new(),
            Default::default(),
            Default::default(),
            None,
        )
    }

    /// A one-qubit emission travelling `direction`, or resolved in place if `None`.
    fn emit_spec(direction: Option<Direction>) -> Emit {
        Emit {
            direction,
            partition: Partition::singletons(1),
            parts: vec![EmitPart {
                dist: DistKey(0),
                draw: 0,
                adjoint: false,
            }],
        }
    }

    /// A one-qubit emission as an instruction, with the same empty interner keys as
    /// [`box_instruction`] and for the same reason.
    fn emission_instruction(direction: Option<Direction>) -> PackedInstruction {
        PackedInstruction::from_custom_operation(
            emit_spec(direction),
            Default::default(),
            Default::default(),
            None,
        )
    }

    fn collect_spec() -> Collect {
        Collect {
            partition: Partition::singletons(1),
            parts: vec![CollectPart {
                synthesizer: SynthesizerType::RzSx,
            }],
        }
    }

    fn twirl_spec() -> Twirl {
        Twirl {
            distribution: DistributionType::UniformPauli,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        }
    }

    /// What a content box carries in the walk fixtures below.
    fn content_annotations() -> Vec<Arc<dyn Annotation>> {
        vec![Arc::new(twirl_spec())]
    }

    /// What a collect box carries in the walk fixtures below.
    fn collect_annotations() -> Vec<Arc<dyn Annotation>> {
        vec![Arc::new(collect_spec())]
    }

    /// The site of the first standard gate of this kind in a scope.
    fn gate_site(dag: &DAGCircuit, scope: Vec<NodeIndex>, gate: StandardGate) -> Site {
        let node = dag
            .topological_op_nodes(false)
            .find(|node| {
                matches!(
                    dag.dag()[*node].unwrap_operation().op.view(),
                    OperationRef::StandardGate(found) if found == gate
                )
            })
            .expect("the fixture should hold that gate");
        Site { scope, node }
    }

    /// The node of the first box in a scope.
    fn box_node(dag: &DAGCircuit) -> NodeIndex {
        dag.topological_op_nodes(false)
            .find(|node| is_box(dag.dag()[*node].unwrap_operation()))
            .expect("the fixture should hold a box")
    }

    /// The standard gate a site names, if it names one, for saying where a walk landed.
    fn gate_at(root: &DAGCircuit, site: &Site) -> Option<StandardGate> {
        match site.instruction(root).unwrap().op.view() {
            OperationRef::StandardGate(gate) => Some(gate),
            _ => None,
        }
    }

    /// One qubit: `h`, then a content box holding `x` then `y`, then `z`.
    fn content_nest() -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(1, 0, 3).unwrap().into_builder();
        append(&mut out, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();
        let mut body = new_dag_body(1, 0, 2).unwrap().into_builder();
        append(&mut body, StandardGate::X.into(), None, &[Qubit(0)], &[]).unwrap();
        append(&mut body, StandardGate::Y.into(), None, &[Qubit(0)], &[]).unwrap();
        write_box(
            &mut out,
            body.build(),
            content_annotations(),
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();
        append(&mut out, StandardGate::Z.into(), None, &[Qubit(0)], &[]).unwrap();
        out.build()
    }

    /// One qubit: `h`, then a collect box holding `x`, then `z`.
    fn collector_nest() -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(1, 0, 3).unwrap().into_builder();
        append(&mut out, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();
        let mut body = new_dag_body(1, 0, 1).unwrap().into_builder();
        append(&mut body, StandardGate::X.into(), None, &[Qubit(0)], &[]).unwrap();
        write_box(
            &mut out,
            body.build(),
            collect_annotations(),
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();
        append(&mut out, StandardGate::Z.into(), None, &[Qubit(0)], &[]).unwrap();
        out.build()
    }

    /// Two qubits: `h` and `z` on qubit 0, and between them a two-qubit content box whose body only
    /// touches qubit 1.
    fn box_missing_a_wire() -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(2, 0, 3).unwrap().into_builder();
        append(&mut out, StandardGate::H.into(), None, &[Qubit(0)], &[]).unwrap();
        let mut body = new_dag_body(2, 0, 1).unwrap().into_builder();
        append(&mut body, StandardGate::Y.into(), None, &[Qubit(1)], &[]).unwrap();
        write_box(
            &mut out,
            body.build(),
            content_annotations(),
            None,
            &[Qubit(0), Qubit(1)],
            &[],
        )
        .unwrap();
        append(&mut out, StandardGate::Z.into(), None, &[Qubit(0)], &[]).unwrap();
        out.build()
    }

    /// One qubit: a collect box with a collector in its body, then a content box with a collector in
    /// its body.
    ///
    /// The collector inside the collector is the fixture's point: a collect box's body holds only what
    /// was already absorbed into it, so a sweep must not report what is in there.
    fn nested_collectors() -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(1, 0, 2).unwrap().into_builder();
        for annotations in [collect_annotations(), content_annotations()] {
            let mut body = new_dag_body(1, 0, 1).unwrap().into_builder();
            write_box(
                &mut body,
                new_dag_body(1, 0, 0).unwrap(),
                collect_annotations(),
                None,
                &[Qubit(0)],
                &[],
            )
            .unwrap();
            write_box(&mut out, body.build(), annotations, None, &[Qubit(0)], &[]).unwrap();
        }
        out.build()
    }

    /// One qubit: an emission travelling rightward, then a content box holding an emission travelling
    /// leftward and a collect box with a resolved emission in it.
    ///
    /// Three depths and three answers, which is what separates the two tally readings.
    fn emissions_at_depth() -> DAGCircuit {
        Python::initialize();
        let mut out = new_dag_body(1, 0, 2).unwrap().into_builder();
        append_emission(&mut out, Some(Direction::Right));
        let mut body = new_dag_body(1, 0, 2).unwrap().into_builder();
        append_emission(&mut body, Some(Direction::Left));
        let mut absorbed = new_dag_body(1, 0, 1).unwrap().into_builder();
        append_emission(&mut absorbed, None);
        write_box(
            &mut body,
            absorbed.build(),
            collect_annotations(),
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();
        write_box(
            &mut out,
            body.build(),
            content_annotations(),
            None,
            &[Qubit(0)],
            &[],
        )
        .unwrap();
        out.build()
    }

    fn append_emission(out: &mut DAGCircuitBuilder, direction: Option<Direction>) {
        let op = PackedOperation::from_custom_operation(Box::new(emit_spec(direction)));
        append(out, op, None, &[Qubit(0)], &[]).unwrap();
    }

    #[test]
    fn test_collect_annotation_reads_spec() {
        let spec = collect_spec();
        let inst = box_instruction(vec![Arc::new(spec.clone())]);
        assert_eq!(collect_annotation(&inst), Some(spec));
        assert!(is_collector(&inst));
        assert!(is_box(&inst));
    }

    #[test]
    fn test_is_collector_rejects_content_box() {
        // A content box is still a box, and it still carries a samplex annotation. What it is not is a
        // collector — the distinction every IR2 walk descends on.
        let inst = box_instruction(vec![Arc::new(twirl_spec())]);
        assert!(is_box(&inst));
        assert!(!is_collector(&inst));
        assert_eq!(collect_annotation(&inst), None);
    }

    #[test]
    fn test_collect_annotation_ignores_foreign_annotation() {
        // A stranger's annotation riding along on a collector must not hide the collector, and on its
        // own must not fabricate one.
        let spec = collect_spec();
        let shared = box_instruction(vec![Arc::new(Foreign), Arc::new(spec.clone())]);
        assert_eq!(collect_annotation(&shared), Some(spec));

        let foreign_only = box_instruction(vec![Arc::new(Foreign)]);
        assert!(is_box(&foreign_only));
        assert!(!is_collector(&foreign_only));
    }

    #[test]
    fn test_is_box_rejects_a_gate() {
        let gate = PackedInstruction::from_standard_gate(StandardGate::H, None, Default::default());
        assert!(!is_box(&gate));
        assert!(!is_collector(&gate));
        assert_eq!(collect_annotation(&gate), None);
    }

    #[test]
    fn test_collect_op_writes_one_annotation_and_no_duration() {
        // Pins what `collect_op`'s doc comment claims, and what both callers rely on when they replace
        // a collector wholesale: exactly one annotation, no duration, the width it was asked for.
        let spec = collect_spec();
        let op = collect_op(spec.clone(), 2, 1);
        let OperationRef::ControlFlow(cf) = op.view() else {
            panic!("collect_op must produce a control-flow operation");
        };
        assert_eq!(cf.num_qubits, 2);
        assert_eq!(cf.num_clbits, 1);
        let ControlFlow::Box {
            duration,
            annotations,
        } = &cf.control_flow
        else {
            panic!("collect_op must produce a box");
        };
        assert!(duration.is_none());
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].downcast_ref::<Collect>(), Some(&spec));
    }

    #[test]
    fn test_deeper_than_compares_depth_alone() {
        // Sites are never compared by path, only by depth, because a walk that reached one can only
        // have gone down. Two unrelated scopes of equal depth are therefore neither inside the other.
        let root = Site {
            scope: Vec::new(),
            node: NodeIndex::new(1),
        };
        let nested = Site {
            scope: vec![NodeIndex::new(1)],
            node: NodeIndex::new(2),
        };
        assert!(nested.deeper_than(&root.scope));
        assert!(!root.deeper_than(&root.scope));
        assert!(!root.deeper_than(&nested.scope));
        assert!(!nested.deeper_than(&nested.scope));
    }

    #[test]
    fn test_cursor_descends_into_content_and_climbs_back_out() {
        // The walk's whole reach: into a content box, along its body, out again, and on to whatever
        // follows the box in the scope the walk started in.
        let root = content_nest();
        let start = gate_site(&root, Vec::new(), StandardGate::H);
        let mut cursor = WireCursor::at(&start, Qubit(0));

        let x = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &x), Some(StandardGate::X));
        assert!(
            x.deeper_than(&start.scope),
            "descended into the content box"
        );

        let y = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &y), Some(StandardGate::Y));

        let z = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &z), Some(StandardGate::Z));
        assert!(!z.deeper_than(&start.scope), "climbed back out of it");

        assert!(cursor.advance(&root, Direction::Right).unwrap().is_none());
    }

    #[test]
    fn test_cursor_never_ascends_above_its_base() {
        // The other half of the asymmetry, and the load-bearing half: a walk started inside a box stops
        // at the end of that body rather than reporting the gate outside. A collector in there must not
        // reach out and take an enclosing box's content, and this is what stops it.
        let root = content_nest();
        let scope = vec![box_node(&root)];
        let body = scope_dag(&root, &scope).unwrap();

        let mut rightward =
            WireCursor::at(&gate_site(body, scope.clone(), StandardGate::X), Qubit(0));
        let y = rightward.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &y), Some(StandardGate::Y));
        assert!(
            rightward
                .advance(&root, Direction::Right)
                .unwrap()
                .is_none(),
            "the z after the box is outside the walk's base"
        );

        let mut leftward = WireCursor::at(&gate_site(body, scope, StandardGate::Y), Qubit(0));
        let x = leftward.advance(&root, Direction::Left).unwrap().unwrap();
        assert_eq!(gate_at(&root, &x), Some(StandardGate::X));
        assert!(
            leftward.advance(&root, Direction::Left).unwrap().is_none(),
            "the h before the box is outside it too"
        );
    }

    #[test]
    fn test_cursor_reports_a_collector_rather_than_descending() {
        // A collect box is a barrier: the walk names it and carries on past it, and never offers the
        // content it has already absorbed.
        let root = collector_nest();
        let start = gate_site(&root, Vec::new(), StandardGate::H);
        let mut cursor = WireCursor::at(&start, Qubit(0));

        let collector = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert!(collector.collector(&root).unwrap().is_some());
        assert!(!collector.deeper_than(&start.scope));

        let after = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &after), Some(StandardGate::Z));
    }

    #[test]
    fn test_cursor_passes_through_a_box_missing_this_wire() {
        // A box this wire crosses without meeting anything inside is not a barrier: there is nothing in
        // there to reorder against, so the walk comes straight out the other side.
        let root = box_missing_a_wire();
        let start = gate_site(&root, Vec::new(), StandardGate::H);
        let mut cursor = WireCursor::at(&start, Qubit(0));

        let reached = cursor.advance(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(gate_at(&root, &reached), Some(StandardGate::Z));
        assert!(!reached.deeper_than(&start.scope));
    }

    #[test]
    fn test_peek_leaves_the_cursor_where_it_was() {
        let root = content_nest();
        let cursor = WireCursor::at(&gate_site(&root, Vec::new(), StandardGate::H), Qubit(0));
        let (probe, first) = cursor.peek(&root, Direction::Right).unwrap().unwrap();
        let (_, again) = cursor.peek(&root, Direction::Right).unwrap().unwrap();
        assert_eq!(first, again, "peeking twice finds the same site");
        assert_eq!(probe.site(), first, "the probe is the cursor that took it");
    }

    #[test]
    fn test_collectors_sweeps_innermost_scopes_first() {
        let root = nested_collectors();
        let sites = collectors(&root, ScopeOrder::Innermost).unwrap();
        assert_eq!(
            sites.len(),
            2,
            "the collector inside a collector is not swept"
        );
        for site in &sites {
            assert!(site.collector(&root).unwrap().is_some());
        }
        assert_eq!(sites[0].scope.len(), 1, "the one inside the content box");
        assert_eq!(sites[1].scope.len(), 0, "then the one at the root");
    }

    #[test]
    fn test_collectors_sweeps_outermost_scopes_first() {
        let root = nested_collectors();
        let sites = collectors(&root, ScopeOrder::Outermost).unwrap();
        assert_eq!(sites.len(), 2);
        assert_eq!(sites[0].scope.len(), 0, "the root collector comes first");
        assert_eq!(sites[1].scope.len(), 1);
    }

    #[test]
    fn test_qubits_in_rejects_a_base_deeper_than_the_site() {
        // Lifting only ever goes outwards. A base that is not an ancestor is a caller mistake, and
        // saying so beats remapping wires against a frame they do not belong to.
        Python::initialize();
        let root = new_dag_body(1, 0, 0).unwrap();
        let site = Site {
            scope: Vec::new(),
            node: NodeIndex::new(0),
        };
        assert!(site.qubits_in(&root, &[NodeIndex::new(0)]).is_err());
    }

    #[test]
    fn test_sighting_reads_what_a_rule_asks_about() {
        // No DAG anywhere in here, which is the whole reason sightings exist: the four cases a rule
        // distinguishes are read off an instruction alone.
        assert_eq!(
            Sighting::of(&PackedInstruction::from_standard_gate(
                StandardGate::H,
                None,
                Default::default()
            )),
            Sighting::EasyGate
        );
        assert_eq!(
            Sighting::of(&PackedInstruction::from_standard_gate(
                StandardGate::CX,
                None,
                Default::default()
            )),
            Sighting::HardContent,
            "a dressing multiplies in single-qubit gates only"
        );
        assert_eq!(
            Sighting::of(&box_instruction(collect_annotations())),
            Sighting::CollectBox
        );
        assert_eq!(
            Sighting::of(&box_instruction(content_annotations())),
            Sighting::HardContent,
            "a content box is one opaque thing to a wire reading its own scope"
        );
        assert_eq!(
            Sighting::of(&emission_instruction(Some(Direction::Left))),
            Sighting::Emission(Some(Direction::Left))
        );
        assert_eq!(
            Sighting::of(&emission_instruction(None)),
            Sighting::Emission(None),
            "an absorbed emission has resolved in place and travels nowhere"
        );
    }

    #[test]
    fn test_wire_sights_stay_in_their_own_scope() {
        // The other reading of the same nest that `WireCursor` descends into: a box is one sighting,
        // reported and passed over, so a rule judging a run never sees the run inside a box.
        let root = content_nest();
        let start = gate_site(&root, Vec::new(), StandardGate::H);
        let ahead: Vec<Sighting> =
            WireSights::ahead(&root, start.node, Qubit(0), Direction::Right).collect();
        assert_eq!(
            ahead,
            vec![Sighting::HardContent, Sighting::EasyGate],
            "the content box, then the z after it — not the x and y inside"
        );
    }

    #[test]
    fn test_emission_tally_here_ignores_a_nested_body() {
        // The reading absorption makes: a box's own twirl point, never a nested box's, which that
        // box's own collectors have already fenced.
        let root = emissions_at_depth();
        let tally = EmissionTally::here(&root);
        assert!(tally.any());
        assert_eq!(tally.towards(Direction::Right), 1);
        assert_eq!(
            tally.towards(Direction::Left),
            0,
            "the nested one is not here"
        );
    }

    #[test]
    fn test_emission_tally_subtree_counts_at_any_depth() {
        // The reading an escape makes: every emission that will arrive at the collector being folded
        // into, however deep in the box it was written, and a collector's own body descended into as
        // well because what is in there has resolved.
        let root = emissions_at_depth();
        let tally = EmissionTally::subtree(&root).unwrap();
        assert_eq!(tally.towards(Direction::Right), 1);
        assert_eq!(tally.towards(Direction::Left), 1, "the one inside the box");
        assert_eq!(
            tally,
            [
                Sighting::Emission(Some(Direction::Right)),
                Sighting::Emission(Some(Direction::Left)),
                Sighting::Emission(None),
            ]
            .into_iter()
            .collect(),
            "the resolved one is counted too, and towards nothing"
        );
    }

    #[test]
    fn test_emission_tally_counts_only_emissions() {
        // Collected straight from sightings, which is how a rule's test states the case it decides
        // without a circuit to imply it.
        let tally: EmissionTally = [Sighting::EasyGate, Sighting::CollectBox]
            .into_iter()
            .collect();
        assert!(!tally.any());
        assert_eq!(tally.towards(Direction::Left), 0);
    }
}
