// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Lower an annotated circuit into emits and collects.

use std::sync::Arc;

use hashbrown::HashSet;
use thiserror::Error;

use qiskit_circuit::annotation::Annotation;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::dag_circuit::{
    DAGCircuit, DAGCircuitBuilder, DAGError, DuplicateWireError, NodeIndex,
};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    BoxDuration, ControlFlow, ControlFlowInstruction, ControlFlowView, CustomOperation, Operation,
    OperationRef,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Block, BlocksMode, Clbit, Qubit, VarsMode};

use super::Direction;
use super::annotations::{
    DressingSide, InjectNoise, InjectionSite, InvalidAnnotationError, RandomizationAnnotations,
    Twirl, annotation_kind,
};
use super::distributions::Distribution;
use super::emissions::{Collect, Emit, EmitError, Hem};
use super::navigation::{ContentKind, op_at};
use super::partition::Partition;
use super::synthesizer::Synthesizer;

/// Why a circuit could not be lowered.
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("only `box` control flow can be randomized, not `{0}`")]
    UnsupportedControlFlow(String),
    #[error("qubit {0} is out of its scope")]
    QubitOutOfScope(usize),
    #[error("clbit {0} is out of its scope")]
    ClbitOutOfScope(usize),

    #[error(transparent)]
    Emit(#[from] EmitError),
    #[error(transparent)]
    InvalidAnnotation(#[from] InvalidAnnotationError),
    #[error(transparent)]
    Dag(#[from] DAGError),
    #[error(transparent)]
    DuplicateWire(#[from] DuplicateWireError),
}

pub type Result<T> = std::result::Result<T, BuildError>;

/// Lower an annotated circuit into emits and collects.
pub fn lower_annotations(dag: &DAGCircuit) -> Result<DAGCircuit> {
    let mut out = dag
        .copy_empty_like_with_capacity(dag.num_ops(), 0, VarsMode::Alike, BlocksMode::Drop)
        .into_builder();
    let scope = Scope::identity(dag.num_qubits(), dag.num_clbits());
    Build.walk(dag, &mut out, &scope)?;
    Ok(out.build())
}

/// Where in a lowered box an emit happens.
///
/// A box is written as `Start | easy run | Between | hard content | End` under a left dressing and
/// `Start | hard content | Between | easy run | End` under a right one. `Between` is the twirl
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    Start,
    Between,
    End,
}

impl Location {
    /// The direction to the nearer collect.
    fn outward(self, dressing: DressingSide) -> Direction {
        match (self, dressing) {
            (Location::Start, _) => Direction::Backward,
            (Location::End, _) => Direction::Forward,
            (Location::Between, DressingSide::Left) => Direction::Backward,
            (Location::Between, DressingSide::Right) => Direction::Forward,
        }
    }
}

/// An operation to be inserted onto the given wires.
pub(crate) struct Wired<T> {
    pub(crate) op: T,
    /// The wires to place the operation on as indices into its box's qargs.
    pub(crate) wires: Vec<u32>,
}

/// A box's emits ordered by location.
struct EmitLayout {
    /// Which side of the hard content a collect could absorb into.
    dressing: DressingSide,
    runs: [Vec<Wired<Emit>>; 3],
}

impl EmitLayout {
    /// An empty layout for a box with this dressing.
    fn new(dressing: DressingSide) -> Self {
        EmitLayout {
            dressing,
            runs: Default::default(),
        }
    }

    /// Take the emits at `location`.
    fn take(&mut self, location: Location) -> Vec<Wired<Emit>> {
        std::mem::take(&mut self.runs[location as usize])
    }

    /// Add an emit at `location` inside everything already there.
    fn push_inward(&mut self, location: Location, emit: Emit, wires: Vec<u32>) {
        let dressing = self.dressing;
        let placed = Wired { op: emit, wires };
        let run = &mut self.runs[location as usize];
        if location.outward(dressing) == Direction::Backward {
            run.push(placed);
        } else {
            run.insert(0, placed);
        }
    }
}

/// How a walked scope's bits map into the output being written.
struct Scope {
    qubits: Vec<usize>,
    clbits: Vec<usize>,
}

impl Scope {
    /// The map a scope of this shape writes into itself.
    fn identity(num_qubits: usize, num_clbits: usize) -> Self {
        Scope {
            qubits: (0..num_qubits).collect(),
            clbits: (0..num_clbits).collect(),
        }
    }

    /// The map a box's own wires take into the scope around it.
    fn within(qargs: &[Qubit], cargs: &[Clbit]) -> Self {
        Scope {
            qubits: qargs.iter().map(|qubit| qubit.index()).collect(),
            clbits: cargs.iter().map(|clbit| clbit.index()).collect(),
        }
    }

    fn out_qubits(&self, locals: &[Qubit]) -> Result<Vec<Qubit>> {
        locals
            .iter()
            .map(|q| {
                self.qubits
                    .get(q.index())
                    .map(|&i| Qubit(i as u32))
                    .ok_or(BuildError::QubitOutOfScope(q.index()))
            })
            .collect()
    }

    fn out_clbits(&self, locals: &[Clbit]) -> Result<Vec<Clbit>> {
        locals
            .iter()
            .map(|c| {
                self.clbits
                    .get(c.index())
                    .map(|&i| Clbit(i as u32))
                    .ok_or(BuildError::ClbitOutOfScope(c.index()))
            })
            .collect()
    }
}

/// State carried across the whole walk.
///
/// Carries nothing yet.
struct Build;

impl Build {
    /// Lower every operation in `dag` into `out`.
    fn walk(&mut self, dag: &DAGCircuit, out: &mut DAGCircuitBuilder, scope: &Scope) -> Result<()> {
        self.walk_nodes(dag, dag.topological_op_nodes(false), out, scope)
    }

    /// Lower `nodes` of `dag` into `out`, lowering an annotated box in place.
    fn walk_nodes(
        &mut self,
        dag: &DAGCircuit,
        nodes: impl IntoIterator<Item = NodeIndex>,
        out: &mut DAGCircuitBuilder,
        scope: &Scope,
    ) -> Result<()> {
        for node in nodes {
            let inst = op_at(dag, node);
            match inst.op.view() {
                OperationRef::ControlFlow(_) => self.walk_box(dag, inst, out, scope)?,
                _ => copy_instruction(dag, inst, out, scope)?,
            }
        }
        Ok(())
    }

    /// Lower one annotated box, dissolving as much of it as the user did not ask to keep.
    ///
    /// A kept box holds its hard content and nothing else. See [`KeptBox`].
    fn walk_box(
        &mut self,
        dag: &DAGCircuit,
        inst: &PackedInstruction,
        out: &mut DAGCircuitBuilder,
        scope: &Scope,
    ) -> Result<()> {
        // One view of the instruction, which is where the body and everything the box declares
        // both come from. A box is the only control flow we randomize so anything else is refused.
        let Some(ControlFlowView::Box {
            duration,
            annotations,
            body,
        }) = dag.try_view_control_flow(inst)
        else {
            return Err(BuildError::UnsupportedControlFlow(
                inst.op.view().name().to_string(),
            ));
        };
        let randomization = RandomizationAnnotations::new(annotations)?;
        let foreign: Vec<Arc<dyn Annotation>> = annotations
            .iter()
            .filter(|annotation| annotation_kind(annotation.as_ref()).is_none())
            .cloned()
            .collect();

        let out_qargs = scope.out_qubits(dag.qargs_interner().get(inst.qubits))?;
        let out_cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;
        let kept = KeptBox::declared(foreign, duration.cloned(), &out_qargs, &out_cargs);
        let inner = Scope::within(&out_qargs, &out_cargs);

        let Some(twirl) = randomization.twirl else {
            // No twirl point so there is no easy run to separate from the hard content. The body
            // is one run which either dissolves into the circuit or stays whole in its box.
            let Some(kept) = kept else {
                return self.walk(body, out, &inner);
            };
            let whole: Vec<NodeIndex> = body.topological_op_nodes(false).collect();
            let content = self.boxed_run(body, &whole, &kept)?;
            return kept.append(out, content);
        };

        let width = out_qargs.len();
        let synthesizer = twirl.decomposition.resolve(width as u32);
        let layout = self.build_emits(twirl, randomization.inject_noise, width)?;
        // A box gets a collect on each side and both start empty, so the second is built rather
        // than cloned from the first. The synthesizer is shared because it is compared by value and
        // holds none of what a collect gathers.
        append_wired(out, collect_wires(width, &synthesizer), &out_qargs)?;
        self.content(body, layout, out, &inner, &out_qargs, kept)?;
        append_wired(out, collect_wires(width, &synthesizer), &out_qargs)?;
        Ok(())
    }

    /// Walk a box's body into the circuit around it with its emits where they belong.
    ///
    /// Only the hard content can need a box. The easy run and the emits are written around it.
    fn content(
        &mut self,
        body: &DAGCircuit,
        mut layout: EmitLayout,
        out: &mut DAGCircuitBuilder,
        inner: &Scope,
        qargs: &[Qubit],
        mut kept: Option<KeptBox>,
    ) -> Result<()> {
        let (easy_nodes, hard_nodes) = classify_body(body, layout.dressing);
        // The absorbable run goes on the dressing side of the twirl point and the hard content on
        // the other, so a right dressing is the same walk with the two runs exchanged.
        let runs: [Run<'_>; 2] = match layout.dressing {
            DressingSide::Left => [Run::Easy(&easy_nodes), Run::Hard(&hard_nodes)],
            DressingSide::Right => [Run::Hard(&hard_nodes), Run::Easy(&easy_nodes)],
        };
        append_wired(out, layout.take(Location::Start), qargs)?;
        // The hard run is written once, so the box it may go in is handed over once.
        self.write_run(body, runs[0], out, inner, &mut kept)?;
        append_hem_band(out, layout.take(Location::Between), qargs)?;
        self.write_run(body, runs[1], out, inner, &mut kept)?;
        append_wired(out, layout.take(Location::End), qargs)?;
        Ok(())
    }

    /// Write one of a body's two runs, in its own box if it is hard content the user boxed.
    fn write_run(
        &mut self,
        body: &DAGCircuit,
        run: Run,
        out: &mut DAGCircuitBuilder,
        inner: &Scope,
        kept: &mut Option<KeptBox>,
    ) -> Result<()> {
        if let Run::Hard(nodes) = run
            && let Some(kept) = kept.take()
        {
            let content = self.boxed_run(body, nodes, &kept)?;
            return kept.append(out, content);
        }
        self.walk_nodes(body, run.nodes().iter().copied(), out, inner)
    }

    /// Copy a run of a body's nodes into a fresh body as wide as the box they stay in.
    fn boxed_run(
        &mut self,
        body: &DAGCircuit,
        nodes: &[NodeIndex],
        kept: &KeptBox,
    ) -> Result<DAGCircuit> {
        let (num_qubits, num_clbits) = (kept.qargs.len(), kept.cargs.len());
        let within = Scope::identity(num_qubits, num_clbits);
        let mut builder = new_dag_body(num_qubits, num_clbits, nodes.len())?.into_builder();
        self.walk_nodes(body, nodes.iter().copied(), &mut builder, &within)?;
        Ok(builder.build())
    }

    /// Turn a box's annotations into the emits they declare at the locations they go at.
    fn build_emits(
        &mut self,
        twirl: &Twirl,
        injection: Option<&InjectNoise>,
        box_width: usize,
    ) -> Result<EmitLayout> {
        let mut layout = EmitLayout::new(twirl.dressing);
        if let Some(injection) = injection {
            self.noise_emits(injection, box_width, &mut layout)?;
        }
        self.twirl_emits(twirl, box_width, &mut layout)?;
        Ok(layout)
    }

    /// The inverse pair a twirl declares over its box.
    ///
    /// One emit each way, each placing every factor of the one sample.
    fn twirl_emits(
        &mut self,
        twirl: &Twirl,
        box_width: usize,
        layout: &mut EmitLayout,
    ) -> Result<()> {
        let distribution = twirl
            .distribution
            .resolve(across(box_width))
            .expect("a kind resolves over one factor per qubit");
        let location = Location::Between;
        let outward = location.outward(layout.dressing);

        let (near, wires) = place_whole_sample(&distribution, outward, false)?;
        layout.push_inward(location, near, wires);
        let (far, wires) = place_whole_sample(&distribution, outward.reverse(), true)?;
        layout.push_inward(location, far, wires);
        Ok(())
    }

    /// The emit a noise injection declares, placing every factor of one sample.
    fn noise_emits(
        &mut self,
        injection: &InjectNoise,
        box_width: usize,
        layout: &mut EmitLayout,
    ) -> Result<()> {
        let dressing = layout.dressing;
        // A site names a side of the hard content. `Between` is the twirl point itself and the
        // other two are a box end beyond the content, which is the side the dressing decides.
        let location = match (injection.site, dressing) {
            (InjectionSite::Between, _) => Location::Between,
            (InjectionSite::Before, DressingSide::Left) => Location::Between,
            (InjectionSite::Before, DressingSide::Right) => Location::Start,
            (InjectionSite::After, DressingSide::Left) => Location::End,
            (InjectionSite::After, DressingSide::Right) => Location::Between,
        };
        let distribution = injection
            .resolve(across(box_width))
            .expect("a noise map resolves over one factor per qubit");
        let (emit, wires) = place_whole_sample(&distribution, location.outward(dressing), false)?;
        layout.push_inward(location, emit, wires);
        Ok(())
    }
}

/// One factor per qubit of a box this wide.
///
/// What a distribution resolves over until something derives a factorization of its own.
fn across(box_width: usize) -> Partition<u32> {
    Partition::singletons(0..box_width as u32).expect("a range of qubits has no repeats")
}

/// An emit placing every factor of one sample, with the box wires it goes on.
fn place_whole_sample(
    distribution: &Arc<dyn Distribution>,
    direction: Direction,
    adjoint: bool,
) -> Result<(Emit, Vec<u32>)> {
    let factors: Vec<u32> = (0..distribution.num_factors() as u32).collect();
    let wires = emit_wires(distribution, &factors);
    let emit = Emit::new(direction, Arc::clone(distribution), factors, adjoint)?;
    Ok((emit, wires))
}

/// The box wires an emit's qargs take, placing `factors` of `distribution`.
///
/// The factorization read in `factors` order, which may permute the qubits.
fn emit_wires(distribution: &Arc<dyn Distribution>, factors: &[u32]) -> Vec<u32> {
    factors
        .iter()
        .flat_map(|&factor| {
            distribution
                .factor_qubits(factor)
                .expect("an emit places factors its distribution has")
                .iter()
                .copied()
        })
        .collect()
}

/// The one collect a box side gets, spanning every wire of the box.
fn collect_wires(box_width: usize, synthesizer: &Arc<dyn Synthesizer>) -> Vec<Wired<Collect>> {
    vec![Wired {
        op: Collect::new(Arc::clone(synthesizer)),
        wires: (0..box_width as u32).collect(),
    }]
}

/// One of the two runs a box body splits into at its twirl point.
#[derive(Clone, Copy)]
enum Run<'a> {
    /// What a dressing can fold in.
    Easy(&'a [NodeIndex]),
    /// What stays where it is.
    Hard(&'a [NodeIndex]),
}

impl<'a> Run<'a> {
    /// The nodes of this run in the order they are written.
    fn nodes(self) -> &'a [NodeIndex] {
        match self {
            Run::Easy(nodes) | Run::Hard(nodes) => nodes,
        }
    }
}

/// Sweep a box body from the dressing edge to split the absorbable run from the rest.
///
/// Which gates are absorbable depends on the dressing, since absorbability is ancestry as much as
/// it is the gate itself. The sweep poisons a wire at the first gate it cannot fold.
fn classify_body(body: &DAGCircuit, dressing: DressingSide) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    let right = matches!(dressing, DressingSide::Right);
    let mut order: Vec<NodeIndex> = body.topological_op_nodes(false).collect();
    if right {
        order.reverse();
    }

    let mut poisoned: HashSet<usize> = HashSet::new();
    let mut easy_nodes = Vec::new();
    let mut hard_nodes = Vec::new();
    for node in order {
        let inst = op_at(body, node);
        let qargs = body.qargs_interner().get(inst.qubits);
        let absorbable =
            ContentKind::of(inst).is_absorbable() && !poisoned.contains(&qargs[0].index());
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

/// The box the user asked to keep, which the hard content stays in.
///
/// A box survives lowering when it carries an annotation we do not interpret or a duration. It
/// holds the hard content alone and is kept even where that is empty.
struct KeptBox<'a> {
    annotations: Vec<Arc<dyn Annotation>>,
    duration: Option<BoxDuration>,
    /// The wires the user wrote the box on, which the kept box keeps.
    qargs: &'a [Qubit],
    cargs: &'a [Clbit],
}

impl<'a> KeptBox<'a> {
    /// The box to keep, or `None` if the box carries nothing that needs one.
    fn declared(
        annotations: Vec<Arc<dyn Annotation>>,
        duration: Option<BoxDuration>,
        qargs: &'a [Qubit],
        cargs: &'a [Clbit],
    ) -> Option<Self> {
        let needed = !annotations.is_empty() || duration.is_some();
        needed.then(|| KeptBox {
            annotations,
            duration,
            qargs,
            cargs,
        })
    }

    /// Append this box holding `content`.
    fn append(self, out: &mut DAGCircuitBuilder, content: DAGCircuit) -> Result<()> {
        append_box(
            out,
            content,
            self.annotations,
            self.duration,
            self.qargs,
            self.cargs,
        )
    }
}

// --- Appending ------------------------------------------------------------------------------------

/// Create an empty body with the given dimensions and anonymous wires.
fn new_dag_body(num_qubits: usize, num_clbits: usize, capacity: usize) -> Result<DAGCircuit> {
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
fn append(
    out: &mut DAGCircuitBuilder,
    op: PackedOperation,
    params: Option<Parameters<Block>>,
    qargs: &[Qubit],
    cargs: &[Clbit],
) -> Result<()> {
    out.apply_operation_back(op, qargs, cargs, params, None)?;
    Ok(())
}

/// Copy an instruction verbatim into `out` and remap its bits through `scope`.
fn copy_instruction(
    dag: &DAGCircuit,
    inst: &PackedInstruction,
    out: &mut DAGCircuitBuilder,
    scope: &Scope,
) -> Result<()> {
    let qargs = scope.out_qubits(dag.qargs_interner().get(inst.qubits))?;
    let cargs = scope.out_clbits(dag.cargs_interner().get(inst.clbits))?;
    let own = inst.params_view();
    let params = (!own.is_empty()).then(|| Parameters::Params(own.iter().cloned().collect()));
    append(out, inst.op.clone(), params, &qargs, &cargs)
}

/// Append operations in the order given, each onto its own box wires.
fn append_wired<T: CustomOperation + 'static>(
    out: &mut DAGCircuitBuilder,
    wired: Vec<Wired<T>>,
    qargs: &[Qubit],
) -> Result<()> {
    for Wired { op, wires } in wired {
        let on: Vec<Qubit> = wires.iter().map(|wire| qargs[*wire as usize]).collect();
        let op = PackedOperation::from_custom_operation(Box::new(op));
        append(out, op, None, &on, &[])?;
    }
    Ok(())
}

/// Append the twirl point, with the emits standing either side of it.
///
/// The hem goes between the halves heading one way and the halves heading the other.
fn append_hem_band(
    out: &mut DAGCircuitBuilder,
    emits: Vec<Wired<Emit>>,
    qargs: &[Qubit],
) -> Result<()> {
    let (backward, forward): (Vec<_>, Vec<_>) = emits
        .into_iter()
        .partition(|wired| wired.op.direction == Direction::Backward);
    append_wired(out, backward, qargs)?;
    append_hems(out, qargs)?;
    append_wired(out, forward, qargs)
}

/// Append one hem per wire of the box.
fn append_hems(out: &mut DAGCircuitBuilder, qargs: &[Qubit]) -> Result<()> {
    for wire in qargs {
        let op = PackedOperation::from_custom_operation(Box::new(Hem));
        append(out, op, None, &[*wire], &[])?;
    }
    Ok(())
}

/// Append a `box` with this body and these annotations.
fn append_box(
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use qiskit_circuit::bit::QuantumRegister;
    use qiskit_circuit::operations::StandardGate;

    use crate::randomization::SynthesizerKind;
    use crate::randomization::distributions::{DistributionKind, Given, Noise};
    use crate::randomization::navigation::as_emit;

    pub(crate) fn circuit(width: u32, fill: impl FnOnce(&mut DAGCircuitBuilder)) -> DAGCircuit {
        let mut dag = DAGCircuit::new();
        dag.add_qreg(QuantumRegister::new_owning("q", width))
            .expect("a fresh register");
        let mut builder = dag.into_builder();
        fill(&mut builder);
        builder.build()
    }

    pub(crate) fn entangler(builder: &mut DAGCircuitBuilder, control: u32, target: u32) {
        builder
            .apply_operation_back(
                PackedOperation::from_standard_gate(StandardGate::CX),
                &[Qubit(control), Qubit(target)],
                &[],
                None,
                None,
            )
            .expect("an entangler on qubits the circuit has");
    }

    pub(crate) fn gate(builder: &mut DAGCircuitBuilder, gate: StandardGate, qubit: u32) {
        builder
            .apply_operation_back(
                PackedOperation::from_standard_gate(gate),
                &[Qubit(qubit)],
                &[],
                None,
                None,
            )
            .expect("a gate on a qubit the circuit has");
    }

    pub(crate) fn annotated_box(
        builder: &mut DAGCircuitBuilder,
        qubits: &[u32],
        annotations: Vec<Arc<dyn Annotation>>,
        body: DAGCircuit,
    ) {
        let qargs: Vec<Qubit> = qubits.iter().copied().map(Qubit).collect();
        append_box(builder, body, annotations, None, &qargs, &[]).expect("a box on real wires");
    }

    pub(crate) fn twirl(
        distribution: DistributionKind,
        dressing: DressingSide,
    ) -> Arc<dyn Annotation> {
        Arc::new(Twirl {
            distribution,
            dressing,
            decomposition: SynthesizerKind::RzSx,
        })
    }

    pub(crate) fn noise(reference: &str, site: InjectionSite) -> Arc<dyn Annotation> {
        Arc::new(InjectNoise {
            reference: reference.to_string(),
            modifier: None,
            site,
        })
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }
    }

    pub(crate) fn foreign() -> Arc<dyn Annotation> {
        Arc::new(Foreign)
    }

    fn name(inst: &PackedInstruction) -> String {
        inst.op.view().name().to_string()
    }

    fn ops_in_order(dag: &DAGCircuit) -> Vec<&PackedInstruction> {
        dag.topological_op_nodes(false)
            .map(|node| op_at(dag, node))
            .collect()
    }

    fn on_wire(dag: &DAGCircuit, wire: Qubit) -> Vec<&PackedInstruction> {
        ops_in_order(dag)
            .into_iter()
            .filter(|inst| dag.qargs_interner().get(inst.qubits).contains(&wire))
            .collect()
    }

    fn names_on_wire(dag: &DAGCircuit, wire: Qubit) -> Vec<String> {
        on_wire(dag, wire).into_iter().map(name).collect()
    }

    fn shape(dag: &DAGCircuit) -> Vec<(String, Vec<u32>)> {
        let mut out = Vec::new();
        for inst in ops_in_order(dag) {
            let qubits: Vec<u32> = dag
                .qargs_interner()
                .get(inst.qubits)
                .iter()
                .map(|qubit| qubit.0)
                .collect();
            out.push((name(inst), qubits));
            if let [block] = inst.blocks_view() {
                out.extend(shape(&dag.blocks()[*block]));
            }
        }
        out
    }

    fn wired_of<T: CustomOperation + Clone + 'static>(dag: &DAGCircuit) -> Vec<Wired<T>> {
        let mut out = Vec::new();
        for inst in ops_in_order(dag) {
            if let OperationRef::CustomOperation(op) = inst.op.view()
                && let Some(found) = op.downcast_ref::<T>()
            {
                out.push(Wired {
                    op: found.clone(),
                    wires: dag
                        .qargs_interner()
                        .get(inst.qubits)
                        .iter()
                        .map(|qubit| qubit.0)
                        .collect(),
                });
            }
            if let [block] = inst.blocks_view() {
                out.extend(wired_of(&dag.blocks()[*block]));
            }
        }
        out
    }

    fn emits(dag: &DAGCircuit) -> Vec<Wired<Emit>> {
        wired_of(dag)
    }

    fn collects(dag: &DAGCircuit) -> Vec<Wired<Collect>> {
        wired_of(dag)
    }

    fn injections(dag: &DAGCircuit) -> Vec<Wired<Emit>> {
        emits(dag)
            .into_iter()
            .filter(|wired| wired.op.distribution.downcast_ref::<Noise>().is_some())
            .collect()
    }

    fn twirl_halves(dag: &DAGCircuit) -> Vec<Wired<Emit>> {
        emits(dag)
            .into_iter()
            .filter(|wired| wired.op.distribution.downcast_ref::<Noise>().is_none())
            .collect()
    }

    #[test]
    fn test_a_twirled_box_gets_collects_and_a_pair() {
        let body = circuit(1, |b| gate(b, StandardGate::H, 0));
        let lowered = lower_annotations(&circuit(1, |b| {
            annotated_box(
                b,
                &[0],
                vec![twirl(DistributionKind::UniformPauli, DressingSide::Left)],
                body,
            )
        }))
        .expect("a twirled box lowers");

        assert_eq!(collects(&lowered).len(), 2, "one collect on each side");
        let pair = emits(&lowered);
        assert_eq!(pair.len(), 2, "a twirl is one pair");
        assert!(
            pair[0].op.same_sample(&pair[1].op),
            "one sample, two halves"
        );
        assert_ne!(pair[0].op.adjoint, pair[1].op.adjoint);
        assert_ne!(pair[0].op.direction, pair[1].op.direction);
    }

    #[test]
    fn test_the_two_collects_of_a_box_share_one_synthesizer_but_gather_separately() {
        let lowered = lower_annotations(&circuit(2, |b| {
            annotated_box(
                b,
                &[0, 1],
                vec![twirl(DistributionKind::UniformPauli, DressingSide::Left)],
                circuit(2, |b| entangler(b, 0, 1)),
            )
        }))
        .expect("a twirled box lowers");

        let pair = collects(&lowered);
        assert_eq!(pair.len(), 2);
        assert!(
            Arc::ptr_eq(&pair[0].op.synthesizer, &pair[1].op.synthesizer),
            "one synthesizer resolved once for the box"
        );
        assert_eq!(pair[0].wires, vec![0, 1]);
        assert_eq!(pair[1].wires, vec![0, 1]);
        assert!(pair[0].op.is_empty() && pair[1].op.is_empty());
    }

    #[test]
    #[ignore = "no correlated distribution is resolvable yet, so nothing factorizes into anything but singletons; revive with DistributionKind::UniformLocalC1"]
    fn test_a_correlated_distribution_is_placed_one_factor_at_a_time() {
        unimplemented!("revive with a correlated DistributionKind");
    }

    #[test]
    fn test_an_injection_is_one_emit_placing_every_factor_of_one_map() {
        let lowered = lower_annotations(&circuit(3, |b| {
            annotated_box(
                b,
                &[0, 1, 2],
                vec![
                    twirl(DistributionKind::UniformPauli, DressingSide::Left),
                    noise("layer_1", InjectionSite::Before),
                ],
                circuit(3, |_| {}),
            )
        }))
        .expect("noise beside a twirl lowers");

        let injected = injections(&lowered);
        assert_eq!(injected.len(), 1, "one sample, however wide the map");
        let emit = &injected[0].op;
        assert_eq!(emit.factors(), [0, 1, 2], "one factor per qubit of the box");
        assert_eq!(emit.distribution.num_factors(), 3);
        assert_eq!(injected[0].wires, vec![0, 1, 2]);
        let map = emit
            .distribution
            .downcast_ref::<Noise>()
            .expect("a noise map");
        assert_eq!(&*map.reference, "layer_1");
    }

    #[test]
    fn test_an_injection_without_a_twirl_is_refused() {
        let error = lower_annotations(&circuit(1, |b| {
            annotated_box(
                b,
                &[0],
                vec![noise("layer_1", InjectionSite::Before)],
                circuit(1, |_| {}),
            )
        }))
        .expect_err("an injection has no collect to resolve at without a twirl");
        assert!(matches!(
            error,
            BuildError::InvalidAnnotation(InvalidAnnotationError::InjectionWithoutTwirl)
        ));
    }

    #[test]
    fn test_an_emit_travels_towards_the_collect_it_resolves_at() {
        for (dressing, outward) in [
            (DressingSide::Left, Direction::Backward),
            (DressingSide::Right, Direction::Forward),
        ] {
            let lowered = lower_annotations(&circuit(1, |b| {
                annotated_box(
                    b,
                    &[0],
                    vec![twirl(DistributionKind::UniformPauli, dressing)],
                    circuit(1, |b| gate(b, StandardGate::H, 0)),
                )
            }))
            .expect("a twirled box lowers");

            let pair = emits(&lowered);
            assert_eq!(pair.len(), 2, "{dressing:?}");
            let near = pair
                .iter()
                .find(|wired| !wired.op.adjoint)
                .expect("the pair has a near half");
            assert_eq!(near.op.direction, outward, "{dressing:?}");
            let far = pair
                .iter()
                .find(|wired| wired.op.adjoint)
                .expect("the pair has a far half");
            assert_eq!(far.op.direction, outward.reverse(), "{dressing:?}");
        }
    }

    #[test]
    fn test_an_injection_sits_where_its_site_says_and_faces_away_from_the_content() {
        for (site, outward) in [
            (InjectionSite::Before, Direction::Backward),
            (InjectionSite::After, Direction::Forward),
        ] {
            let lowered = lower_annotations(&circuit(1, |b| {
                annotated_box(
                    b,
                    &[0],
                    vec![
                        twirl(DistributionKind::UniformPauli, DressingSide::Left),
                        noise("layer_1", site),
                    ],
                    circuit(1, |b| gate(b, StandardGate::H, 0)),
                )
            }))
            .expect("a noise injection lowers");

            let injected = injections(&lowered);
            assert_eq!(injected.len(), 1, "{site:?}");
            assert_eq!(injected[0].op.direction, outward, "{site:?}");
            assert!(!injected[0].op.adjoint, "an injection is never undone");
        }
    }

    #[test]
    fn test_a_boxs_emits_sit_together_at_the_twirl_point() {
        let lowered = lower_annotations(&circuit(2, |b| {
            annotated_box(
                b,
                &[0, 1],
                vec![
                    twirl(DistributionKind::UniformPauli, DressingSide::Left),
                    noise("layer_1", InjectionSite::Before),
                ],
                circuit(2, |b| entangler(b, 0, 1)),
            )
        }))
        .expect("a twirled box with an injection lowers");

        let names: Vec<String> = ops_in_order(&lowered)
            .into_iter()
            .map(name)
            .filter(|name| name != "collect")
            .collect();

        let hard = names
            .iter()
            .position(|name| name == "cx")
            .expect("the entangler stays hard");
        assert!(
            names[..hard]
                .iter()
                .all(|name| name == "emit" || name == "hem"),
            "every emit is at the hem: {names:?}"
        );
        assert_eq!(hard, names.len() - 1, "and the content follows them all");
        assert_eq!(
            hard, 5,
            "a twirl's pair, one noise emit, and a hem on each of the two wires"
        );

        assert_eq!(
            injections(&lowered).len(),
            1,
            "one noise emit over the box rather than one per wire"
        );
        let twirls = twirl_halves(&lowered);
        assert_eq!(
            twirls.len(),
            2,
            "one pair over the whole box rather than one per wire"
        );
        assert!(twirls[0].op.same_sample(&twirls[1].op), "of one sample");
        assert_ne!(twirls[0].op.adjoint, twirls[1].op.adjoint);
        assert_eq!(twirls[0].op.direction, twirls[1].op.direction.reverse());
        for half in &twirls {
            assert_eq!(half.op.factors(), [0, 1], "each placing every factor");
        }
    }

    #[test]
    fn test_the_absorbable_run_sits_on_the_dressing_side_of_the_twirl_point() {
        for (dressing, body, expected) in [
            (
                DressingSide::Left,
                circuit(2, |b| {
                    gate(b, StandardGate::H, 0);
                    entangler(b, 0, 1);
                }),
                vec!["h", "emit", "hem", "emit", "cx"],
            ),
            (
                DressingSide::Right,
                circuit(2, |b| {
                    entangler(b, 0, 1);
                    gate(b, StandardGate::H, 0);
                }),
                vec!["cx", "emit", "hem", "emit", "h"],
            ),
        ] {
            let lowered = lower_annotations(&circuit(2, |b| {
                annotated_box(
                    b,
                    &[0, 1],
                    vec![twirl(DistributionKind::UniformPauli, dressing)],
                    body,
                )
            }))
            .expect("a twirled box lowers");

            let on_one_wire: Vec<String> = names_on_wire(&lowered, Qubit(0))
                .into_iter()
                .filter(|name| name != "collect")
                .collect();
            assert_eq!(on_one_wire, expected, "{dressing:?}");
        }
    }

    #[test]
    fn test_a_hem_separates_the_halves_of_every_twirl() {
        for dressing in [DressingSide::Left, DressingSide::Right] {
            let lowered = lower_annotations(&circuit(2, |b| {
                annotated_box(
                    b,
                    &[0, 1],
                    vec![twirl(DistributionKind::UniformPauli, dressing)],
                    circuit(2, |b| entangler(b, 0, 1)),
                )
            }))
            .expect("a twirled box lowers");

            for wire in [Qubit(0), Qubit(1)] {
                let on_one_wire = on_wire(&lowered, wire);
                let positions = |wanted: &str| -> Vec<usize> {
                    on_one_wire
                        .iter()
                        .enumerate()
                        .filter(|(_, inst)| name(inst) == wanted)
                        .map(|(at, _)| at)
                        .collect()
                };

                let markers = positions("hem");
                assert_eq!(
                    markers.len(),
                    1,
                    "one marker per wire: {dressing:?} {wire:?}"
                );
                let marker = markers[0];

                let collects = positions("collect");
                assert_eq!(
                    collects,
                    vec![0, on_one_wire.len() - 1],
                    "{dressing:?} {wire:?}"
                );
                assert!(
                    collects[0] < marker && marker < collects[1],
                    "the marker stands between the pair: {dressing:?}"
                );

                for (at, inst) in on_one_wire.iter().enumerate() {
                    let Some(emit) = as_emit(inst) else { continue };
                    let heading = if at < marker {
                        Direction::Backward
                    } else {
                        Direction::Forward
                    };
                    assert_eq!(
                        emit.direction, heading,
                        "an emit at {at} with the marker at {marker}: {dressing:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_a_box_declaring_nothing_is_dissolved() {
        let body = circuit(1, |b| gate(b, StandardGate::H, 0));
        let lowered = lower_annotations(&circuit(1, |b| annotated_box(b, &[0], Vec::new(), body)))
            .expect("an unannotated box lowers");
        assert_eq!(shape(&lowered), vec![("h".to_string(), vec![0])]);
        assert!(emits(&lowered).is_empty() && collects(&lowered).is_empty());
    }

    #[test]
    fn test_emit_wires_follow_a_permuting_factorization_rather_than_assuming_ascending_wires() {
        let permuting: Arc<dyn Distribution> = Arc::new(Given::new([vec![2], vec![0], vec![1]]));
        assert_eq!(emit_wires(&permuting, &[0, 1, 2]), vec![2, 0, 1]);
        assert_eq!(
            emit_wires(&permuting, &[1, 0]),
            vec![0, 2],
            "the wires follow the factors in the order the emit places them"
        );
    }

    #[test]
    fn test_emit_wires_follow_a_factorization_covering_fewer_qubits_than_it_spans() {
        let partial: Arc<dyn Distribution> = Arc::new(Given::new([vec![3, 1], vec![0]]));
        assert_eq!(
            partial.num_qubits(),
            3,
            "three of the four it reaches across"
        );
        assert_eq!(emit_wires(&partial, &[0]), vec![3, 1]);
        assert_eq!(emit_wires(&partial, &[0, 1]), vec![3, 1, 0]);
    }

    #[test]
    fn test_a_box_we_do_not_randomize_is_kept_with_its_body_copied() {
        let inner = circuit(1, |b| gate(b, StandardGate::H, 0));
        let body = circuit(1, |b| {
            gate(b, StandardGate::S, 0);
            annotated_box(
                b,
                &[0],
                vec![twirl(DistributionKind::UniformPauli, DressingSide::Left)],
                inner,
            );
            gate(b, StandardGate::T, 0);
        });
        let lowered = lower_annotations(&circuit(1, |b| {
            annotated_box(b, &[0], vec![foreign()], body)
        }))
        .expect("a box with a foreign annotation lowers");

        assert_eq!(
            shape(&lowered),
            vec![
                ("box".to_string(), vec![0]),
                ("s".to_string(), vec![0]),
                ("collect".to_string(), vec![0]),
                ("h".to_string(), vec![0]),
                ("emit".to_string(), vec![0]),
                ("hem".to_string(), vec![0]),
                ("emit".to_string(), vec![0]),
                ("collect".to_string(), vec![0]),
                ("t".to_string(), vec![0]),
            ]
        );
        assert_eq!(emits(&lowered).len(), 2);
        assert_eq!(collects(&lowered).len(), 2);
    }

    #[test]
    fn test_a_box_we_keep_gets_the_hard_content_and_nothing_else() {
        let lowered = lower_annotations(&circuit(2, |b| {
            annotated_box(
                b,
                &[0, 1],
                vec![
                    foreign(),
                    twirl(DistributionKind::UniformPauli, DressingSide::Left),
                ],
                circuit(2, |b| {
                    gate(b, StandardGate::H, 0);
                    entangler(b, 0, 1);
                }),
            )
        }))
        .expect("a twirled box with a foreign annotation lowers");

        assert_eq!(
            names_on_wire(&lowered, Qubit(0)),
            ["collect", "h", "emit", "hem", "emit", "box", "collect"],
            "the box holds the hard content and sits inside the collects"
        );

        let body = ops_in_order(&lowered)
            .into_iter()
            .find_map(|inst| match inst.blocks_view() {
                [block] => Some(&lowered.blocks()[*block]),
                _ => None,
            })
            .expect("the box survived");
        assert_eq!(
            ops_in_order(body).into_iter().map(name).collect::<Vec<_>>(),
            ["cx"],
            "and holds only what cannot move"
        );
        assert_eq!(
            body.num_qubits(),
            2,
            "at the width the user wrote it, not the width of its content"
        );
    }

    #[test]
    fn test_control_flow_that_is_not_a_box_is_refused() {
        let dag = circuit(1, |builder| {
            builder
                .apply_operation_back(
                    PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
                        control_flow: ControlFlow::BreakLoop,
                        num_qubits: 1,
                        num_clbits: 0,
                    })),
                    &[Qubit(0)],
                    &[],
                    None,
                    None,
                )
                .expect("a break on a real wire");
        });
        let error = lower_annotations(&dag).expect_err("only a box can be randomized");
        assert!(matches!(error, BuildError::UnsupportedControlFlow(_)));
    }
}
