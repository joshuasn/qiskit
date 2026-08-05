# Samplex: IR chain and pass structure

Architecture for the samplex crate (`crates/samplex`). This document describes the intended design; see
`SAMPLEX_MIGRATION.md` for the staged work getting there from the current code.

> **Supersedes part of `.notebooks/design/contextual_collection.md`.** That document places the
> contextual-collection model inside the circuit *builder*. Here it is an optimization pass over IR2
> (`merge_collectors`), which is what allows it to rebuild rather than mutate. Its rules — two-sided
> collection, shared middle collectors, closing triggers — are still the specification of what that
> pass does.

## Organising principle

**IRs and objects each get their own file; anything that acts on an IR or moves between IRs is a pass.**
Everything is, or will be, a pass in a transpiler chain.

An "IR" is a legal op vocabulary over a shared circuit type — a dialect — not a bespoke Rust type per
stage. This matches how `crates/transpiler` works: one file per pass, plain free functions, no `Pass`
trait (see `crates/transpiler/src/passes/elide_permutations.rs`).

## The chain

```
QuantumCircuit
  --boxing (exists, Python)-->
IR1  annotated circuit        DAGCircuit + samplex annotations on boxes
                              declarative: what randomization is wanted
  --build-->
IR2  emission circuit         DAGCircuit dialect: Emit instructions, Collect boxes,
     + DistributionTable      hard-gate boxes. Positional: where randomization lives.
                              Nothing sampled, no parameters bound.
  --optimize (IR2 -> IR2)-->  absorb emissions, merge collectors, sink emissions, prune
  --lower-->
IR3  sampling graph           qubit-indexed dataflow over virtual state
     + template circuit       parameterized circuit; synth templates in place of collectors
  --optimize (IR3 -> IR3)-->  merge propagators, prune, infer virtual types
  --lower-->
IR4  register-level graph     (future)
```

Two IRs earn their existence because they admit **disjoint optimization sets**: merging collectors only
makes sense where position and template structure are visible, and merging propagators only makes sense
on a dataflow graph.

IR2 and IR3 are both qubit-indexed and declarative. The difference is **shape, not level**: IR2 is
ordered-with-scopes, which is what you need to reason about position, template construction and
parameter labelling; IR3 is a dataflow graph, which is what you need to reason about what commutes with
what. IR2 takes over the role samplomatic's `PreSamplex` played.

## Pass inventory

| stage | pass | signature |
|---|---|---|
| build | emission circuit construction | IR1 → IR2 |
| optimize | absorb emissions | IR2 → IR2 |
| optimize | merge collectors | IR2 → IR2 |
| optimize | sink emissions into boxes | IR2 → IR2 |
| optimize | prune vacuous collectors | IR2 → IR2 |
| lower | template construction + parameter labelling + graph construction | IR2 → IR3 |
| optimize | merge propagators | IR3 → IR3 |
| optimize | prune unreachable | IR3 → IR3 |
| optimize | infer virtual types | IR3 → IR3 |

## Two decisions that shape everything

**1. The build pass is purely local.** One collect box per side per annotated box, no cross-box state —
no qubit-to-collector map, no detach logic, no shared-middle-collector reasoning. Cross-scope wiring
moves into `absorb_emissions`; the contextual-collection model moves into `merge_collectors`.

This is what makes the design tractable. A merge pass is free to **rebuild** the circuit rather than
mutate it, which sidesteps a hard API limit: `substitute_node_with_dag`
(`crates/circuit/src/dag_circuit.rs:6755`) maps the replacement's qubits into the replaced node's
existing qargs and errors if the counts differ, so a box **cannot be widened in place**, and there is no
general API for inserting a wider op mid-circuit. Under merge-as-rebuild you never widen a box; you
re-emit a wider one. It also makes contextual collection an optimization you can switch off, so build
output is trivially checkable.

**2. Parameters are not assigned in IR2.** Merging changes how many collectors exist and how wide they
are, so any label assigned before the merge pass churns. An IR2 collect box carries a synthesizer and
the emission ids it consumes — nothing about parameters. Labelling happens in the IR2→IR3 lowering,
alongside template construction.

## Emissions and collection

An annotated box resolves to *emissions*, each a source of virtual gates:

- A `Twirl` yields **two** emissions — the inverse pair — sharing one distribution-table key and
  carrying opposite directions. Inversion is implied by the direction rather than recorded.
- `InjectNoise` yields one, on the side its `site` names. `ChangeBasis` / `InjectLocalClifford` yield
  one, on the side their `placement` names. Neither ever propagates through hard content.

An emission carries its own id, source kind, distribution key, direction, virtual type and subsystem
partition. A collect box carries a synthesizer and an **ordered** list of what it composes — the
emissions it consumes, interleaved with the gates it absorbed. See *Ordered collection* below.

**Propagation is derived from placement, not recorded.** An emission's state propagates through exactly
the real gates lying between it and the collect box naming it, walking in the emission's own direction
and filtering by qubit overlap. `Emit` instructions are skipped because they are not gates; a collect
box's virtual content is skipped for the same reason, while its absorbed gates *are* crossed.

**Where an emission is written is load-bearing, and it is not the same edge for every kind.**

- **A twirl's pair goes on the dressing edge**, both halves together, because that edge *is* the twirl
  point and the easy/hard split of the body is defined relative to it. Left dressing puts the pair before
  the hard box, so the right factor walks rightward through it; right dressing puts it after, so the left
  factor walks leftward through it. Getting this wrong silently inverts which factor is propagated.
- **A basis change or noise injection goes on the edge its own `placement` / `site` names**, which is
  where it physically happens. This differs from the dressing edge half the time — a
  `ChangeBasis(placement="end")` on a left-dressed box — and writing it on the dressing edge instead
  leaves the hard box between it and the collector consuming it, so the walk conjugates it by content it
  is meant to sit outside of. That was a real bug: the rule "neither ever propagates through hard
  content" was documented and only half implemented, and it held only when the two edges coincided.

So the box layout is `left collector, left-edge emissions, hard box, right-edge emissions, right
collector` — the dressing decides which group the twirl pair joins, not the order things are written in.

**Within one edge, emissions nest by how close they sit to the content.** The vocabulary implies the
order, and it fixes both the spine and each collector's composition order:

| depth | emission | why |
|---|---|---|
| 0, innermost | twirl pair | it *is* the easy/hard boundary |
| 1 | `InjectNoise`, `InjectLocalClifford` | happens *to the hard content*, so just outside the twirl point |
| 2, outermost | `ChangeBasis` | applies to the box as a whole, so it wraps everything |

The absorbed easy gates belong in this ordering too, at depth 1.5: they are part of the box's content,
so a frame change for the whole box wraps them, while anything attached to the hard content composes
nearer to it than they do. Read in circuit order that is outermost-first on the left edge and
innermost-first on the right, a left-dressed box with all of them reads

```
collector, basis start, easy gates, injections before, twirl pair, hard box,
           injections after, basis end, collector
```

and a right-dressed one is the mirror — `collector, basis start, injections before, hard box, twirl
pair, injections after, easy gates, basis end, collector`. The rule is side-agnostic; only which group
the twirl pair joins depends on the dressing.

### Ordered collection

**A collector records that order as data.** `CollectSpec.items` is one ordered sequence of emission
entries and `Gates(n)`, and IR3's `Collect.steps` is the same sequence with the gates inlined. Two
independent lists — collected emissions here, absorbed gates there — could not say that a basis change
composes outside the easy gates while a twirl composes inside them.

`Gates` carries a **count**, not an index range: merging concatenates bodies, and a count needs no
offsetting when it does, so a merge is just "concatenate the items, concatenate the bodies". A
well-formed collector's counts sum to its body length, which is worth asserting. A merged collector
holds one run per contribution, and because each side runs outermost-first/innermost-first respectively,
the two runs meet outermost-to-outermost — which is the correct composition order with no reordering.

### Local vs. incoming emissions (scope-agnostic)

**Collection is scope-agnostic.** The `absorb_emissions` pass scans from each `Emit` instruction in its
travel direction, crossing box boundaries recursively, and absorbs it into the **first compatible
collector** it reaches — regardless of which scope either lives in. Compatibility is determined by
`SynthesizerType::accepts(VirtualType)`.

The locality distinction is physical, not positional:

- **Local:** no gates between the emission and its target collector → `CollectItem::Emission(LocalEmission)`.
  The emission is removed from the spine.
- **Propagating:** gates lie between them → `CollectItem::Incoming(id)`. The emission stays on the spine
  as a standalone `Emit` instruction; the id wires it to the collector for graph construction.

The common cases:

| source | typical outcome |
|--------|-----------------|
| Twirl (near half) | local — adjacent to its collector, no gates between |
| Twirl (far half) | propagating — the hard box content separates it from the far collector |
| ChangeBasis | local — placed on the same edge as its collector |
| InjectNoise | local — placed on the same edge as its collector |

**Cross-scope absorption.** An outer emission can descend into a box to find a compatible inner
collector. An inner emission that finds no compatible collector in its scope escapes outward to an
outer collector. The scan crosses boundaries freely; the only stopping conditions are:
1. First compatible collector reached (absorb).
2. Incompatible collector encountered (stop — leave standalone).
3. End of circuit reached (standalone).

**Local emissions are data on the collector, not standalone instructions.** A `CollectItem::Emission`
carries the emission's distribution key, direction, virtual type, partition and source directly. It
never appears as an `Emit` instruction in the circuit, and never gets a VFG `Emission` node. At
sampling time the collector reads the sampled value from the distribution table and composes it at the
position `items` dictates — a direct table read, no graph traversal.

**Incoming emissions remain standalone `Emit` instructions.** A `CollectItem::Incoming(id)` references
a propagating emission by its id. That `Emit` instruction stays on the spine and walks in its
direction through the intervening gates, accumulating `Propagate` nodes on the way. The value arrives
at the Collect VFG node via graph edges; the id matches the incoming edge's source to the correct step
position.

```rust
pub struct LocalEmission {
    pub dist: DistIndex,
    pub direction: Direction,
    pub virtual_type: VirtualType,
    pub partition: Partition,
    pub source: EmitSource,
}

pub enum CollectItem {
    /// Adjacent emission, owned by this collector. Table read at sampling time.
    Emission(LocalEmission),
    /// Absorbed body gates at this position in the composition.
    Gates(usize),
    /// A propagating emission (far twirl half). The value arrives via graph edges.
    Incoming(u32),
}
```

IR3's `CollectStep` mirrors this split:

```rust
pub enum CollectStep {
    /// Read a value from the distribution table and compose it here.
    Local(LocalEmission),
    /// A value that arrived via graph edges after propagating through gates.
    Incoming(u32),
    /// A constant gate folded into the layer.
    Gate(AbsorbedGate),
}
```

**No VFG Emission node for local emissions.** They don't propagate, so there is no chain to model.
The Collect node's `Local` steps are self-contained: dist key → table → value → compose. The VFG
only has Emission nodes for the incoming (far twirl half) case, where the graph topology encodes which
gates conjugate the value and in what order.

In the common non-twirl case (a box annotated only with `ChangeBasis` or `InjectNoise`), the Collect
VFG node has **zero incoming edges** from emissions. It is entirely self-contained: table reads and
absorbed gates. Even a twirled box only produces one Emission node (the far half); the near half is a
local table read on the same-side collector.

**The locality test is whether any gates lie between the emission and its target.** The
`absorb_emissions` pass scans the circuit directionally from each emission. If it reaches its target
collector without crossing any gate content, the absorption is local. If gates intervene (typically the
hard box content for a far twirl half), it is propagating. Cross-scope absorption into a nested box is
local when the collector sits at the box's near edge with no gate content before it.

**Consequences for passes:**

- `build.rs`: writes ALL emissions as standalone `Emit` instructions, with collectors carrying only
  `Gates(n)` items. The build pass is purely local — no cross-box reasoning.
- `absorb_emissions.rs`: scans from each `Emit`, determines local vs. propagating, removes local
  emissions from the spine and adds them as `CollectItem::Emission`, wires propagating ones via
  `CollectItem::Incoming(id)`. Scope-agnostic: crosses box boundaries in both directions.
- `merge_collectors.rs`: unchanged in structure — concatenating items still works. Local emissions from
  different boxes concatenate as before. Incoming emissions keep their ids; the standalone instructions
  they reference still exist in the circuit.
- `lower.rs` template path: skips `Emit` instructions (they are not executable); local emissions
  produce no circuit instruction to skip.
- `lower.rs` graph path: does NOT create Emission VFG nodes for `Local` steps. Only standalone `Emit`
  instructions produce Emission VFG nodes, and only those get `walk_emission`. The Collect node's
  `Local` steps record what to read from the table at sampling time; the `Incoming` steps record which
  graph-edge values to compose at that position.

**Why the id on `Incoming` is still needed.** IR3's node indices are not stable under
`merge_parallel_nodes` and `prune`. The id is assigned once by `build` and never changes, so it is the
only stable handle for matching an incoming graph edge's source to the correct composition position in
the collector's steps.

**`InjectLocalClifford` is an injection, not a basis change**, even though `resolve_annotations` turns
both into a `ResolvedBasis`. Placement is the one thing they do not share, so that distinction has to
survive resolution — which is what `BasisOrigin` on `ResolvedBasis` records. `mode` cannot stand in for
it: `ChangeBasis(reference, mode="local_clifford")` is legal and produces an identical `ResolvedBasis`.

## Nesting semantics for twirled boxes

**Why an inner twirl is transparent.** To twirl content `U`, a twirl draws a random group element `P`
and inserts it before `U` with the compensating `Q = U P† U†` after, so that

```
Q U P  =  U P† U† U P  =  U P† P  =  U
```

The inserted pair *together with the content it wraps* equals the content. So an outer twirl's factor
propagating through an inner twirled box is conjugated by the inner box's **logical content only** — it
never sees the inner box's random element. This holds for any group (Pauli, C1, U2).

**The twirl point is the easy/hard boundary, not the box edge.** This matters for everything below. For
left dressing the left collector synthesizes `P · g_easy` — the drawn element composed with the absorbed
gates — and the right factor is `Q = conj(P, g_hard)`, propagated through the **hard** gates only. It
never touches its own box's easy gates, since those were folded into the same layer as `P`:

```
Q · g_hard · (P · g_easy)  =  g_hard · g_easy
```

**An enclosing emission propagates through a nested box's WHOLE content, not just its hard part.**

This is the same gate list read two different ways, and conflating them is the easiest way to get
nesting wrong:

| reader | how it treats a box's gates |
|---|---|
| the box's **own** emissions | split: easy gates *multiply* into the near factor, hard gates *propagate* the far one |
| an **enclosing** emission | one unit: the *whole logical content*, easy and hard alike, is propagated |

The asymmetry is because the split happens at the box's own twirl point, whereas all of the box —
including its easy gates — sits inside the *enclosing* box's twirl point.

So for left-dressed `T` containing left-dressed `I` whose body is `h; cx`, the gate `h` does **double
duty**:

- multiplied into `I`'s left factor, since it is absorbed into `I`'s left collector, and
- propagated for `T`'s right factor, since `Q_T = conj(P_T, h · cx · …)`.

Both roles are live at once. Build output handles this without any special case: a collector's body
still holds real gates there, so the propagation walk crosses them.

**Cross-scope absorption replaces the need for collector promotion.** With scope-agnostic absorption,
an outer emission can descend into a nested box and be absorbed by an inner collector directly — no
structural promotion needed. The outer far twirl half, if it finds a compatible inner collector at
the box's near edge (with no gates between them), is absorbed **locally** into that inner collector.
This achieves the same reduction in dressing layers without requiring segment structure.

Merging *siblings* is safe and unchanged — in a merged middle collector holding `[A_R, B_L]` with
`B`'s easy gates absorbed, those gates multiply into `B_L` and are not in `A`'s content, so each
absorbed gate has exactly one role.

**Merging collectors across box boundaries remains deferred.** The merge pass still recurses with fresh
state at box boundaries. Cross-scope *absorption* handles the common case (outer emission → inner
collector), while cross-scope *merging* (fusing an inner collector with an outer one, relocating
absorbed gates) would still require segment structure for recording propagation through relocated gates.
The win from cross-scope absorption covers the motivating nested-RB case without this complexity.

**What the build pass does for nesting:** recurse into nested annotated boxes rather than treating them
as opaque; the outer hard box contains the inner box's whole lowered form, so the propagation walk
descends into boxes; a nested annotated box trips the outer easy/hard latch. Nested boxes absorb
normally. Inner factors that cannot find a collector inside their scope escape outward to be absorbed by
an outer collector.

**The real limit on supported circuits is virtual-type preservation, not nesting.** An emission may only
propagate through gates for which a propagation rule exists for its virtual type — a Pauli stays a Pauli
through a Clifford, which is what samplomatic's `PAULI_PAST_CLIFFORD` tables encode. A gate on the path
with no rule for that virtual type must be a hard error in the lowering, not a silent wrong answer.

The motivating case is binary-RB-shaped: an outer box sampling a random local-C1 layer, a run of
back-to-back inner boxes containing only simple Cliffords, the induced Pauli tracked through them, and a
second local-C1 layer at the far end returning it to the computational basis. Note that if those inner
boxes carry no annotations they are fully transparent wrappers that build flattens outright — the
semi-transparency rule is what is needed once they are annotated.

## Control flow inside a twirled box

Not supported yet; non-box control flow is rejected. Recorded so the extension path is decided rather
than discovered. Everything follows from three invariants:

1. **Static content on propagation paths.** `Q = U P† U†` is computed at build time, so `U` must be
   known then. Runtime branching means it is not.
2. **Static parameter count.** The template circuit has a fixed parameter vector that the sampling graph
   fills, so the number of randomizations must be known at build time.
3. **Condition integrity.** A twirl must not alter a clbit that a control-flow condition reads, unless
   the induced flip is propagated into the condition. A Pauli twirl flips measurement outcomes, so a
   measurement inside a twirled region feeding a branch condition silently corrupts the branch decision.
   This is what samplomatic's `measurement_flips` / `twirled_clbits` / `verify_no_twirled_clbits`
   machinery exists for. Of the three this is the one that fails *silently*, so it needs to be a
   validation in lowering rather than a documented caveat.

**Control-flow ops are collection barriers.** No emission may propagate across a branch boundary — this
is invariant 1, and a stronger form of the trigger where a bare gate closes the open collector.

**`IfElseOp` / `SwitchCaseOp` violate only invariant 1, so they are supportable.** Dressing scheme:
share the outer dressing, put the inner dressing in each branch.

```
Collect_left  (collects P_L)
Emit(P_L), Emit(P_R)
IfElse {
  A: { A's gates, Collect (collects P_R) }
  B: { B's gates, Collect (collects P_R) }
}
```

One emission, n collectors, as a fan-out: `P_R` is propagated through `A` on one path and through `B` on
the other, both computed, one used at runtime. This costs one emission where twirling each branch
independently would cost n, and keeps a shared dressing layer.

It relaxes one thing in the model: **an emission may be named by more than one collector**, provided
those collectors lie on mutually exclusive control-flow paths. That is a well-formedness condition to
check, not just a convention.

**`ForLoopOp` violates invariant 2 unless the trip count is resolved.** With a static count, unrolling
fixes it, and unrolling is an ordinary transpiler pass — so it belongs as a *prerequisite* pass rather
than something the twirl handles. Keeping the loop rolled requires parameter values indexed by the loop
variable, a runtime capability rather than a build one; it is adjacent to what the `ShotLoop`
randomization axis in `.notebooks/design/samplex_backendv3_design.md` already does, so it may come free
later.

**`WhileLoopOp` and dynamic trip counts are rejected**, but the reason is sharper than "while loops do
not work": a twirl fully contained within one iteration, reusing the same parameters every pass, is a
*correct* circuit. What is impossible is **independent** randomization per iteration, since that needs
unbounded parameter generation. Stated that way the rule also catches `BreakLoop` and `ContinueLoop`,
which make an otherwise-fine `for` loop's trip count dynamic.

Branch bodies are `Block`s exactly like box bodies, so the build pass's box recursion generalizes to
branches with no new machinery beyond the barrier rule.

## Layout

Data and objects at the crate root, transforms under `passes/`, matching `crates/transpiler`. Each file
is named for the IR it defines, so the vocabulary of a stage is in one place.

```
src/
  lib.rs
  annotated_circuit.rs  IR1 vocabulary: Twirl, ChangeBasis, InjectLocalClifford, InjectNoise, Tag,
                        BoxAnnotation, AnnotationKind, the *Spec types, string parsers, and the
                        shared enums the later IRs borrow (SynthesizerType, DistributionType,
                        ChangeBasisMode, Dressing, Placement, InjectionSite)
  emission_circuit.rs   IR2 vocabulary: the Emit instruction (EmitSpec, EmitSource) and the Collect
                        annotation (CollectSpec) — one file, two halves of one dialect
  distributions.rs      DistributionTable, DistKey, DistEntry — IR2's side object
  virtual_flow_graph.rs IR3: the graph, Node, Edge, NodeKind
  virtual_type.rs       VirtualType — an object, not part of any one IR
  partition.rs          Partition
  error.rs              LowerError
  passes/
    build.rs                    IR1 -> IR2
    absorb_emissions.rs         IR2 -> IR2
    merge_collectors.rs         IR2 -> IR2
    prune_collectors.rs         IR2 -> IR2
    lower.rs                    IR2 -> IR3
    merge_parallel_nodes.rs     IR3 -> IR3
    prune.rs                    IR3 -> IR3
    set_virtual_types.rs        IR3 -> IR3
```

## Invariants to hold deliberately

- **Never build a `Partition` from a `HashSet`.** `Partition::from_elements` preserves iterator order.
  Feeding it a `HashSet`, or iterating a `HashSet` to decide node creation order, makes output
  nondeterministic run to run: qubit order within partitions varies, `NodeIndex` assignment varies, and
  parameter indices become irreproducible — so a seed does not pin a result. Sort before constructing
  partitions, and iterate sorted `Vec`s. (This was a real bug in the original builder, confirmed by
  three consecutive runs of one binary producing different output.)
- Golden-file comparison of rendered graph output is therefore invalid. Assert on counts and structure.
- **Write the pure rules as pure functions.** Exactly three things need the GIL — reading input
  annotations, constructing the `Collect` annotation, and constructing `Emit`. The `DAGCircuit` does
  not: `new`, `with_capacity`, `add_qreg`, `apply_operation_back`, `remove_op_node` and
  `substitute_node_with_dag` all take no `Python` token. Annotations are expected to become Rust objects
  eventually (blocked upstream: `ControlFlow::Box { annotations: Vec<Py<PyAny>> }` at
  `crates/circuit/src/operations.rs:511-514`), so the GIL requirement is temporary and should not be
  baked in. Keep decision logic — easy/hard classification, merge compatibility, the absorption guard,
  the propagation walk — as functions over plain data rather than inline in GIL-holding code. Those stay
  unit-testable in Rust, and when annotations go native the passes become Rust-testable by moving a
  boundary rather than by a rewrite. Note that promoting `Emit` to a native `PackedOperation` variant
  alone would not achieve that; the collect-box marker has to go native too.

## IR3 shape

IR2 made three simplifications available, and all three are now in place. What each turned out to mean:

1. **Emission kinds unify.** `NodeKind::Emission` is one kind for twirls, basis changes and noise
   injections, carrying the `DistEntry` it draws from. The entry's discriminant *is* the source tag, so
   nothing separate is stored, and `is_source` is two arms rather than four. It also carries the
   `VirtualType`, read off the IR2 emission rather than re-derived from the distribution — deriving it
   twice is how the two could disagree.

   `Reset` stayed its own kind rather than joining them, despite being a source. It is a real
   instruction with no distribution and no direction, and unlike an emission it survives into the
   template. Folding it in would have meant an optional distribution on every emission, moving the
   fan-out from the enum into the payload.

2. **Absorbed gates stop being nodes.** `Collect` owns its absorbed gate sequence and `PropagateMode` is
   gone, so every `Propagate` node is a genuine conjugation.

3. **Direction moves off edges — onto nodes, not derived by walking back.** The original reasoning
   ("fixed when an emission is created, therefore derivable from the originating emission") is right
   per *path* and wrong per *node*: a node can sit on two paths at once. A gate can be crossed by one
   flow going left and another going right, and then there is no single originating emission to read.

   So a `Propagate` node is created **per conjugation, not per gate occurrence**: keyed by
   `(occurrence, direction, virtual_type)`. Both extra key components change what the node computes —
   conjugating a Pauli by CX leftward and rightward are different operations, as are conjugating a Pauli
   and a local C1 by the same gate — so sharing across them would fuse operations that cannot be
   evaluated as one. Direction then lives unambiguously on the node (`NodeKind::direction`), and `Edge`
   carries only the virtual type.

   The both-directions case is reachable, not hypothetical: an outer left-dressed box's right factor
   walks rightward through an inner right-dressed box's hard content, while the inner box's own left
   factor walks leftward through the same gates. Before nodes were keyed by direction those became one
   node with incoming edges of both handednesses.

   `merge_parallel_nodes` therefore includes direction in its merge key — two conjugations of the same
   gate in opposite directions must not fuse into one wider node.

**Still redundant, deliberately left alone:** now that a `Propagate` node is unique per virtual type,
`Edge.virtual_type` and the `set_virtual_types` pass that fills it in are derivable from the nodes
alone. Removing them would delete a pass from the inventory, which is a bigger decision than a
simplification — recorded here rather than acted on.

## Emission width and subsystem partitioning

**Should `Emit.num_qubits` match the distribution's `num_subsystems` rather than the full box width?**

Currently (`build.rs:389`) every emission covers all box qubits as singleton subsystems:

```rust
let partition = Partition::from_elements(qubits.iter().copied());

The alternative: shrink the emission's qargs to only the qubits the distribution samples.

Why it is tempting. A narrower emission produces a smaller graph — walk_emission's frontier only
tracks its own qubits, so chain() skips gates that do not overlap, halving Propagate nodes when a
distribution acts on half the box. It directly encodes distribution semantics (UniformLocalC1 on a pair
is better represented by a 2-qubit Emit than a 4-qubit one), and it is the natural unit for
gate-dependent partitioning where different 2Q gate types induce separate emissions.

Why it breaks: propagation through entangling gates requires joint qubit tracking. Consider a 4-qubit
box with hard content cx(0,1); cx(2,3). A CX propagation for a Pauli is a joint 2-qubit → 2-qubit
lookup: P_in(q0)⊗P_in(q1) → P_out(q0)⊗P_out(q1). Under one 4-qubit emission, one walk crosses the CX
with both qubits in its frontier simultaneously — the Propagate node gets both inputs from a single
source. Under two 1-qubit emissions, two independent walks each reach the same Propagate node (shared via
GateKey), but each walk's frontier tracks only its own qubit. The evaluator must then reconstruct that
the two inputs come from the same draw (same DistKey) and wait for both before evaluating — a
scheduling constraint the current graph topology does not encode.

Downstream consequences of narrow emissions:

- Collector width is unchanged. Propagation through entangling gates spreads correlation to all
coupled qubits, so the collector cannot be narrower than the union of all reachable qubits. Narrower
emissions with full-width collectors creates an IR asymmetry.
- CollectStep::Emission(id) has no per-qubit indexing. If the emission covers fewer qubits than the
collector, a mapping from emission subsystems to collector subsystems is needed — which
CollectSpec.items does not carry.
- merge_collectors qubit release breaks. The merge pass uses emission qubits to release frontiers.
Narrower emissions release fewer, potentially allowing later collectors to commute past emissions they
depend on through entangling propagation.

The distinction that matters is subsystem grouping, not qubit coverage. Changing
Partition::from_elements (singletons) to Partition::with_parts(k, groups) is safe: it tells the
evaluator which qubits are jointly sampled without removing any qubit from the propagation walk. For
UniformLocalC1 this means [[0,1], [2,3]] rather than [[0], [1], [2], [3]]; for UniformPauli
singletons remain correct.

Truly narrow emissions require explicit routing infrastructure (the backendv3 Slice/Combine nodes) that
fans virtual state out of a narrow emission and into a wide collector, handling the correlation introduced
by entangling gates on the propagation path. Without that infrastructure, narrow emissions are unsound for
any box whose hard content includes entangling gates.

## Open questions

- **Cross-scope collector merging** (fusing an inner collector with an outer one). Cross-scope
  *absorption* now covers the primary motivating case — an outer emission descends into a nested box and
  is absorbed by the inner collector. The remaining case is merging the inner collector's *structure*
  (absorbed gates, items) into an outer collector, which still requires segment structure for recording
  propagation through relocated gates. Low priority given that absorption handles the dataflow correctly.
- Whether `VirtualFlowGraph` should be renamed. `SamplingGraph` matches how it is described, and the
  file has already been renamed once (`virtual_dependency_graph.rs` → `virtual_flow_graph.rs`), but it
  churns the three existing IR3 passes and the design docs.
