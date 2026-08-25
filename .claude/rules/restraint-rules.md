---
paths:
  - src/**/*.jl
  - test/**/*.jl
  - validation/**/*.jl
  - examples/**/*.jl
---

# Restraint Rules

Machine-written code fails in a characteristic way: it is **correct and unreadable**. Every branch is handled,
every failure anticipated, every invariant checked at three layers — and no human can hold the result in their
head. A reviewer facing 700 lines cannot tell which 40 are the feature.

These rules exist to keep the diff small, the mechanisms singular, and the prose scarce. They are **always in
effect**, and they outrank any instinct toward completeness.

> **The test behind every rule below:** could a reviewer who knows the physics read this hunk once, in order,
> and say yes?

## Rule 1 — The feature is the diff; everything else is scope creep

Before opening a PR, ask what the smallest diff is that delivers the stated change. That diff is the PR.
Anything that is not the feature — validation, fallbacks, gap-fillers, helper APIs — is a **separate concern**
and belongs in a separate PR, or nowhere.

- A generalization that **deletes** an abstraction is the best possible change. A change that ends net-negative
  needs no justification; celebrate it and stop there.
- If the feature commit is net-zero and the PR is +700, the PR is 95 % scaffolding. Drop the scaffolding.
- Adding lines to `src/` is a cost paid forever. The bar is not "is this useful?" but "is the codebase better
  off carrying this than not?"

❌ Feature lands in commit 1 (+142/−143); commits 4 and 7 add +680 lines of validation nobody asked for.
✅ Feature lands in commit 1. PR ends. Validation, if genuinely needed, is proposed separately on its merits.

## Rule 2 — One invariant, one mechanism

An invariant is enforced in exactly one place. Enforcing it at setup *and* in the kernel *and* through a
fallback *and* in a gap-filler does not make it four times as safe — it makes it impossible to tell which
mechanism is load-bearing, and guarantees the four drift apart.

Before adding a check, find where the invariant is already enforced. If a runtime floor already exists,
setup-time validation is redundant; if setup-time validation is added, the runtime floor should be deleted.

❌ `evaluable_roughness_length` checked in `validate_flux_formulation`, again in `local_roughness_length`,
   again in `guard_local_roughness_length`, again in `fill_aerodynamic_roughness_gaps!` — while
   `displaced_profile_height`'s `max(Δh - d, 2ℓ)` already floored it before the PR.
✅ One check, at the layer that owns the contract. State in one line where that is.

## Rule 3 — No guards for states the code cannot reach

Never add machinery to defend against a state that is unreachable through the public API. "A user might mutate
this field after construction" and "a downstream package might pass a wrong type" are hypotheticals, not
requirements. Code written against hypotheticals is permanent; the hypothetical usually never arrives.

A **loud failure is better than a silent fallback.** A `NaN` that reaches the coupled state is visible in the
first output; a roughness silently floored at `1e-5` produces plausible wrong fluxes forever.

- If a guard's own docstring has to say *"this is a guard, not a physical parameter"* — delete the guard.
- Never grow a struct, a keyword argument, or a `show` line to hold a value that is not physics.

❌ `minimum_roughness_length` — a new public keyword, struct field, `FT` conversion and `show` row, existing
   only so a field mutated after validation cannot produce `NaN`.
✅ Validate at construction, or let it fail loudly. No new API.

## Rule 4 — Don't invent an API to solve a problem the same PR created

If a change makes previously valid data invalid, and the PR then ships a new exported helper to repair that
data, the PR has manufactured its own problem. Fix the producer, reuse what exists, or loosen the new
constraint.

❌ New validation rejects `urban_roughness`'s `NaN` cells → same PR adds `fill_aerodynamic_roughness_gaps!`,
   two kernels and a 20-line docstring, to undo it — while noting `inpaint_mask!` already exists.
✅ `urban_roughness` does not emit values its only consumer rejects, or the consumer tolerates them.

## Rule 5 — Dispatch is for polymorphism, not for `if`

Do not write families of one-line methods whose only purpose is to select a branch. Dispatch is the right tool
when the *behavior* differs by type; it is the wrong tool when a single `ifelse` says it plainly. A reader must
not have to collect four scattered methods to learn that a value is clamped when it comes from a `Field`.

❌ `clearance_roughness_length` (3 methods) + `guard_local_roughness_length` (3 methods) +
   `guard_local_zero_plane_displacement` (3 methods) — nine one-liners implementing one clamp.
✅ One `@inline` with an `ifelse`, or the clamp written where the value is used.

## Rule 6 — No table-driven generalization of a small fixed set

Four things are not a collection. Do not build a registry of `(name, slot, predicate, message)` tuples and loop
over it to save writing four explicit checks — the table costs more to read, hides the control flow, and its
promised "adding one is a single edit" is usually false the moment anything unpacks it positionally.

Reach for a table at roughly a dozen homogeneous entries, when they genuinely vary at runtime — not before.

❌ `flux_formulation_slots` returning 4 tuples of name/value/predicate/prose, walked by two functions, then
   destructured positionally as `ℓuᵛ, ℓθᵛ, ℓqᵛ, dᵛ = values` right afterward.
✅ Four explicit one-line checks, in order, readable top to bottom.

## Rule 7 — Comments describe this code, now — nothing else

Reinforces the comment rules in `style-rules.md`, which this codebase violates most often. A comment must not:

- explain what **another** function does, or what a **caller** or **kernel elsewhere** would do;
- justify a design decision, compare it to a rejected alternative, or narrate history;
- describe code that **is not there** ("the ocean kernel would fail with…", "a bare `MethodError` on CPU…");
- teach the language (`==` vs `===`, how `@inbounds` works, why `Adapt` needs field order);
- carry a multi-line design memo or `TODO` about a different module's construction order.

If the reasoning genuinely cannot be dropped, it belongs in the PR description — which reviewers read once and
which does not age in the source tree.

❌ Five lines above `Adapt.@adapt_structure SimilarityTheoryFluxes` explaining that a positional rebuild
   elsewhere would mis-wire the struct and that a test pins the field order.
❌ `# Grids are compared with `==` … rather than `===`: two references to one grid are not guaranteed to be egal`
✅ `# ifelse, not ?:, so the kernel stays branch-free on GPU`

## Rule 8 — Docstrings describe, they do not argue

A docstring says what the function does, its arguments, and its units. It does not defend the function's
existence, quantify what would go wrong without it, or send the reader to three other functions to understand
it. A cluster of `@ref`s pointing at siblings means the abstraction is wrong, not that the docs are thorough.

- No "this is a backstop, not a supported regime" framing. No bulleted rationale lists.
- No tuning advice for a parameter the same docstring calls non-physical.
- Keep the argument list; drop the essay.

❌ 18 lines documenting `minimum_roughness_length`, of which 2 describe it and 16 argue about when it binds.
✅ `` `minimum_roughness_length`: Floor [m] on the roughness length used in the similarity profile. Default: 1e-5. ``

## Rule 9 — Error messages are one sentence

Name the offending object, say what was required, stop. An `ArgumentError` is not a tutorial: it should not
explain the physics, list downstream consequences, quote numbers, or recommend a helper function. Six errors of
six lines each is fifty lines of string literal in `src/`.

❌ `"$name has $(count(...)) cells that are not $requirement (minimum $(minimum(values))). A bad cell either
   propagates NaN through u★, θ★ and q★ into the coupled state or silently zeroes every turbulent flux there.
   Fill the gaps first, e.g. with fill_aerodynamic_roughness_gaps!."`
✅ `"momentum_roughness_length must be finite and positive, found $(minimum(values))"`

## Rule 10 — Test behavior, not scaffolding

Tests pin the physics and the public contract. A test that asserts an internal helper's return type, a struct's
field order, or the exact value a fallback produces makes the scaffolding **load-bearing**: it can no longer be
deleted without "breaking tests", which is precisely how machine-written scaffolding becomes permanent.

- Never write a test whose only purpose is to lock in a guard added by the same PR.
- No tautological assertions (`@test !isnothing(constructor(...))`) — a constructor returning `nothing` is a bug
  in itself, not something to assert.
- One clear analytical check (`u★ = ϰ ΔU / log((h − d) / ℓᵐ)`, cell by cell) beats twenty defensive ones.

## Rule 11 — Notation is from `notation.md`, or it is English

No inventing decorations. A superscript must mean what it means in the physics, not encode a code-level
distinction like "host-side copy" or "clearance contribution". If a value needs that kind of qualifier, it
needs a verbose English name.

Reread `style-rules.md` Rules 1–3 before naming anything: identifiers are **fully math or fully English**.

❌ `ℓᵛ`, `zᵛ`, `dᵛ`, `ℓᵐᵃˣᵛ`, `ℓuᶜ` (invented `ᵛ` = "values array", `ᶜ` = "clearance")
❌ `ℓmin` (math + truncated English in one identifier), `to_FT` (ad-hoc local closure)
✅ `roughness_length_values`, `ℓᵐᵃˣ` (in `notation.md`), `minimum_roughness_length`

## Rule 12 — Use constructors as designed

Already pitfall 6 in `AGENTS.md`; it is broken most often in new doctests and examples. Never reach into a
constructor's result to prove it worked, and never write a doctest whose assertion is that `set!` assigned a
number.

❌ ```jldoctest``` that builds a full coupled interface to check
   `interface.flux_formulation.zero_plane_displacement[2, 1, 1] == 4.0`
✅ A doctest that shows the object, via `show`/`summary`, or nothing at all.

## Reviewer's checklist

Run this over your own diff before asking for review. Any **yes** means cut before pushing.

1. Is the diff more than ~2× the size of the change actually described in the title?
2. Does any invariant get enforced in more than one place?
3. Does any new code defend against a state unreachable through the public API?
4. Does the PR add an API to repair damage the PR itself caused?
5. Are there ≥3 one-line methods that together implement a single `ifelse`?
6. Is there a table, registry, or predicate list covering fewer than ~12 fixed entries?
7. Does any comment mention a function, module, kernel, or test other than the one it sits in?
8. Does any docstring argue for the function's existence or carry `@ref`s to siblings to be understood?
9. Is any error message longer than one sentence?
10. Does any test pin an internal helper, a field order, or a fallback value?
11. Does any identifier carry an invented sub/superscript, or mix math with English?
12. Is there a new public keyword argument that the docs describe as non-physical?
