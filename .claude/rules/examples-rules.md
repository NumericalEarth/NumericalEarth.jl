---
paths:
  - examples/**/*.jl
---

# Examples Rules

## Writing Examples

- Explain at the top of the file what a simulation is doing
- Let code "speak for itself" - keep explanations concise (Literate style)
- Use visualization interspersed with model setup when needed to illustrate
  complex grids, initial conditions, or other model properties
- New examples should add value while remaining simple: judiciously introduce
  new features and do creative, surprising things with simulations
- Don't "over import". Use names exported by `using NumericalEarth`. If needed
  names aren't exported, consider exporting them from `NumericalEarth.jl`

## Literate.jl Comment Conventions

Examples in `examples/` are processed by Literate.jl:
- Single `#` comments become markdown blocks in generated documentation
- Double `##` comments remain as code comments within code blocks
- Use `##` for inline code comments that should stay with the code
- Use single `#` only for narrative text that should render as markdown

## Simulation Idioms

Examples are scripts built on Oceananigans machinery — use that machinery, never hand-rolled
substitutes:

- **Output goes through output writers.** Attach a `JLD2Writer` (or `NetCDFWriter`) with a
  `schedule` to `simulation.output_writers`, and read results back with `FieldTimeSeries` for
  post-processing and animation. Never accumulate arrays, slices, or frames in memory from a
  callback.
- **Save 2-D slices, not 3-D fields.** Full 3-D output overflows the disk. Pass `indices = (:, :, k)`
  (or `(:, j, :)`, `(i, :, :)`) to the writer: it computes and slices each output — plain field,
  `Reduction`, or `AbstractOperation` — at save time. Pick a level with
  `k = searchsortedfirst(znodes(grid, Center()), height)`.
- **Diagnostics are AbstractOperations, saved directly.** Hand `VirtualPotentialTemperature(model)`,
  `sqrt(u^2 + v^2)`, `ρq / ρ`, etc. straight to the writer — the operation is recomputed each save,
  no manual `compute!`. Prefer the model's exported diagnostic accessors over hand-rolled arithmetic.
- **Operations auto-interpolate** operands at differing staggered locations; `@at (LX, LY, LZ) op`
  is needed only to *force* a location. Reductions (`maximum`, `minimum`) act directly on `Field`s
  and operations.
- **Never `Array(interior(...))` for plotting, never broadcast fields.** Pass a 2-D `Field`,
  `view(field, …)`, indexed `FieldTimeSeries`, or operation straight to `heatmap!`/`surface!`/
  `lines!` — the Oceananigans Makie extension computes it, NaN-masks immersed cells, drops singleton
  dimensions, reads the field's own coordinates, and moves GPU→CPU. A pointwise anomaly is a lazy
  field op (`fts[n] - fts[1]`), not broadcast arrays. Terrain-following fields plot directly; Makie
  maps the deformed grid to physical coordinates.
- **Callbacks are for lightweight actions only**: progress logging, updating forcing fields. A
  `NestedModel` does not forward `getproperty` — reach child state through `sim.model.child`.
- **Let objects display themselves.** Literate/Documenter renders each block's final value, so
  end blocks with `grid`, `model`, or `fig` instead of `@info`-printing the same information.
  No "wrote a file" notices. Suppress noisy outputs (arrays, frame vectors) with `nothing #hide`;
  embed movies with `# ![](movie.mp4)` markdown.
- **No `const` in example scripts** — it buys nothing at script scope and prevents re-running
  blocks interactively.
