# NumericalEarth.jl

Couples Earth system components (Oceananigans oceans, ClimaSeaIce sea ice, Breeze or
SpeedyWeather atmospheres, land) to each other or to prescribed datasets such as JRA55, ERA5,
and ECCO. It runs on CPUs and GPUs through Oceananigans and KernelAbstractions, so code that
passes on a CPU can still fail on a GPU; `.claude/rules/kernel-rules.md` covers why.

## Commands

```sh
# Run one test file on CPU (test names are file names under test/, without .jl)
CUDA_VISIBLE_DEVICES=-1 julia --project -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=`test_breeze_coupling`)'

# Explicit imports and Aqua checks (run after any change to src/ or ext/)
CUDA_VISIBLE_DEVICES=-1 julia --project -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=`test_quality_assurance`)'

# Trailing whitespace and blank lines at end of file
git diff --check origin/main

# Syntax check without loading any packages
julia -e 'Meta.parseall(read("file.jl", String))'
```

## Environments

- The root environment treats Breeze, Makie, and other extension triggers as weak dependencies,
  so `julia --project=.` cannot `using Breeze`. Run Breeze-coupled code with `--project=test` or
  `--project=docs`.
- Manifests go stale when compat bounds or `[sources]` pins change. Re-resolve before diagnosing
  load errors or "undeclared at import time" warnings.
- Tests run offline by default: `NUMERICALEARTH_DATA_DIRECTORY` points at a fresh temporary
  directory, and any file that lands there is an accidental download. New tests use the analytic
  stand-ins in `test/synthetic_datasets.jl`. Tests that need real data go in `remote_data_tests` in
  `test/runtests.jl` and run with `NUMERICALEARTH_TEST_REMOTE_DATA=true`; `*_downloading` tests
  run only in the DataDownload workflow.
- Slurm and multi-GPU runs: read `.agents/cluster.md` first.

## Before you change these, ask

- **`[deps]`, `[weakdeps]`, and `[sources]` in `Project.toml`**. They change load time, CI, and
  every downstream environment. Touch `[compat]` only when asked.
- **Expected values and tolerances in tests**. A numerical test that starts failing is evidence of
  a behavior change; find the cause instead of updating the number.
- **Exported names and keyword arguments of public constructors**. User scripts and the examples
  depend on them.

## Verifying your work

- Read the current definition of anything you call (`@which`, `methods`, or the source),
  including NumericalEarth's, Oceananigans', and Breeze's own APIs. They change quickly and
  remembered signatures go stale. Search all of an installed package, including `ext/`, before
  concluding a feature does not exist.
- A test that fails on your branch is yours until you reproduce the same failure on `main`.
- Report results by quoting the test summary line. An exit code alone is not a pass.
- If a fix makes a failing test run but you cannot explain why it was failing, the fix is probably
  wrong. Revisit the change that broke it.
- GPU "dynamic invocation error": rerun on CPU. If it passes there, the cause is almost always a
  type instability that the CPU tolerates.
- Before presenting a change, review your own `git diff` against the checklist at the end of
  `.claude/rules/restraint-rules.md`, cut what it catches, and report the result in one line.

## Design

- **Never unpack a property immediately after a constructor** (`foo(args...).bar`). It means the
  constructor returns the wrong type for the call site. Add a constructor that returns what is
  needed (for example `atmosphere_model(grid; …)` alongside `atmosphere_simulation(grid; …)`).
- **The example drives the API.** When an example hand-rolls infrastructure (region padding,
  relaxation masks, terrain preparation, initialization, output slicing), move it into the library
  as a constructor keyword, dataset hook, or exported utility instead of polishing it in place.
- **Constructors own their domain.** Derive what is derivable: regions from grids plus the
  dataset's `default_horizontal_padding`, anchors from the dataset, physics defaults internally.
  The user supplies intent (`grid`, `dataset`, `dates`), not plumbing.
- **Dataset objects carry product identity only** (cadence, levels, native grid), never variable
  names, regions, or dates. Dataset-specific behavior enters through `DataWrangling` hooks
  dispatched on the dataset type, so downstream packages can add datasets without touching
  NumericalEarth.
- **Date windows are `(start_date, end_date)` tuples**, expanded to the dataset's native cadence by
  `DataWrangling.expand_dates`. Don't add `start_date`/`end_date` keyword arguments.
- **Put key identity in `Base.summary`** (for example a regional atmosphere's domain bounds) so
  composite models' displays inherit it; never print by hand what `show`/`summary` already shows.
- **Extension-implemented API**: declare a documented, exported stub in `src` (`function foo end`)
  and define the method in the extension as `NumericalEarth.Module.foo(...) = ...`.
- **Materialization pattern**: a user-facing constructor builds a skeleton struct with placeholder
  type parameters (such as `Nothing`); `materialize_*` builds the fully typed version once the grid
  and model are known.
- Structs are concretely typed; never use `Any` as a type parameter or field type. For mutable
  state inside an immutable struct, use a `mutable struct` as the field type.
- When something would be better in Oceananigans, add a detailed TODO note rather than a local
  workaround.

## Conventions that are not visible from the code

- Source code uses explicit imports, checked by `test_quality_assurance`. Extend functions with
  `Module.function_name(...) = ...`, not `import`. Exports go at the top of module files. Import
  Oceananigans/NumericalEarth names first, then external packages; internal imports use absolute
  paths. Examples and docs use `using Oceananigans` and `using NumericalEarth`.
- Docstrings use `$(TYPEDSIGNATURES)` and `jldoctest` examples; docs pages use `@example` blocks.
  Details are in `.claude/rules/docstring-rules.md` and `.claude/rules/docs-rules.md`.
- Variable names are full English (`latitude`, not `lat`) or Unicode math from
  `docs/src/appendix/notation.md`, never a mix in one identifier. Add new symbols to that table.
  A leading `_` is reserved for `@kernel` functions. Details are in `.claude/rules/style-rules.md`.
- American English in code, comments, docstrings, and docs: `center`, `meter`, `neighbor`,
  `behavior`, `-ize`. Proper nouns keep their spelling (European Centre for Medium-Range Weather
  Forecasts).
- Keyword arguments: no spaces inline, `f(x=1)`; single spaces when split over lines,
  `f(a = 1, b = 2)`.
- Never extend `getproperty` to make an undefined-property error go away; fix the caller.
- A "type is not callable" error usually means a local variable shadows a function name.
- Keep a PR to one concern and base it on `main`; never merge another feature branch into it.

## Where to look

Rules in `.claude/rules/` load automatically in Claude Code when you edit matching files. Other
agents should read the one that matches the task:

| Task | Read |
|------|------|
| Writing or editing kernels, operators, or anything in `src/` or `ext/` | `.claude/rules/kernel-rules.md` |
| Any change to `src/`, `test/`, or `examples/` | `.claude/rules/restraint-rules.md` |
| Naming, notation, comments | `.claude/rules/style-rules.md` |
| Docstrings | `.claude/rules/docstring-rules.md` |
| Tests | `.claude/rules/testing-rules.md` |
| Docs pages | `.claude/rules/docs-rules.md` |
| Examples | `.claude/rules/examples-rules.md` |
| Slurm or multi-GPU runs | `.agents/cluster.md` |
