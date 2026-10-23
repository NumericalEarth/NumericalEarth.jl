---
paths:
  - docs/**/*
---

# Documentation Rules

## Building Docs

```sh
julia --project=docs/ docs/make.jl
```

## Fast Local Builds

Long-running examples (`build_always = false`) are skipped unless
`NUMERICAL_EARTH_BUILD_ALL_EXAMPLES=true`. For faster local testing, temporarily modify `docs/make.jl`:
1. Comment out entries in `examples` and `developer_examples`
2. Add `warnonly = [:cross_references, :example_block, :linkcheck]`
3. Optional: `doctest = false`, `linkcheck = false`, `draft = true`

**Remember to revert these changes before committing!**

## Viewing Docs

```julia
using LiveServer
serve(dir="docs/build")
```

## Style

- Use Documenter.jl syntax for cross-references
- Add paper references via bibtex in `NumericalEarth.bib` with corresponding citations
- Make use of cross-references with equations
- In example code, rely on `using NumericalEarth`; explicitly importing an exported name hides
  what users actually need to type
- Code on docs pages goes in `@example <label>` blocks, which Documenter runs and renders with
  their output. Plain `julia` fences are never executed and go stale. Reuse one label across a
  page to share state between blocks

Docstring conventions are in `.claude/rules/docstring-rules.md`.
