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
- In example code, NEVER explicitly import names already exported by `using NumericalEarth`

## Docstrings

- ALWAYS use `jldoctest` blocks, NEVER plain `julia` blocks
- See `.claude/rules/docstring-rules.md` for full details
