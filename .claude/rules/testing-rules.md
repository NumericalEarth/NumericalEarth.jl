---
paths:
  - test/**/*.jl
---

# Testing Rules

## Writing Tests

- Every `test/*.jl` file is discovered automatically as a test. Helper files must be removed from
  the test suite in `test/runtests.jl` (as `runtests_setup` and `synthetic_datasets` are)
- Tests run offline. Use the analytic datasets in `test/synthetic_datasets.jl`; a test that needs
  real data goes in `remote_data_tests` in `test/runtests.jl`
- Test on both CPU and GPU when possible
- Name test files descriptively (snake_case)
- Include both unit tests and integration tests
- Test numerical accuracy where analytical solutions exist

## Debugging

- GPU "dynamic invocation error": run on CPU first to isolate GPU-specific issues
- Julia version issues: delete Manifest.toml, then `Pkg.instantiate()`
- Ensure doctests pass; use Aqua.jl for package quality checks

## Quality

- Ensure all explicit imports are correct (tests check this automatically)
- Always add tests for new functionality
- Avoid `@allowscalar` in new tests; it hides the scalar indexing that fails on GPUs. Transfer
  data to the CPU with `Array(interior(field))` first
- Use minimal grid sizes to reduce CI time
- Avoid hardcoded grid indices — use `size(grid, d)` instead of literal numbers
