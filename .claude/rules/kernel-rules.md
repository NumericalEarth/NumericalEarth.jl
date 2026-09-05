---
paths:
  - src/**/*.jl
---

# Kernel Function Rules

GPU-compatible kernel functions are critical for NumericalEarth performance.

## Requirements

- Use KernelAbstractions.jl syntax: `@kernel`, `@index`
- Keep kernels **type-stable** and **allocation-free**
- Use `ifelse` instead of short-circuiting `if`/`else` statements
- No error messages inside kernels
- Models **never** go inside kernels
- Mark functions called inside kernels with `@inline`
- **Never use loops outside kernels**: Always replace `for` loops that iterate over grid points
  with kernels launched via `launch!`. This ensures code works on both CPU and GPU.

## Type Stability

- All structs must be concretely typed
- Type instability in kernel functions ruins GPU performance
- Julia compiler can infer types; use type annotations primarily for **multiple dispatch**, not documentation

## Numeric Types

- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels
- Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals
- Use `on_architecture` for data transfers — never manual `Array()` / `CuArray()` calls

## Memory Efficiency

- Favor inline computations over allocating temporary memory
- Minimize memory allocation overall
- Design solutions that work within the existing framework

## Staggered Grid & Indexing

- Velocities live at cell faces, tracers at cell centers (Arakawa C-grid)
- Take care of staggered grid location when writing operators or designing diagnostics
- **Always use 3D indexing** for fields (`field[i, j, k]`); 2D indexing works by coincidence
  but is unsupported and may break

## Closure Captures

- **Never reassign a variable captured by a closure that reaches a GPU kernel** (masks,
  forcings, boundary conditions). Reassignment turns the capture into a `Core.Box`, the
  closure stops being `isbits`, and the kernel launch fails — often only at run time on GPU.
- Compute with single-assignment names instead: `λ₁ˡ, λ₂ˡ = x_domain(grid)` then
  `λ₁ = all_reduce(min, λ₁ˡ, arch)` — never `λ₁ = ...` followed by `λ₁ = f(λ₁)`.
- When in doubt, check `isbits(closure)` before launching.

## Dispatch Over Branching

- No value-level guards (`haskey`, `isnothing` chains) inside small GPU helper functions;
  make the decision once outside the kernel — fetch with `get(container, key, nothing)` —
  and provide a `::Nothing` method for the missing case.
- A `MethodError` on an unhandled combination is the intended "not implemented" signal;
  don't paper over it with runtime branches.
