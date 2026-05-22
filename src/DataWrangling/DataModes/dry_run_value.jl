"""
    DryRunValue()

Sentinel returned when a statement under [`build_dataset_manifest`](@ref) tracing either errors or
stands in for a value that real data would have produced. The per-statement `try`/`catch` wrappers
rebind any failed assignment to a `DryRunValue`, so the script continues running and downstream
`download_dataset` calls still register their metadata.

To maximise script reach without touching `src` outside this module, `DryRunValue` absorbs almost
every common operation — call, property access, indexing, iteration, broadcasting, arithmetic, and
comparison all return another `DryRunValue`. Operations that fall outside this set still throw and
are caught by the surrounding per-statement wrapper.
"""
struct DryRunValue end

Base.show(io::IO, ::DryRunValue) = print(io, "DryRunValue()")
Base.print(io::IO, ::DryRunValue) = print(io, "DryRunValue()")
Base.string(::DryRunValue) = "DryRunValue()"

Base.getproperty(::DryRunValue, ::Symbol) = DryRunValue()
Base.setproperty!(::DryRunValue, ::Symbol, _) = DryRunValue()
Base.propertynames(::DryRunValue, ::Bool = false) = ()
Base.hasproperty(::DryRunValue, ::Symbol) = true

(::DryRunValue)(args...; kwargs...) = DryRunValue()

Base.length(::DryRunValue) = 0
Base.size(::DryRunValue) = ()
Base.size(::DryRunValue, ::Int) = 0
Base.axes(::DryRunValue) = ()
Base.axes(::DryRunValue, ::Int) = Base.OneTo(0)
Base.eltype(::Type{DryRunValue}) = DryRunValue
Base.ndims(::DryRunValue) = 0
Base.ndims(::Type{DryRunValue}) = 0
Base.isempty(::DryRunValue) = true
Base.firstindex(::DryRunValue) = 1
Base.lastindex(::DryRunValue) = 0
Base.keys(::DryRunValue) = ()
Base.values(::DryRunValue) = ()
Base.pairs(::DryRunValue) = ()

Base.iterate(::DryRunValue, state = nothing) = nothing
Base.IteratorSize(::Type{DryRunValue}) = Base.HasShape{0}()
Base.IteratorEltype(::Type{DryRunValue}) = Base.HasEltype()

Base.broadcastable(::DryRunValue) = Ref(DryRunValue())
Base.materialize(::DryRunValue) = DryRunValue()

Base.getindex(::DryRunValue, args...) = DryRunValue()
Base.setindex!(::DryRunValue, args...) = DryRunValue()
Base.view(::DryRunValue, args...) = DryRunValue()

Base.adjoint(::DryRunValue) = DryRunValue()
Base.transpose(::DryRunValue) = DryRunValue()
Base.collect(::DryRunValue) = DryRunValue()
Base.copy(::DryRunValue) = DryRunValue()
Base.deepcopy(::DryRunValue) = DryRunValue()
Base.similar(::DryRunValue, args...) = DryRunValue()

Base.convert(::Type{DryRunValue}, ::DryRunValue) = DryRunValue()
Base.promote_rule(::Type{DryRunValue}, ::Type) = DryRunValue
Base.promote_rule(::Type, ::Type{DryRunValue}) = DryRunValue

Base.hash(::DryRunValue, h::UInt) = hash(DryRunValue, h)
Base.:(==)(::DryRunValue, ::DryRunValue) = true
Base.isequal(::DryRunValue, ::DryRunValue) = true

for op in (:+, :-, :*, :/, :\, :^, :%, :÷, :&, :|, :⊻, :>>, :<<, :>>>,
           :<, :>, :<=, :>=, :min, :max)
    @eval Base.$op(::DryRunValue, ::Any) = DryRunValue()
    @eval Base.$op(::Any, ::DryRunValue) = DryRunValue()
    @eval Base.$op(::DryRunValue, ::DryRunValue) = DryRunValue()
end

for op in (:-, :+, :abs, :abs2, :sqrt, :cbrt, :exp, :exp2, :exp10, :expm1,
           :log, :log2, :log10, :log1p, :sin, :cos, :tan, :asin, :acos, :atan,
           :sinh, :cosh, :tanh, :floor, :ceil, :round, :real, :imag, :conj,
           :inv, :sign, :signbit, :one, :zero, :oneunit, :isnan, :isinf, :isfinite,
           :iszero, :isone, :isreal, :isinteger)
    @eval Base.$op(::DryRunValue) = DryRunValue()
end

Base.:(:)(::DryRunValue, ::Any) = DryRunValue()
Base.:(:)(::Any, ::DryRunValue) = DryRunValue()
Base.:(:)(::DryRunValue, ::DryRunValue) = DryRunValue()
Base.:(:)(::DryRunValue, ::Any, ::Any) = DryRunValue()
Base.:(:)(::Any, ::DryRunValue, ::Any) = DryRunValue()
Base.:(:)(::Any, ::Any, ::DryRunValue) = DryRunValue()
