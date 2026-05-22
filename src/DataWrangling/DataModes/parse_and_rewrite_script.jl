const PASSTHROUGH_HEADS = Set([:using, :import, :export, :module, :struct, :abstract, :primitive, :macro, :macrocall, :const])

function is_include_call(s)
    s isa Expr && s.head === :call && !isempty(s.args) || return false
    f = s.args[1]
    f === :include && return true
    if f isa Expr && f.head === :. && length(f.args) == 2
        f.args[2] === QuoteNode(:include) && return true
    end
    return false
end

function is_function_def(s)
    s isa Expr || return false
    s.head === :function && return true
    if s.head === :(=) && length(s.args) == 2
        lhs = s.args[1]
        lhs isa Expr || return false
        lhs.head === :call          && return true
        lhs.head === :where         && return is_function_def(Expr(:(=), lhs.args[1], s.args[2]))
        lhs.head === :(::) && lhs.args[1] isa Expr && lhs.args[1].head === :call && return true
    end
    return false
end

function wrap_assignment(lhs, rhs)
    if lhs isa Symbol
        return Expr(:(=), lhs, :(try; $rhs; catch; $DryRunValue(); end))
    end
    return :(try; $lhs = $rhs; catch; end)
end

wrap_return(args::Vector) = Expr(:try, Expr(:return, args...), false, Expr(:return, :($DryRunValue())))
wrap_bare(expr)           = Expr(:try, expr, false, Expr(:block))

function rewrite_block(body, basedir::AbstractString)
    if body isa Expr && (body.head === :block || body.head === :toplevel)
        return Expr(body.head, [rewrite_statement(a, basedir) for a in body.args]...)
    else
        return rewrite_statement(body, basedir)
    end
end

function rewrite_if(s, basedir::AbstractString)
    new_then = rewrite_block(s.args[2], basedir)
    if length(s.args) >= 3
        else_branch = s.args[3]
        new_else = (else_branch isa Expr && (else_branch.head === :elseif || else_branch.head === :if)) ?
                   rewrite_if(else_branch, basedir) : rewrite_block(else_branch, basedir)
        return Expr(s.head, s.args[1], new_then, new_else)
    end
    return Expr(s.head, s.args[1], new_then)
end

function rewrite_function_body(s, basedir::AbstractString)
    if s.head === :function
        return Expr(:function, s.args[1], rewrite_block(s.args[2], basedir))
    elseif s.head === :(=)
        return Expr(:(=), s.args[1], rewrite_block(s.args[2], basedir))
    elseif s.head === :(->)
        return Expr(:(->), s.args[1], rewrite_block(s.args[2], basedir))
    elseif s.head === :do
        anon = s.args[2]
        new_anon = Expr(:(->), anon.args[1], rewrite_block(anon.args[2], basedir))
        return Expr(:do, s.args[1], new_anon)
    end
    return s
end

function inline_include(s, basedir::AbstractString)
    path_arg = s.args[2]
    path_arg isa AbstractString || return wrap_bare(s)
    full_path = isabspath(path_arg) ? path_arg : joinpath(basedir, path_arg)
    isfile(full_path) || return wrap_bare(s)
    inner_source = read(full_path, String)
    inner_parsed = Meta.parseall(inner_source; filename = full_path)
    inner_basedir = dirname(abspath(full_path))
    return Expr(:toplevel, [rewrite_statement(a, inner_basedir) for a in inner_parsed.args]...)
end

function rewrite_statement(s, basedir::AbstractString)
    s isa LineNumberNode && return s
    s isa Expr || return wrap_bare(s)

    h = s.head

    h in PASSTHROUGH_HEADS && return s

    is_include_call(s) && return inline_include(s, basedir)

    if is_function_def(s) || h === :(->)
        return rewrite_function_body(s, basedir)
    end

    h === :do && return wrap_bare(rewrite_function_body(s, basedir))

    if h === :(=)
        lhs, rhs = s.args
        return wrap_assignment(lhs, rhs)
    end

    h === :return && return wrap_return(s.args)

    if h === :for || h === :while || h === :let
        new_body = rewrite_block(s.args[2], basedir)
        return wrap_bare(Expr(h, s.args[1], new_body))
    end

    (h === :if || h === :elseif) && return wrap_bare(rewrite_if(s, basedir))

    if h === :block || h === :toplevel
        return Expr(h, [rewrite_statement(a, basedir) for a in s.args]...)
    end

    h === :quote && return s

    return wrap_bare(s)
end