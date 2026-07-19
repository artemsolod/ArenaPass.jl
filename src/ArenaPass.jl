# ArenaPass: ArenaPirate semantics (manual @arena scoping, trust the user)
# WITHOUT type piracy and WITHOUT escape analysis.
#
# Mechanism: the same method ArenaPirate redefines globally —
# `Memory{T}(::UndefInitializer, m)` — is overlaid in a
# `Base.Experimental.@MethodTable` that is only consulted when code is
# compiled under our own AbstractInterpreter (the "arena world", following
# the CompilerDevTools pattern from the julia repo). `invoke(f, ci, args...)`
# keeps every statically-resolved callee in-world, so the swap covers the
# whole visible call graph; dynamic dispatch escapes to native code and
# simply GC-allocates (safe fallback). Nothing outside `@arena` is affected,
# and no global method is touched — zero invalidation.
module ArenaPass

export @arena, @noarena, arena_stats, reset_arena_stats!

using Base.ScopedValues: ScopedValue, with

const Compiler = Base.Compiler
using Core.IR

# ---------------- arena runtime (always runs in the native world) ----------------
# CHUNKED arenas over a global, byte-capped chunk store.
#
#  - An Arena is a lightweight list of chunks, bump-allocated lock-free
#    (single task owns it). Creating one is free — no upfront memory — so
#    per-task arenas (SPAWN rewrite) cost only what the task allocates.
#  - Chunks are uniform CHUNK_SIZE blocks; allocations bigger than half a
#    chunk get a dedicated right-sized chunk. All chunks return to the global
#    warm store at scope exit and are reused across scopes/tasks; the store is
#    trimmed to STORE_MAX_BYTES (excess dropped to GC). No fixed arena size,
#    no arena-OOM: arenas grow on demand.
#  - The SPAWN rewrite gives a task spawned inside @arena its own arena; at
#    task end its chunks are DONATED to the enclosing scope's arena, so child
#    allocations (incl. fetch'd arrays) stay valid until the scope exits — the
#    same lifetime as if the spawned code had run synchronously. One contract:
#    nothing arena'd escapes the outermost @arena.
mutable struct Arena
    chunks::Vector{Memory{UInt8}}          # in use, activation order
    cur::Int                               # index of current bump chunk (0 = none)
    pos::Int                               # bump position in chunks[cur]
    donated::Vector{Memory{UInt8}}         # from finished child tasks (donate_lock)
    const donate_lock::ReentrantLock
end
Arena() = Arena(Memory{UInt8}[], 0, 0, Memory{UInt8}[], ReentrantLock())

const CHUNK_SIZE = Ref(1 << 23)        # 8 MiB uniform chunks
const STORE_MAX_BYTES = Ref(1 << 32)   # keep ≤ 4 GiB warm in the store
const MIN_BYTES = Ref(1024)

const CHUNK_STORE = Memory{UInt8}[]
const STORE_LOCK = ReentrantLock()
const CHUNKS_CREATED = Threads.Atomic{Int}(0)
const NALLOCS = Threads.Atomic{Int}(0)
const FALLBACKS = Threads.Atomic{Int}(0)   # kept for API compat; unused (no arena-OOM)

"Take the smallest stored chunk with length ≥ minsz, or allocate a fresh one."
function take_chunk!(minsz::Int)::Memory{UInt8}
    c = lock(STORE_LOCK) do
        best, bestlen = 0, typemax(Int)
        for (i, ch) in enumerate(CHUNK_STORE)
            l = length(ch)
            if minsz <= l < bestlen
                best, bestlen = i, l
            end
        end
        best == 0 ? nothing :
            (ch = CHUNK_STORE[best]; deleteat!(CHUNK_STORE, best); ch)
    end
    c isa Memory{UInt8} && return c
    Threads.atomic_add!(CHUNKS_CREATED, 1)
    return Memory{UInt8}(undef, max(minsz, CHUNK_SIZE[]))
end

function return_chunks!(chunks::Vector{Memory{UInt8}})
    isempty(chunks) && return nothing
    lock(STORE_LOCK) do
        append!(CHUNK_STORE, chunks)
        total = 0
        for ch in CHUNK_STORE
            total += length(ch)
        end
        while total > STORE_MAX_BYTES[] && !isempty(CHUNK_STORE)
            total -= length(pop!(CHUNK_STORE))  # excess goes to GC
        end
    end
    empty!(chunks)
    return nothing
end

function bump!(a::Arena, sz::Int)::Ptr{UInt8}
    if a.cur != 0
        chunk = a.chunks[a.cur]
        pos = (a.pos + 63) & ~63
        if pos + sz <= length(chunk)
            a.pos = pos + sz
            return pointer(chunk) + pos
        end
    end
    if sz > CHUNK_SIZE[] >> 1
        chunk = take_chunk!(sz)          # dedicated; current bump chunk unchanged
        push!(a.chunks, chunk)
        return pointer(chunk)
    else
        chunk = take_chunk!(CHUNK_SIZE[])
        push!(a.chunks, chunk)
        a.cur = length(a.chunks)
        a.pos = sz
        return pointer(chunk)
    end
end

arena_mark(a::Arena) = (length(a.chunks), a.cur, a.pos)
function arena_reset!(a::Arena, mark::Tuple{Int,Int,Int})
    n, cur, pos = mark
    if length(a.chunks) > n
        extra = a.chunks[n+1:end]
        resize!(a.chunks, n)
        return_chunks!(extra)
    end
    a.cur, a.pos = cur, pos
    return nothing
end

"Move a finished child arena's chunks into `parent` (they live until scope exit)."
function donate!(parent::Arena, child::Arena)
    lock(parent.donate_lock) do
        append!(parent.donated, child.chunks)
        append!(parent.donated, child.donated)   # grandchildren chain up
    end
    empty!(child.chunks)
    empty!(child.donated)
    return nothing
end

function release_scope!(a::Arena)
    return_chunks!(a.chunks)
    a.cur = 0
    a.pos = 0
    donated = lock(a.donate_lock) do
        d = copy(a.donated)
        empty!(a.donated)
        d
    end
    return_chunks!(donated)
    return nothing
end

function active_arena()::Union{Arena,Nothing}
    a = get(task_local_storage(), :arenascope, nothing)
    return a isa Arena ? a : nothing
end

# Dynamic-extent suppression: a runtime check in the allocator, so it applies
# through the whole call tree (incl. tasks spawned in scope — ScopedValues are
# inherited). Suppression via scoped state works because the check lives in
# code we compiled; the converse (scoped ENABLEMENT reaching native code) is
# impossible — native code has no check compiled in.
const NOARENA = ScopedValue(false)

@noinline function arena_memorynew(::Type{Memory{T}}, m::Int) where {T}
    a = active_arena()
    a === nothing && return Core.memorynew(Memory{T}, m)  # outside any @arena scope
    NOARENA[] && return Core.memorynew(Memory{T}, m)      # inside @noarena extent
    # sizeof(T)*m overflow would silently hand back an undersized buffer;
    # route anything suspicious to the builtin, which validates and throws
    elsz = sizeof(T)
    (0 <= m <= (typemax(Int) - 64) ÷ max(elsz, 1) && elsz > 0) ||
        return Core.memorynew(Memory{T}, m)
    ptr = bump!(a, elsz * m)
    Threads.atomic_add!(NALLOCS, 1)
    return unsafe_wrap(Memory{T}, Ptr{T}(ptr), m)
end

function arena_stats()
    store_bytes = lock(STORE_LOCK) do
        s = 0
        for ch in CHUNK_STORE
            s += length(ch)
        end
        s
    end
    return (nallocs=NALLOCS[], fallbacks=FALLBACKS[],
            chunks_created=CHUNKS_CREATED[], store_bytes=store_bytes)
end
reset_arena_stats!() = (NALLOCS[] = 0; FALLBACKS[] = 0; nothing)

# ---------------- the scoped method swap ----------------
Base.Experimental.@MethodTable ARENA_TABLE

Base.Experimental.@overlay ARENA_TABLE function (::Type{Memory{T}})(::UndefInitializer, m::Int) where {T}
    if isbitstype(T) && sizeof(T) * m >= MIN_BYTES[]
        # deliberate dynamic call: dynamic dispatch resolves in the NATIVE cache,
        # so the allocator and everything it calls (Dict/TLS machinery) compiles
        # and runs outside the arena world and cannot re-enter this overlay.
        # See run_probe.jl: a direct call here leaves the allocator in-world and
        # its own allocations recurse into the overlay.
        return (Base.invokelatest(arena_memorynew, Memory{T}, m))::Memory{T}
    end
    return Core.memorynew(Memory{T}, m)  # builtin: not overlayable
end

# ---------------- interpreter (CompilerDevTools pattern) ----------------
struct ArenaCacheOwner end

struct ArenaInterp <: Compiler.AbstractInterpreter
    world::UInt
    inf_params::Compiler.InferenceParams
    opt_params::Compiler.OptimizationParams
    inf_cache::Vector{Compiler.InferenceResult}
    codegen_cache::IdDict{CodeInstance,CodeInfo}
    function ArenaInterp(; world::UInt=Base.get_world_counter())
        new(world, Compiler.InferenceParams(), Compiler.OptimizationParams(),
            Compiler.InferenceResult[], IdDict{CodeInstance,CodeInfo}())
    end
end

Compiler.InferenceParams(interp::ArenaInterp) = interp.inf_params
Compiler.OptimizationParams(interp::ArenaInterp) = interp.opt_params
Compiler.get_inference_world(interp::ArenaInterp) = interp.world
Compiler.get_inference_cache(interp::ArenaInterp) = interp.inf_cache
Compiler.cache_owner(::ArenaInterp) = ArenaCacheOwner()
Compiler.codegen_cache(interp::ArenaInterp) = interp.codegen_cache
Compiler.method_table(interp::ArenaInterp) =
    Compiler.OverlayMethodTable(interp.world, ARENA_TABLE)

# NOTE: the CompilerPlugins typeinf anchor is defined at the BOTTOM of this
# module: it pins pipeline execution to its own primary_world via
# invoke_in_world, so every Compiler method override (e.g. our optimize)
# must already exist at that world or it will be silently ignored.

# ---------------- deep mode: close the dynamic-dispatch boundary ----------------
# Piracy's one real advantage is that dynamically-dispatched callees are still
# redirected. Deep mode recovers that without piracy: an IR pass rewrites every
# residual dynamic :call in in-world code into `arena_dispatch(f, args...)`,
# which compiles the callee in-world FOR THE RUNTIME TYPES and invokes it —
# so dynamic dispatch re-enters the world instead of leaking to native.
# NOTE: read at compile time — set DEEP[] before the first @arena call.
const DEEP = Ref(true)   # deep mode ON by default
const SPAWN = Ref(true)  # rewrite task creation so spawned bodies stay arena-aware
const REWRITES = Threads.Atomic{Int}(0)
const TASK_REWRITES = Threads.Atomic{Int}(0)
const DISPATCH_HITS = Threads.Atomic{Int}(0)
const DISPATCH_FALLBACKS = Threads.Atomic{Int}(0)

function rewrite_dynamic_calls!(ir)
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        stmt isa Expr || continue
        # Task creation (Threads.@spawn / @async / Task(f)): the runtime starts
        # a task through the NATIVE dispatcher, so without a rewrite the spawned
        # body silently loses arena coverage. Two shapes to catch:
        if stmt.head === :foreigncall
            # (a) an inlined jl_new_task foreigncall. Layout: args[1:5] =
            # name/RT/argtypes/nreq/cconv, args[6:8] = (f, completion, ssize).
            SPAWN[] || continue
            nm = stmt.args[1]
            nm isa QuoteNode && (nm = nm.value)
            nm === :jl_new_task || continue
            ir.stmts[i][:stmt] = Expr(:call, GlobalRef(ArenaPass, :arena_new_task),
                                      stmt.args[6], stmt.args[7], stmt.args[8])
            ir.stmts[i][:flag] = zero(ir.stmts[i][:flag])
            Threads.atomic_add!(TASK_REWRITES, 1)
            continue
        end
        if SPAWN[] && (stmt.head === :invoke || stmt.head === :call)
            # (b) caller-side construction: depending on inlining depth the site
            # is `Task(f)` / `Task(f, ssize)` or `Core._Task(f, ssize, compl)`,
            # and as an :invoke it can carry a bare MethodInstance — which
            # executes NATIVE code, bypassing any rewrite inside the callee's
            # in-world CodeInstance. Rewriting the construction site itself
            # works regardless of how the callee edge resolved.
            off = stmt.head === :invoke ? 2 : 1
            callee = Compiler.singleton_type(Compiler.argextype(stmt.args[off], ir))
            nargs = length(stmt.args) - off
            tgt = callee === Core._Task && nargs == 3 ? :arena_task_ctor :
                  callee === Task && 1 <= nargs <= 2  ? :arena_Task : nothing
            if tgt !== nothing
                ir.stmts[i][:stmt] = Expr(:call, GlobalRef(ArenaPass, tgt),
                                          stmt.args[off+1:end]...)
                ir.stmts[i][:flag] = zero(ir.stmts[i][:flag])
                Threads.atomic_add!(TASK_REWRITES, 1)
                continue
            end
        end
        DEEP[] || continue
        if stmt.head === :invoke && stmt.args[1] isa MethodInstance &&
           !(ir.stmts[i][:info] isa Compiler.InvokeCallInfo)
            # bare-MethodInstance invoke ("dynamic invoke"): no in-world
            # CodeInstance was attached, so at runtime jl_invoke compiles and
            # runs the NATIVE code for the MI — a silent world escape, same
            # class of leak as dynamic :calls. Since inference derived the MI
            # from ordinary dispatch (info is not an explicit Base.invoke),
            # re-dispatching on runtime types via arena_dispatch is equivalent
            # and keeps the callee in-world.
            ir.stmts[i][:stmt] = Expr(:call, GlobalRef(ArenaPass, :arena_dispatch),
                                      stmt.args[2:end]...)
            ir.stmts[i][:flag] = zero(ir.stmts[i][:flag])
            Threads.atomic_add!(REWRITES, 1)
            continue
        end
        stmt.head === :call || continue
        ft = Compiler.argextype(stmt.args[1], ir)
        callee = Compiler.singleton_type(ft)
        if callee === Core._apply_iterate && length(stmt.args) >= 3
            # dynamic splat f(xs...): the builtin dispatches f internally via the
            # native cache — reroute through arena_apply so the callee compiles
            # in-world too. Skip if the applied callee is a builtin or ours.
            fa = Compiler.singleton_type(Compiler.argextype(stmt.args[3], ir))
            (fa isa Core.Builtin || fa isa Core.IntrinsicFunction) && continue
            fa isa Function && typeof(fa).name.module === ArenaPass && continue
            ir.stmts[i][:stmt] = Expr(:call, GlobalRef(ArenaPass, :arena_apply),
                                      stmt.args[2:end]...)
            ir.stmts[i][:flag] = zero(ir.stmts[i][:flag])
            Threads.atomic_add!(REWRITES, 1)
            continue
        end
        callee isa Core.Builtin && continue          # invoke, throw, _call_latest…
        callee isa Core.IntrinsicFunction && continue
        callee === arena_dispatch && continue
        ir.stmts[i][:stmt] = Expr(:call, GlobalRef(ArenaPass, :arena_dispatch),
                                  stmt.args...)
        ir.stmts[i][:flag] = zero(ir.stmts[i][:flag])
        Threads.atomic_add!(REWRITES, 1)
    end
    return ir
end

function Compiler.optimize(interp::ArenaInterp, opt::Compiler.OptimizationState,
                           caller::Compiler.InferenceResult)
    ir = Compiler.run_passes_ipo_safe(opt.src, opt)
    Compiler.ipo_dataflow_analysis!(interp, opt, ir, caller)
    (DEEP[] || SPAWN[]) && rewrite_dynamic_calls!(ir)
    return Compiler.finish(interp, opt, ir, caller)
end

"Deep-mode shim for dynamic splats: flatten args like _apply_iterate, then dispatch in-world."
@noinline function arena_apply(iter, @nospecialize(f), @nospecialize(splatargs...))
    typeof(f).name.module === ArenaPass &&
        return Core._apply_iterate(iter, f, splatargs...)
    args = Core._apply_iterate(iter, Core.tuple, splatargs...)  # single flattening pass
    ci = try
        arena_codeinstance(f, args)
    catch
        nothing
    end
    ci === nothing && return Core._apply_iterate(iter, f, splatargs...)
    return invoke(f, ci, args...)
end

"Runtime side of deep mode: dispatch on runtime types, but compiled in-world."
@noinline function arena_dispatch(@nospecialize(f), @nospecialize(args...))
    Threads.atomic_add!(DISPATCH_HITS, 1)
    # never pull our own machinery (scope bookkeeping, allocator) into the world
    typeof(f).name.module === ArenaPass && return f(args...)
    ci = try
        arena_codeinstance(f, args)
    catch
        nothing
    end
    if ci === nothing
        Threads.atomic_add!(DISPATCH_FALLBACKS, 1)
        return f(args...)   # no method / uncompilable: stay native
    end
    return invoke(f, ci, args...)
end

# Runtime side of the jl_new_task rewrite. The wrapper is a struct (not a
# closure) so arena_new_task can recognize already-wrapped start functions and
# never double-wrap. The shim itself runs native when the task starts;
# spawned_with_arena then compiles the real body in-world.
struct SpawnShim
    f::Any
    parent::Arena
end
(s::SpawnShim)() = spawned_with_arena(s.parent, s.f)

"Wrap a task start function so the task runs with its own arena and donates
its chunks to the enclosing scope at task end ('as if synchronous').
Captures the CREATING task's arena now — by the time the task runs, tls
belongs to the child. No-op outside a scope, under @noarena, or if wrapped.
To spawn a task that must NOT be arena-managed (e.g. a worker that outlives
the scope), create it under @noarena. For huge task counts where per-task
reclamation matters more, use `Threads.@spawn @arena ...` (independent scope,
reclaimed at task end; nothing arena'd may escape the task)."
function wrap_task_start(@nospecialize(f))
    parent = active_arena()
    (parent === nothing || NOARENA[] || f isa SpawnShim) && return f
    return SpawnShim(f, parent)
end

# rewrite target for inlined jl_new_task foreigncalls
@noinline function arena_new_task(@nospecialize(f), @nospecialize(completion), ssize::Int)
    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int),
                 wrap_task_start(f), completion, ssize)
end

# rewrite target for Core._Task(f, ssize, completion) call sites
@noinline function arena_task_ctor(@nospecialize(f), ssize::Int, @nospecialize(completion))
    return Core._Task(wrap_task_start(f), ssize, completion)
end

# rewrite targets for Task(f) / Task(f, reserved_stack) ctor call sites
@noinline arena_Task(@nospecialize(f)) = Task(wrap_task_start(f))
@noinline arena_Task(@nospecialize(f), ssize::Int) = Task(wrap_task_start(f), ssize)

# ---------------- entry: ArenaPirate-like interface ----------------
const CI_CACHE = Dict{Any,CodeInstance}()
const CI_LOCK = ReentrantLock()

function arena_codeinstance(@nospecialize(f), @nospecialize(args::Tuple))
    # Core.Typeof, NOT typeof: for a type-valued argument typeof collapses to
    # DataType, so e.g. f(String) and f(Int) would share a cache key while the
    # cached CodeInstance is specialized for whichever type value came first —
    # the later call then dies in `invoke` with a signature TypeError.
    # Core.Typeof(String) == Type{String} keeps the keys apart (and matches
    # how jl_method_lookup specializes on the actual values).
    key = Any[Core.Typeof(f)]
    for a in args
        push!(key, Core.Typeof(a))
    end
    sig = Tuple{key...}
    world = Base.tls_world_age()
    lock(CI_LOCK) do
        ci = get(CI_CACHE, sig, nothing)
        if ci isa CodeInstance && ci.min_world <= world <= ci.max_world
            return ci
        end
        miptr = @ccall jl_method_lookup(Any[f, args...]::Ptr{Any}, (1 + length(args))::Csize_t,
                                        world::Csize_t)::Ptr{Cvoid}
        miptr == C_NULL && return nothing
        mi = unsafe_pointer_to_objref(miptr)::Core.MethodInstance
        ci = typeinf(ArenaCacheOwner(), mi, Compiler.SOURCE_MODE_ABI)
        ci isa CodeInstance || return nothing
        CI_CACHE[sig] = ci
        return ci
    end
end

@noinline function arena_invoke(f, args...)
    ci = arena_codeinstance(f, args)
    ci === nothing && return f(args...)   # uncompilable: run native, no scope
    tls = task_local_storage()
    a = get(tls, :arenascope, nothing)
    if a isa Arena  # nested scope: mark/reset on the already-owned arena
        mark = arena_mark(a)
        try
            return invoke(f, ci, args...)
        finally
            arena_reset!(a, mark)
        end
    else            # outermost scope: fresh (empty, free-to-create) arena
        a = Arena()
        tls[:arenascope] = a
        try
            return invoke(f, ci, args...)
        finally
            delete!(tls, :arenascope)
            release_scope!(a)
        end
    end
end

"Runtime body of a SpawnShim'd task: run `f` in the child with its own arena,
then donate the chunks to the enclosing scope's arena."
function spawned_with_arena(parent::Arena, f)
    a = Arena()
    tls = task_local_storage()
    tls[:arenascope] = a
    try
        # the child body must run IN-WORLD (a plain f() here would execute
        # native code and GC-allocate despite the arena being set up)
        ci = arena_codeinstance(f, ())
        return ci === nothing ? f() : invoke(f, ci)
    finally
        delete!(tls, :arenascope)
        donate!(parent, a)
    end
end

import Core.OptimizedGenerics.CompilerPlugins: typeinf, typeinf_edge
@eval @noinline typeinf(::ArenaCacheOwner, mi::MethodInstance, source_mode::UInt8) =
    Base.invoke_in_world(which(typeinf, Tuple{ArenaCacheOwner,MethodInstance,UInt8}).primary_world,
                         Compiler.typeinf_ext_toplevel, ArenaInterp(; world=Base.tls_world_age()),
                         mi, source_mode)

@eval @noinline function typeinf_edge(::ArenaCacheOwner, mi::MethodInstance,
                                      parent_frame::Compiler.InferenceState,
                                      world::UInt, source_mode::UInt8)
    interp = ArenaInterp(; world)
    Compiler.typeinf_edge(interp, mi.def, mi.specTypes, Core.svec(), parent_frame, false, false)
end

"""
    @arena f(args...)
    @arena begin ... end

Run the call/block with temporary isbits allocations (≥ `MIN_BYTES[]`) served
from a task-local bump arena, freed when the (outermost) `@arena` scope exits.
ArenaPirate contract: the caller guarantees no arena-allocated array outlives
the scope. Non-call expressions are wrapped in a closure and compiled the same way.
"""
macro arena(ex)
    # inferencebarrier: when @arena appears inside code that itself runs in the
    # arena world (nested scopes), the dynamic call drops back to the native
    # world so the scope bookkeeping (pool lock, typeinf) never compiles in-world
    if Meta.isexpr(ex, :call) &&
       !any(a -> Meta.isexpr(a, (:kw, :parameters, :(...))), ex.args)
        return esc(:($Base.inferencebarrier($ArenaPass.arena_invoke)($(ex.args...))))
    else
        return esc(:($Base.inferencebarrier($ArenaPass.arena_invoke)(() -> $ex)))
    end
end

"""
    @noarena expr

Suppress arena allocation for the dynamic extent of `expr` (ArenaPirate's
`@noarena`, but composable: inherited by tasks spawned inside, and it cannot
leak — it is a runtime check in the allocator, not a compilation mode).
"""
macro noarena(ex)
    return esc(:($ArenaPass.with_noarena(() -> $ex)))
end
with_noarena(f) = with(f, NOARENA => true)

# ---------------- package lifecycle ----------------
function __init__()
    # entries created during precompile carry that session's world ranges and
    # state — start every session clean (pkgimage-cached CodeInstances are
    # still found through the runtime's own cache when valid)
    empty!(CI_CACHE)
    empty!(CHUNK_STORE)
    NALLOCS[] = 0; FALLBACKS[] = 0; CHUNKS_CREATED[] = 0
    DISPATCH_HITS[] = 0; DISPATCH_FALLBACKS[] = 0; TASK_REWRITES[] = 0
    return nothing
end

# ---------------- precompile workload ----------------
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    _pc_work(N) = (tmp = zeros(N); tmp .= 1:N; sum(tmp))
    _pc_sortslice(v) = (s = sort(v); sum(s[1:end÷2]))
    @compile_workload begin
        @arena _pc_work(100)
        @arena _pc_sortslice(rand(100))
        @arena sum(collect(1:50))
    end
    empty!(CHUNK_STORE)  # never serialize chunk buffers into the pkgimage
    empty!(CI_CACHE)
end

end # module
