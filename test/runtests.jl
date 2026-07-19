# Tests for ArenaWorld: correctness, coverage through non-inlined callees,
# thunk form, and the user-responsibility contract.
using ArenaPass

# --- 1. the original motivating example, allocation behind @noinline -------
# (proves the swap reaches callees the outer frame does NOT inline — the
# coverage the single-frame EA spike could not give)
@noinline inner(N) = zeros(N)
function outer(N)
    tmp = inner(N)
    tmp .= 1:N
    return sum(tmp)
end

reset_arena_stats!()
t_compile = @elapsed r = @arena outer(1_000_000)
@assert r == outer(1_000_000)
st = arena_stats()
println("outer: OK (first call incl. compile: $(round(t_compile, digits=2))s)")
println("  arena allocations: ", st.nallocs, " (expect ≥1: the zeros(N) inside @noinline inner)")
@assert st.nallocs >= 1

# --- 2. correctness across sizes, incl. below MIN_BYTES ---------------------
for N in (1, 10, 127, 1000, 999_999)
    @assert (@arena outer(N)) == outer(N)
end
println("sizes: OK")

# --- 3. thunk form ----------------------------------------------------------
x = 42.0
r = @arena begin
    tmp = zeros(10_000)
    tmp .= x
    sum(tmp)
end
@assert r == 420_000.0
println("thunk form: OK")

# --- 4. nesting: inner @arena shares the task arena, resets to its own mark -
function nested(N)
    a = @arena outer(N)      # nested scope
    tmp = zeros(N)           # outer scope alloc
    tmp .= 2.0
    return a + sum(tmp)
end
@assert (@arena nested(1000)) == outer(1000) + 2000.0
println("nesting: OK")

# --- 5. allocation counting: GC traffic per call ---------------------------
warm() = @arena outer(1_000_000)
warm()
gcb = @allocated warm()
println("GC bytes per @arena outer(1_000_000) call: ", gcb, " (GC version: ~8MB)")

# --- 6. the contract: escaping the scope is the USER's bug ------------------
leak(N) = (tmp = zeros(N); tmp .= 7.0; tmp)
leaked = @arena leak(1000)
before = leaked[1]
@arena outer(1000)   # reuses the arena region
after = leaked[1]
println("contract demo: leaked[1] before=$before after=$after",
        after == before ? "  (undetected this time!)" : "  <- corrupted, as the contract warns")

# --- 7. multithreaded: pooled arenas, bounded by concurrency ----------------
function mtsum(n)
    res = zeros(n)
    Threads.@threads for i in 1:n
        res[i] = @arena outer(100_000 + i)
    end
    return res
end
expected = [outer(100_000 + i) for i in 1:32]
@assert mtsum(32) == expected
created_first = arena_stats().chunks_created
@assert mtsum(32) == expected   # second run must reuse warm chunks
created_second = arena_stats().chunks_created
println("threaded ($(Threads.nthreads()) threads): OK")
println("  chunks created: $created_first after 1st run, $created_second after 2nd",
        " (warm store reuse across 2×32 tasks)")
@assert created_second == created_first  # store reuse: no new chunks on 2nd run

# --- 8. @noarena: scoped suppression with dynamic extent --------------------
function partly_suppressed(N)
    a = sum(zeros(N))            # arena
    b = @noarena sum(zeros(N))   # suppressed: GC, even though we're in-world
    c = sum(zeros(N))            # arena again
    return a + b + c
end
reset_arena_stats!()
@assert (@arena partly_suppressed(100_000)) == 0.0
st = arena_stats()
println("@noarena: OK (arena allocs = $(st.nallocs), expect 2 of 3)")
@assert st.nallocs == 2

# --- 9. SPAWN rewrite: plain Threads.@spawn / @async are arena-aware — child
#         allocations arena'd, fetch'd arrays valid until the scope exits
#         (chunk donation), task-creation sites rewritten by the pass
function plain_spawny(N)
    t1 = Threads.@spawn (v = zeros(N); v .= 2.0; v)
    t2 = @async sum(zeros(N) .+ 1.0)
    v = fetch(t1)
    return sum(v) + fetch(t2)
end
reset_arena_stats!()
tr0 = ArenaPass.TASK_REWRITES[]
@assert (@arena plain_spawny(100_000)) == 300_000.0
st = arena_stats()
println("SPAWN rewrite (Threads.@spawn/@async): OK (arena allocs incl. children = $(st.nallocs), ",
        "task sites rewritten = $(ArenaPass.TASK_REWRITES[] - tr0))")
@assert st.nallocs >= 3
@assert ArenaPass.TASK_REWRITES[] - tr0 >= 1

# a task created under @noarena stays unmanaged (escape hatch for workers
# that outlive the scope)
function noarena_spawny(N)
    t = @noarena Threads.@spawn sum(zeros(N) .= 4.0)
    return fetch(t)
end
reset_arena_stats!()
@assert (@arena noarena_spawny(100_000)) == 400_000.0
@assert arena_stats().nallocs == 0
println("@noarena spawn escape hatch: OK")

# --- 10. deep-mode dispatch with Type-valued arguments: distinct type values
#          at one dynamic site must get distinct CodeInstances (regression:
#          typeof-based cache keys collapsed all Types to DataType, and the
#          second type value died in invoke with a signature TypeError)
@noinline typearg_callee(::Type{T}, s::Symbol) where {T} = (zeros(2000); T)
typearg_caller(T) = Base.inferencebarrier(typearg_callee)(T, :x)
@assert (@arena typearg_caller(String)) === String
@assert (@arena typearg_caller(Int)) === Int          # was: TypeError in invoke
mkvec(T, n) = Base.inferencebarrier(T)(undef, n)      # ctor in f-position
@assert length(@arena mkvec(Vector{Float64}, 7)) == 7
@assert length(@arena mkvec(Vector{Int32}, 8)) == 8   # same erasure class
println("Type-valued args at dynamic sites: OK")

# --- 11. hugepage-backed chunks: mmap-path lifecycle (alloc, warm reuse,
#          munmap finalizer). madvise(MADV_HUGEPAGE) only applies on Linux;
#          the mmap machinery itself is exercised on every platform.
old_cap = ArenaPass.STORE_MAX_BYTES[]
ArenaPass.STORE_MAX_BYTES[] = 0          # drain: scope exits drop all chunks
@arena outer(1_000_000)
ArenaPass.STORE_MAX_BYTES[] = old_cap
old_hp = ArenaPass.HUGEPAGES[]
ArenaPass.HUGEPAGES[] = true
c0 = arena_stats().chunks_created
@assert (@arena outer(1_000_000)) == outer(1_000_000)   # fresh mmap'd chunk
@assert arena_stats().chunks_created > c0
GC.gc(); GC.gc()                         # run munmap finalizers of dropped chunks
@assert (@arena outer(1_000_000)) == outer(1_000_000)   # warm mmap chunk reused
c1 = arena_stats().chunks_created
@assert (@arena outer(1_000_000)) == outer(1_000_000)
@assert arena_stats().chunks_created == c1
ArenaPass.HUGEPAGES[] = old_hp
println("hugepage chunk path: OK (mmap alloc, reuse, finalizer)")

println("\nall ArenaPass tests passed")
