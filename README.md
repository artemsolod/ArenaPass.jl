# ArenaPass.jl

Arena (bump) allocation for temporary Julia arrays — **without type piracy**.

```julia
using ArenaPass

f(N) = (tmp = zeros(N); tmp .= 1:N; sum(tmp))

@arena f(1_000_000)   # tmp comes from a bump arena, freed at scope exit — no GC garbage
```

`@arena` runs the call with all sufficiently large isbits array allocations
(anywhere in the call graph — including inside library code and spawned
tasks) served from a task-local bump arena that is released when the scope
exits. The caller promises that no arena-allocated array outlives the scope;
in exchange, temporaries cost near-zero GC traffic.

This is the successor experiment to
[ArenaPirate.jl](https://github.com/artemsolod/ArenaPirate.jl), which gets
the same effect by pirating `Memory{T}(undef, n)` globally — at the cost of
mass invalidation and package-incompatible semantics. ArenaPass touches no
global method: nothing outside `@arena` is affected, and loading it
invalidates nothing.

## How it works

The same method ArenaPirate redefines globally is instead **overlaid in a
method table that only exists inside a custom compilation world**
(`Base.Experimental.@MethodTable` + a custom `AbstractInterpreter`, following
the CompilerDevTools pattern from the julia repo). `@arena f(x)` compiles `f`
in that world and invokes the result; statically resolved callees stay
in-world, so the swap covers the visible call graph.

Two IR rewrites close the gaps that would otherwise leak back to native
(GC-allocating) code:

- **Deep mode** (`ArenaPass.DEEP[]`, default on): residual dynamic call
  sites — dynamic `:call`s, unknown-length splats via `_apply_iterate`, and
  bare-`MethodInstance` invokes — are rewritten to a shim that compiles the
  callee in-world for the runtime types and invokes it. This is what makes
  type-erased library code (e.g. DataFrames internals) allocate from the
  arena too.
- **Spawn rewrite** (`ArenaPass.SPAWN[]`, default on): task creation
  (`Threads.@spawn`, `@async`, `Task(f)`) is rewritten so a task spawned
  inside `@arena` gets its **own** arena (single owner — no bump races), and
  at task end its chunks are donated to the enclosing scope. Child
  allocations, including arrays you `fetch`, stay valid until the scope
  exits — as if the spawned code had run synchronously.

The runtime is a set of chunked arenas over a global, byte-capped warm chunk
store: arenas are free to create, grow on demand (8 MiB uniform chunks;
oversized allocations get a dedicated right-sized chunk), bump lock-free,
and return their chunks to the store at scope exit for reuse across scopes
and tasks. There is no fixed arena size and no arena-OOM.

## Usage

```julia
using ArenaPass

# call form and block form
@arena f(x, y)
@arena begin
    tmp = zeros(N)
    ...
end

# suppress arena allocation for a dynamic extent (composable, task-inherited)
@arena begin
    inner = heavy_computation()       # arena
    result = @noarena copy(inner)     # GC-backed: safe to return from the scope
    result
end

# tasks spawned inside a scope are arena-aware automatically
@arena begin
    t = Threads.@spawn make_temporary(N)
    consume(fetch(t))                 # fetch'd array valid until scope exit
end

# a task that must OUTLIVE the scope (e.g. a long-lived worker): opt out
@arena begin
    @noarena Threads.@spawn worker_loop()
    ...
end

arena_stats()          # (nallocs, chunks_created, store_bytes, ...)
reset_arena_stats!()
```

**The contract** (same as ArenaPirate): nothing arena-allocated may escape
the outermost `@arena` scope. Escaping values must be copied out under
`@noarena` (or be scalars/small arrays — allocations under
`ArenaPass.MIN_BYTES[]`, default 1024 bytes, and non-isbits eltypes always
use the GC). Violations are not detected: the memory is reused after scope
exit and the escaped array silently corrupts.

## What it buys

Measured on Julia 1.12.6 (all workloads verbatim, no code changes):

| workload | GC bytes/call | time |
|---|---|---|
| `median(rand(10^7))` ×32, 6 threads | 2.45 GiB → 120 KiB | 0.41 s → 0.22 s |
| `median(df[!, :a])` on a `DataFrame` (dynamic dispatch throughout) | 16.5 MB → 5 KB | 7.7× faster |
| `groupby` + `combine`, 10⁶ rows / 10⁵ groups | 41.7 MB → 0.05 MB | 19.8 ms → 10.5 ms |

The target scenario is multi-threaded code whose temporaries create GC
pressure that serial benchmarks never show.

**Anti-pattern**: deep mode makes each *runtime* dynamic dispatch ~70×
slower than native dispatch (it re-drives compilation machinery). Well-typed
code has none on hot paths; a type-unstable tight loop (e.g. summing a
`Vector{Any}` element-wise) inside `@arena` will crawl. Type-erased *library*
code is generally fine — e.g. DataFrames' function-barrier architecture does
a constant ~100 dispatches per `groupby`+`combine` call regardless of data
size.

## Status and caveats

Experimental. Relies on `Base.Compiler` internals and
`Core.OptimizedGenerics.CompilerPlugins`; **requires Julia 1.12** (developed
and tested on 1.12.6).

- First `@arena` call on a new call graph pays in-world inference (seconds
  for a large library graph, ~0.1–2 s typically; steady state is ~50–70 µs
  of scope overhead per call). The compiler-side fixed cost is precompiled
  into the pkgimage; per-graph inference cannot be cached across sessions on
  1.12 (external cache-owner CodeInstances don't serialize).
- Only isbits element types with allocations ≥ `MIN_BYTES[]` are arena'd.
- Calls initiated by the C runtime (finalizers, async callbacks) stay
  native — which is the correct behavior, since those may outlive the scope.
- Method redefinition is handled by ordinary backedge invalidation (the
  arena world's caches are invalidated like any other); redefining a
  function mid-session just triggers recompilation.

## Knobs

All runtime-settable; the compile-time flags (`DEEP`, `SPAWN`) must be set
before the first `@arena` call.

| knob | default | meaning |
|---|---|---|
| `ArenaPass.DEEP[]` | `true` | rewrite dynamic call sites to stay in-world |
| `ArenaPass.SPAWN[]` | `true` | rewrite task creation to be arena-aware |
| `ArenaPass.MIN_BYTES[]` | `1024` | smaller allocations use the GC |
| `ArenaPass.CHUNK_SIZE[]` | `8 MiB` | uniform chunk size |
| `ArenaPass.STORE_MAX_BYTES[]` | `max(4 GiB, 512 MiB × nthreads)` | warm-store cap; excess chunks go to GC |
| `ArenaPass.HUGEPAGES[]` | `true` on Linux | back chunks with anonymous mmap + `madvise(MADV_HUGEPAGE)` (2 MiB-aligned; what [julia#59858](https://github.com/JuliaLang/julia/pull/59858) does for large GC allocations from 1.13) |
| `ArenaPass.MMAP_MAX_BYTES[]` | RAM/2 | ceiling on total mmap'd chunk bytes (`arena_stats().mmap_live`); above it chunks fall back to GC allocation |
| `ArenaPass.SERIAL_COMPILE[]` | `true` | serialize in-world compilation (cache hits stay concurrent) |

**Hunting contract violations**: set `ArenaPass.QUARANTINE[] = true` (before
the first `@arena`) and rerun the suspect workload. Released chunks are then
mprotect'ed and leaked instead of reused — RSS stays flat, but **any use of
an arena array that escaped its scope faults deterministically at the guilty
access**, with a clean backtrace naming the code holding the reference
(instead of silent corruption, or a delayed crash hidden inside the
unwinder). `arena_stats().quarantined` counts sealed chunks. Debug-only:
address space grows for the length of the session.

**Sizing the store**: the warm working set is (concurrent scopes) ×
(per-scope peak allocation). If the cap is below it, chunks are trimmed to
the GC at every scope exit and freshly allocated at every entry — arena
traffic silently degenerates into GC traffic. `arena_stats().chunks_trimmed`
growing across steady-state runs is the tell; raise `STORE_MAX_BYTES[]`
until it stops.

## License

MIT
