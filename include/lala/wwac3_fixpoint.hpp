// Copyright 2026 - WWAC3 (warp-tile-granular AC3 worklist) fixpoint.

#ifndef LALA_CORE_WWAC3_FIXPOINT_HPP
#define LALA_CORE_WWAC3_FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"

#ifdef __CUDACC__

namespace lala {

/** WWAC3 device fixpoint: WAC3 at WARP-TILE granularity.
 *
 * Same role and interface as `wac3_fixpoint` (event-driven AC3 worklist), but
 * the unit of the frontier is a TILE of 32 CONSECUTIVE bytecodes instead of a
 * single bytecode. A warp owns one frontier tile: lane ℓ runs bytecode
 * `tile*32 + ℓ` to local fixpoint (when in range), and any lane that narrows a
 * variable dirties it. Phase 2 re-enqueues the incident TILES (not bytecodes)
 * of every dirtied variable via a var->tile CSR.
 *
 * Why it wins (vs per-bytecode WAC3, −12.3% geomean in the standalone
 * prototype): the 32 lanes process the tile's 32 bytecodes IN PARALLEL, so the
 * non-narrowing re-examinations a tile incurs are latency-hidden on this
 * latency-bound kernel; meanwhile the coarser (tile) frontier cuts the worklist
 * atomics and the number of rounds/barriers. Bimodal: big wins on deep/large
 * networks, small losses on tiny ones (no parallelism to hide the
 * over-approximation). A complement to WAC3, selected per problem.
 *
 * Frontier-buffer note: a tile id is in `[0, num_tiles)` with
 * `num_tiles = ceil(num_bytecodes / 32) <= num_bytecodes`, so this is a drop-in
 * for the WAC3 call site — the same per-block scratch (`frontier_a/b`,
 * `in_next`, all sized `num_bytecodes`) is large enough; only the leading
 * `num_tiles` entries are used.
 *
 * @tparam TPB    threads per block (must match the kernel launch).
 * @tparam A      an IProp-shaped abstract domain (same contract as
 *                `wac3_fixpoint`): `num_deductions()`, `load_deduce(int)`,
 *                `deduce_with_mask(ded)`, `is_bot()`, and a deduction `ded` with
 *                `.x/.y/.z.vid()`. PIR and PC satisfy this.
 *
 * @param iprop       abstract propagator (read+write).
 * @param vt_off      var->TILE CSR offsets, size `num_vars + 1` (built once in
 *                    UnifiedData; a var's incident tiles, deduped per tile).
 * @param vt          var->TILE CSR incidence list.
 * @param frontier_a  worklist buffer A (tile ids), size >= `num_tiles`.
 * @param frontier_b  worklist buffer B (tile ids), size >= `num_tiles`.
 * @param in_next     dedup flag over TILES, size >= `num_tiles`. PERSISTENT:
 *                    all-zero on entry; the algorithm restores it to all-zero on
 *                    exit (including the bot-exit cleanup tail).
 * @param var_dirty   in-list marker, size `num_vars`. PERSISTENT, all-zero on
 *                    entry; algorithm keeps it clean.
 * @param dirty_list  dense list of dirty vars (scratch), size `num_vars`.
 *
 * @return number of rounds performed (>=1; round 1 is a dense process-all-tiles
 *         pass, rounds 2+ process only the frontier tiles).
 */
template <int TPB, class A>
__device__ int wwac3_fixpoint(
    A& iprop,
    const int* __restrict__ vt_off,
    const int* __restrict__ vt,
    int* __restrict__ frontier_a,
    int* __restrict__ frontier_b,
    int* __restrict__ in_next,
    int* __restrict__ var_dirty,
    int* __restrict__ dirty_list,
    int* __restrict__ out_total_deduces)
{
  constexpr int NW = TPB / 32;            // warps per block
  const int wid  = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;

  __shared__ int  fsize;
  __shared__ int  nsize;
  __shared__ int  dirty_count;
  __shared__ int  rounds;
  __shared__ int  tot;
  __shared__ int* cur;                    // SHARED: the round-end swap is done by
  __shared__ int* nxt;                    // thread 0 and MUST be seen by all lanes.
  __shared__ int  num_bytecodes;
  __shared__ int  num_tiles;

  if(threadIdx.x == 0) {
    rounds = 0; tot = 0; nsize = 0; dirty_count = 0;
    cur = frontier_a;
    nxt = frontier_b;
    num_bytecodes = iprop.num_deductions();
    num_tiles = (num_bytecodes + 31) / 32;
  }
  __syncthreads();

  // One warp processes one tile: lane ℓ runs bytecode `t*32 + ℓ` to local
  // fixpoint, then dirties the vars it narrowed. Lanes are independent (no warp
  // cooperation): correctness comes from monotone narrowing + the outer
  // fixpoint, exactly as in WAC3. Out-of-range lanes (last partial tile) skip.
  #define WWAC3_PROCESS_TILE(T)                                                  \
    do {                                                                         \
      int _bidx = (T) * 32 + lane;                                              \
      if(_bidx < num_bytecodes) {                                              \
        auto _ded = iprop.load_deduce(_bidx);                                  \
        int _mask = 0, _m;                                                      \
        while((_m = iprop.deduce_with_mask(_ded)) && !iprop.is_bot()) { _mask |= _m; } \
        if((_mask & 1) && atomicExch(&var_dirty[_ded.x.vid()], 1) == 0)         \
          dirty_list[atomicAdd(&dirty_count, 1)] = _ded.x.vid();                \
        if((_mask & 2) && atomicExch(&var_dirty[_ded.y.vid()], 1) == 0)         \
          dirty_list[atomicAdd(&dirty_count, 1)] = _ded.y.vid();                \
        if((_mask & 4) && atomicExch(&var_dirty[_ded.z.vid()], 1) == 0)         \
          dirty_list[atomicAdd(&dirty_count, 1)] = _ded.z.vid();                \
      }                                                                          \
    } while(0)

  // ===== Round 1 (dense): process ALL tiles directly.
  for(int t = wid; t < num_tiles; t += NW) { WWAC3_PROCESS_TILE(t); }
  __syncthreads();
  for(int d = threadIdx.x; d < dirty_count; d += TPB) {
    auto v = dirty_list[d];
    var_dirty[v] = 0;
    for(int j = vt_off[v]; j < vt_off[v + 1]; ++j) {
      auto t = vt[j];
      if(atomicExch(&in_next[t], 1) == 0) cur[atomicAdd(&nsize, 1)] = t;
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    fsize = nsize;
    tot = num_bytecodes;
    rounds = 1;
    nsize = 0;
    dirty_count = 0;
  }
  __syncthreads();

  // ===== Rounds 2+ (worklist): process only the active frontier of tiles. =====
  while(fsize > 0 && !iprop.is_bot()) {
    // ~32 bytecode activations per frontier tile (stats only; thread-0 owns it).
    if(threadIdx.x == 0) { tot += fsize * 32; rounds++; }

    for(int k = wid; k < fsize; k += NW) {
      int t = cur[k];
      if(lane == 0) in_next[t] = 0;       // allow this tile to be re-queued
      WWAC3_PROCESS_TILE(t);
    }
    __syncthreads();

    for(int d = threadIdx.x; d < dirty_count; d += TPB) {
      auto v = dirty_list[d];
      var_dirty[v] = 0;
      for(int j = vt_off[v]; j < vt_off[v + 1]; ++j) {
        auto t = vt[j];
        if(atomicExch(&in_next[t], 1) == 0) nxt[atomicAdd(&nsize, 1)] = t;
      }
    }
    __syncthreads();

    if(threadIdx.x == 0) {
      fsize = nsize;
      int* tmp = cur;
      cur = nxt;
      nxt = tmp;
      nsize = 0;
      dirty_count = 0;
    }
    __syncthreads();
  }

  // Bot-exit cleanup: restore the persistent `in_next` to all-zero so the next
  // call's precondition holds (mirrors WAC3's bot-exit tail).
  if(iprop.is_bot()) {
    for(int k = threadIdx.x; k < fsize; k += TPB) {
      in_next[cur[k]] = 0;
    }
  }

  if(threadIdx.x == 0) {
    *out_total_deduces += tot;
  }

  #undef WWAC3_PROCESS_TILE
  return rounds;
}

} // namespace lala

#endif // __CUDACC__

#endif // LALA_CORE_WWAC3_FIXPOINT_HPP
