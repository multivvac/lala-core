// Copyright 2026 - AWWAC3 (async streaming warp-tile AC3 worklist) fixpoint.

#ifndef LALA_CORE_AWWAC3_FIXPOINT_HPP
#define LALA_CORE_AWWAC3_FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"

#ifdef __CUDACC__

namespace lala {

namespace impl {
  /** Backoff used by the AWWAC3 spin loops. `__nanosleep` (sm_70+) lets the
   * scheduler deprioritise a spinning lane so the lane it waits on can progress;
   * on older arches it degrades to a bare spin. */
  __device__ __forceinline__ void awwac3_backoff(unsigned ns) {
  #if __CUDA_ARCH__ >= 700
    __nanosleep(ns);
  #else
    (void)ns;
  #endif
  }
} // namespace impl

/** AWWAC3 device fixpoint: ASYNC (streaming) warp-tile AC3 worklist.
 *
 * Same role/interface as `wwac3_fixpoint` (warp-tile event-driven AC3), but the
 * BSP round barriers (rounds 2+) are replaced by a single lock-free MPMC ring
 * drained by PERSISTENT warps — NO per-round `__syncthreads()`. Round 1 stays a
 * dense BSP pass that seeds the ring. This removes the per-round barrier (which
 * blocks every warp for the slowest warp's last dirty emission) that bounds
 * WWAC3 on deep/many-round constraint graphs (in the standalone prototype this
 * was geomean −9.1% over BSP WWAC3, with big wins on round-bound fixtures and
 * small regressions where the lock-free order does extra re-fires).
 *
 * Ring protocol (the validated WSAC3/Stage-B design, now per-TILE): slot value
 * −1 == FREE, >= 0 == a pending TILE id. `q_avail` is a counting semaphore of
 * claimable tiles; `q_head`/`q_tail` are WRAPPING cursors via
 * atomicInc(addr, cap-1) (returns the slot directly — no %cap, no overflow);
 * `q_out` = outstanding work (enqueued minus processed) for termination.
 * PRODUCERS = all lanes (each enqueues the incident tiles of the variables IT
 * narrowed, via the var->tile CSR); CONSUMERS = lane 0 of each warp (claims a
 * tile, `__shfl`-broadcasts it to the warp, all 32 lanes process it). `in_next`
 * dedups tiles, capping ring occupancy at cap=num_tiles, so a producer never
 * needs a slot beyond a consumer mid-claim (deadlock-free); `q_out == 0` is a
 * stable terminal state.
 *
 * @tparam TPB    threads per block (must match the kernel launch).
 * @tparam A      an IProp-shaped abstract domain (same contract as
 *                `wwac3_fixpoint`): `num_deductions()`, `load_deduce(int)`,
 *                `deduce_with_mask(ded)`, `is_bot()`, ded `.x/.y/.z.vid()`.
 *
 * @param iprop       abstract propagator (read+write).
 * @param vt_off      var->TILE CSR offsets, size `num_vars + 1`.
 * @param vt          var->TILE CSR incidence list.
 * @param ring        the MPMC ring (tile ids), size >= `num_tiles` (reuses
 *                    WWAC3's `frontier_a`). The kernel self-inits it to the −1
 *                    sentinel each call, so the caller only ALLOCATES it.
 * @param (unused)    WWAC3's `frontier_b`: a single ring suffices. Kept in the
 *                    signature so AWWAC3 is a drop-in for the WWAC3 call site.
 * @param in_next     tile dedup flag, size >= `num_tiles`. PERSISTENT: all-zero
 *                    on entry; the algorithm keeps it clean (drain clears each
 *                    claimed tile; the bot-exit tail clears any in-flight ones).
 * @param var_dirty   round-1 seed marker, size `num_vars`. PERSISTENT, all-zero
 *                    on entry; round 1 keeps it clean.
 * @param dirty_list  dense list of round-1 dirty vars (scratch), size `num_vars`.
 *
 * @return 1 — AWWAC3 has no discrete rounds (one dense seed pass + a streaming
 *         drain); real work is reported via `out_total_deduces`.
 */
template <int TPB, class A>
__device__ int awwac3_fixpoint(
    A& iprop,
    const int* __restrict__ vt_off,
    const int* __restrict__ vt,
    int* __restrict__ ring,
    int* __restrict__ in_next,
    int* __restrict__ var_dirty,
    int* __restrict__ dirty_list,
    int* __restrict__ out_total_deduces)
{
  // wid / lane are NOT held in registers: threadIdx.x is a free special register,
  // so rematerialising (threadIdx.x>>5) / (threadIdx.x&31) at each use keeps two
  // values off the live set across the persistent drain — relieves the register
  // pressure that spills gpu_barebones_solve (inlines this) at its __launch_bounds.
  __shared__ int num_bytecodes;
  __shared__ int num_tiles;
  __shared__ int cap;
  __shared__ int nsize;            // round-1 seed count
  __shared__ int dirty_count;      // round-1 dirty-var count
  __shared__ int tot;              // round-1 dense activations (stats)
  __shared__ unsigned int q_head;  // next claim slot (wrapping)
  __shared__ unsigned int q_tail;  // next enqueue slot (wrapping)
  __shared__ int q_avail;          // claimable tiles (<= cap)
  __shared__ int q_out;            // outstanding work (termination)
  __shared__ int warp_tile[TPB / 32];  // per-warp claimed tile, broadcast via shared (vs __shfl)

  if(threadIdx.x == 0) {
    num_bytecodes = iprop.num_deductions();
    num_tiles = (num_bytecodes + 31) / 32;
    cap = num_tiles;
    nsize = 0; dirty_count = 0; tot = 0;
  }
  __syncthreads();

  // Self-init the ring to the −1 FREE sentinel (independent of round-1 work).
  for(int s = threadIdx.x; s < cap; s += TPB) ring[s] = -1;

  // ===== Round 1 (dense BSP): warp strides over all tiles, dirties narrowed vars.
  for(int tw = (threadIdx.x >> 5); tw < num_tiles; tw += (TPB / 32)) {
    int bidx = tw*32 + (threadIdx.x & 31);
    if(bidx < num_bytecodes) {
      auto ded = iprop.load_deduce(bidx);
      int mask = 0, m;
      while((m = iprop.deduce_with_mask(ded)) && !iprop.is_bot()) { mask |= m; }
      if((mask & 1) && atomicExch(&var_dirty[ded.x.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.x.vid();
      if((mask & 2) && atomicExch(&var_dirty[ded.y.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.y.vid();
      if((mask & 4) && atomicExch(&var_dirty[ded.z.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.z.vid();
    }
  }
  __syncthreads();

  // Seed the ring with the incident tiles of round-1 narrowed vars.
  for(int d = threadIdx.x; d < dirty_count; d += TPB) {
    int v = dirty_list[d]; var_dirty[v] = 0;
    for(int j = vt_off[v]; j < vt_off[v + 1]; ++j) {
      int t = vt[j];
      if(atomicExch(&in_next[t], 1) == 0) ring[atomicAdd(&nsize, 1)] = t;  // seed over −1
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    q_head = 0u;
    q_tail = (nsize < cap) ? (unsigned)nsize : 0u;   // wrap if the seed filled the ring
    q_avail = nsize; q_out = nsize;
    tot = num_bytecodes;
  }
  __syncthreads();
  
  // ===== Async drain: persistent warps, NO round barriers. =====
  while(*(volatile int*)&q_out != 0 && !iprop.is_bot()) {
    if((threadIdx.x & 31) == 0) {                      // only lane 0 consumes
      int t = -1;
      if(*(volatile int*)&q_avail > 0) {               // test-before-atomic
        if(atomicSub(&q_avail, 1) > 0) {               // claimed a real tile
          unsigned h = atomicInc(&q_head, (unsigned)cap - 1u);
          // Wait for the producer to publish slot h — but BAIL on bot: once bot is
          // detected the producer responsible for h may have already exited the
          // drain (top-of-loop !is_bot()), so this would otherwise spin forever.
          while((t = *(volatile int*)&ring[h]) < 0) {
            if(iprop.is_bot()) break;
            impl::awwac3_backoff(20);
          }
          if(t >= 0) {                                 // got a real tile (not a bot bailout)
            *(volatile int*)&ring[h] = -1;             // free the slot
            atomicExch(&in_next[t], 0);                // allow re-enqueue
          }
          __threadfence_block();
        } else {
          atomicAdd(&q_avail, 1);                      // lost race to empty; undo
          t = -1;
        }
      }
      warp_tile[threadIdx.x >> 5] = t;                 // publish claim to the warp via shared
    }
    __syncwarp();                                      // make lane 0's claim visible warp-wide
    if(warp_tile[threadIdx.x >> 5] < 0) { impl::awwac3_backoff(40); continue; }

    // All 32 lanes process the claimed tile; each lane enqueues the incident
    // tiles of the variables IT narrowed.
    int bidx = warp_tile[threadIdx.x >> 5]*32 + (threadIdx.x & 31);
    if(bidx < num_bytecodes) {
      auto ded = iprop.load_deduce(bidx);
      int mask = 0, m;
      while((m = iprop.deduce_with_mask(ded)) && !iprop.is_bot()) { mask |= m; }
      // #pragma unroll
      for(int a = 0; a < 3; ++a) {
        if(((mask >> a) & 1) == 0) { continue; }
        int v = (a == 0) ? ded.x.vid() : (a == 1) ? ded.y.vid() : ded.z.vid();
        for(int j = vt_off[v]; j < vt_off[v + 1]; ++j) {
          int tt = vt[j];
          if(atomicExch(&in_next[tt], 1) == 0) {       // tile-level dedup
            atomicAdd(&q_out, 1);                       // count BEFORE publishing
            unsigned slot = atomicInc(&q_tail, (unsigned)cap - 1u);
            // Wait for the slot to free — but BAIL on bot (the consumer that frees
            // it may have already left the drain), else this spins forever.
            while(*(volatile int*)&ring[slot] != -1 && !iprop.is_bot()) { impl::awwac3_backoff(20); }
            if(*(volatile int*)&ring[slot] == -1) {     // got a free slot (not a bot bailout)
              *(volatile int*)&ring[slot] = tt;         // publish
              __threadfence_block();
              atomicAdd(&q_avail, 1);                   // signal claimable
            }
          }
        }
      }
    }
    __syncwarp();
    if((threadIdx.x & 31) == 0) atomicSub(&q_out, 1);   // this tile fully processed
  }
  __syncthreads();   // ALL warps finished the barrier-free drain: makes every
                     // store[v] narrowing + is_at_bot globally visible/consistent
                     // before the caller reads is_bot()/ask()/the store. (WSAC3 had
                     // this; the standalone prototype didn't need it — the kernel
                     // ended there and the host sync provided the barrier.)

  // Bot-exit cleanup: a bot detected mid-drain leaves threads having bailed out of
  // the spin-waits, so some tiles can be reserved-but-unpublished (in_next set, not
  // in the ring) — the old "clear only in-ring tiles" cleanup would miss them and
  // corrupt the next call. Robustly reset the persistent state: clear in_next for
  // ALL tiles and the whole ring, restoring the all-zero / all-(-1) precondition.
  if(iprop.is_bot()) {
    for(int s = threadIdx.x; s < num_tiles; s += TPB) in_next[s] = 0;
    for(int s = threadIdx.x; s < cap; s += TPB) ring[s] = -1;
  }

  if(threadIdx.x == 0) {
    *out_total_deduces += tot;
  }
  return 1;  // no discrete rounds in the streaming model.
}

} // namespace lala

#endif // __CUDACC__

#endif // LALA_CORE_AWWAC3_FIXPOINT_HPP
