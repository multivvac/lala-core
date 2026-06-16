// Copyright 2026 - WAC3 (event-driven AC3 worklist) fixpoint.

#ifndef LALA_CORE_WAC3_FIXPOINT_HPP
#define LALA_CORE_WAC3_FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"

#ifdef __CUDACC__

namespace lala {

/** WAC3 device fixpoint: event-driven AC3 worklist.
 *
 * Runs to fixpoint (or bot) on `iprop`. Returns the number of frontier
 * rounds performed (analogous to WAC1's `fp_iterations`).
 *
 * @tparam TPB    threads per block (must match the kernel launch).
 * @tparam A      an IProp-shaped abstract domain: must provide
 *                `load_deduce(int)`, `deduce(ded)`, `operator[](int) -> Itv`,
 *                `is_bot()`, `num_deductions()`, and `vars()`. PIR and PC
 *                both satisfy this.
 *
 * @param iprop         abstract propagator (read+write).
 * @param adj_off       var->bc CSR offsets, size `num_vars + 1`.
 *                      Built once in `UnifiedData`.
 * @param adj_bc        var->bc CSR incidence list, size `3*num_bytecodes`.
 * @param frontier_a    worklist buffer A, size `num_bytecodes`.
 * @param frontier_b    worklist buffer B, size `num_bytecodes`.
 * @param in_next       dedup flag, size `num_bytecodes`. PERSISTENT: must
 *                      be all-zero on entry; the algorithm restores it
 *                      to all-zero on exit (including the bot-exit cleanup
 *                      tail at the end).
 * @param var_dirty     in-list marker, size `num_vars`. PERSISTENT, must
 *                      be all-zero on entry; algorithm keeps it clean.
 * @param dirty_list    dense list of dirty vars (scratch), size `num_vars`.
 *
 * @return number of rounds performed (>=1; round 1 is a dense process-all
 *         pass, rounds 2+ process only the frontier).
 */
template <int TPB, class A>
__device__ int wac3_fixpoint(
    A& iprop,
    const int* __restrict__ adj_off,
    const int* __restrict__ adj_bc,
    int* __restrict__ frontier_a,
    int* __restrict__ frontier_b,
    int* __restrict__ in_next,
    int* __restrict__ var_dirty,
    int* __restrict__ dirty_list,
    int* __restrict__ out_total_deduces)
{
  __shared__ int  fsize;
  __shared__ int  nsize;
  __shared__ int  dirty_count;
  __shared__ int  rounds;
  __shared__ int  tot;
  __shared__ int* cur;
  __shared__ int* nxt;
  __shared__ int num_bytecodes;

  if(threadIdx.x == 0) {
    rounds = 0; tot = 0; nsize = 0; dirty_count = 0;
    cur = frontier_a;
    nxt = frontier_b;
    num_bytecodes = iprop.num_deductions();
  }
  __syncthreads();

  // ===== Round 1 (dense): process ALL bytecodes directly
  for(int k = threadIdx.x; k < num_bytecodes; k += TPB) {
    auto ded = iprop.load_deduce(k);
    int mask = 0;
    int m;
    while((m = iprop.deduce_with_mask(ded)) && !iprop.is_bot()) { mask |= m; }
    if((mask & 1) && atomicExch(&var_dirty[ded.x.vid()], 1) == 0)
      dirty_list[atomicAdd(&dirty_count, 1)] = ded.x.vid();
    if((mask & 2) && atomicExch(&var_dirty[ded.y.vid()], 1) == 0)
      dirty_list[atomicAdd(&dirty_count, 1)] = ded.y.vid();
    if((mask & 4) && atomicExch(&var_dirty[ded.z.vid()], 1) == 0)
      dirty_list[atomicAdd(&dirty_count, 1)] = ded.z.vid();
  }
  __syncthreads();
  for(int d = threadIdx.x; d < dirty_count; d += TPB) {
    auto v = dirty_list[d];
    var_dirty[v] = 0;
    for(int j = adj_off[v]; j < adj_off[v + 1]; ++j) {
      auto b = adj_bc[j];
      if(atomicExch(&in_next[b], 1) == 0) cur[atomicAdd(&nsize, 1)] = b;
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

  // ===== Rounds 2+ (worklist): process only the active frontier. =====
  while(fsize > 0 && !iprop.is_bot()) {
    if(threadIdx.x == 0) { tot += fsize; rounds++; }

    for(int k = threadIdx.x; k < fsize; k += TPB) {
      int idx = cur[k];
      in_next[idx] = 0;                 // allow this bytecode to be re-queued
      auto ded = iprop.load_deduce(idx);
      int mask = 0;
      int m;
      while((m = iprop.deduce_with_mask(ded)) && !iprop.is_bot()) { mask |= m; }
      if((mask & 1) && atomicExch(&var_dirty[ded.x.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.x.vid();
      if((mask & 2) && atomicExch(&var_dirty[ded.y.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.y.vid();
      if((mask & 4) && atomicExch(&var_dirty[ded.z.vid()], 1) == 0)
        dirty_list[atomicAdd(&dirty_count, 1)] = ded.z.vid();
    }
    __syncthreads();

    for(int d = threadIdx.x; d < dirty_count; d += TPB) {
      auto v = dirty_list[d];
      var_dirty[v] = 0;
      for(int j = adj_off[v]; j < adj_off[v + 1]; ++j) {
        auto b = adj_bc[j];
        if(atomicExch(&in_next[b], 1) == 0) nxt[atomicAdd(&nsize, 1)] = b;
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

  if(iprop.is_bot()) {
    for(int k = threadIdx.x; k < fsize; k += TPB) {
      in_next[cur[k]] = 0;
    }
  }

  if(threadIdx.x == 0) {
    *out_total_deduces += tot;
  }
  return rounds;
}

} // namespace lala

#endif // __CUDACC__

#endif // LALA_CORE_WAC3_FIXPOINT_HPP
