// Copyright 2026 - WAC1 (warp-centric AC1) fixpoint.

#ifndef LALA_CORE_WAC1_FIXPOINT_HPP
#define LALA_CORE_WAC1_FIXPOINT_HPP

#include "logic/logic.hpp"
#include "b.hpp"
#include "fixpoint.hpp"

#ifdef __CUDACC__

namespace lala {

/** WAC1 device fixpoint: warp-centric AC1.
 *
 * Thin, call-compatible wrapper around the ORIGINAL WAC1 composition:
 * `BlockAsynchronousFixpointGPU<true>::fixpoint` driving the `__noinline__`
 * `warp_fixpoint` (both unchanged in `fixpoint.hpp`) — i.e. exactly the code
 * that ran inline in `barebones_dive_and_solve.hpp::propagate` before this
 * file existed. Same signature shape as `wac3_fixpoint`/`wwac3_fixpoint`.
 *
 * IMPORTANT: do NOT re-derive the loop protocol here. A previous version of
 * this file re-implemented the outer rounds (single shared `round_changed`
 * flag, per-thread `is_bot()` in the loop condition) and inlined the body of
 * `warp_fixpoint` (losing its `__noinline__` boundary). By inspection it was
 * semantically equivalent, but on H100/sm_90 release builds it produced
 * UNSOUND results (mzn-challenge-2024 community-detection s12.k3/k4: claimed
 * optimality at 580000 when 588336 is feasible). The plain shared-bool
 * signalling protocol is exactly the code pattern the CUDA compiler may
 * transform once fully inlined (same UB class as the elided plain
 * `in_next`/`var_dirty` clears fixed in `wwac3_fixpoint.hpp`); in particular,
 * every thread evaluating `is_bot()` in the exit condition (instead of
 * thread-0 sampling it into a shared stop slot, as the engine does) risks a
 * divergent `__syncthreads()` if any thread reuses a stale read. Delegating
 * to the originals keeps the battle-tested codegen.
 *
 * @tparam TPB    threads per block (must be a multiple of 32, must match the
 *                kernel launch).
 * @tparam A      IProp-shaped abstract domain: `num_deductions()`,
 *                `load_deduce(int)`, `deduce(ded)`, `is_bot()`.
 *
 * @param iprop             abstract propagator (read+write).
 * @param out_total_deduces accumulator: += (per-warp iteration count) * 32
 *                          summed over all warps — the same statistic the
 *                          pre-extraction code recorded into
 *                          `stats.num_deductions`.
 *
 * @return number of outer fixpoint iterations (engine rounds, >= 1).
 */
template <int TPB, class A>
__device__ int wac1_fixpoint(A& iprop, int* __restrict__ out_total_deduces) {
  static_assert(TPB % 32 == 0, "TPB must be a multiple of 32");
  __shared__ int warp_iterations[TPB/32];
  warp_iterations[threadIdx.x / 32] = 0;
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
  // fp_engine.fixpoint() begins with reset(); barrier(); — that barrier also
  // publishes the warp_iterations zeroing above, exactly as in the original
  // propagate() (which zeroed at the top and relied on the same barrier).
  int fp_iterations = fp_engine.fixpoint(
    iprop.num_deductions(),
    [&](int i){ return warp_fixpoint<TPB>(iprop, i, warp_iterations); },
    [&](){ return iprop.is_bot(); });
  if(threadIdx.x == 0) {
    int tot = 0;
    for(int w = 0; w < TPB/32; ++w) {
      tot += warp_iterations[w] * 32;
    }
    *out_total_deduces += tot;
  }
  return fp_iterations;
}

} // namespace lala

#endif // __CUDACC__

#endif // LALA_CORE_WAC1_FIXPOINT_HPP
