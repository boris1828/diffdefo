#pragma once

// AVX2 + FMA triangular solve of a sparse Cholesky factor for the three coordinates at once (GlobalSolve::ScalarAvx2).
// The right-hand side is N rows of 4 doubles (x, y, z, one unused lane), so one 256-bit FMA updates a whole row and the factor
// is walked once. Hand-written intrinsics: MSVC needs no /arch flag for them, GCC/Clang get a per-function target. Everything here is
// compiled out off x86-64; avx2_fma_available() says whether the running CPU may execute it (construct_pd_lhs falls back to Scalar).

#include "core/diffpd_core.h"

#if defined(_M_X64) || defined(__x86_64__)
#define DIFFPD_AVX2_KERNEL 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define DIFFPD_AVX2_TARGET
#else
#define DIFFPD_AVX2_TARGET __attribute__((target("avx2,fma")))
#endif
#else
#define DIFFPD_AVX2_KERNEL 0
#endif

inline bool avx2_fma_available()
{
#if DIFFPD_AVX2_KERNEL && defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osxsave && avx) || (_xgetbv(0) & 6) != 6) return false; // the OS must save the YMM registers
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
#elif DIFFPD_AVX2_KERNEL
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#if DIFFPD_AVX2_KERNEL
// L: lower Cholesky factor, column major, compressed, diagonal first in each column; dinv[j] = 1 / L_jj (a multiply, not a divide, on the
// critical path). y: N x 4 doubles row-major, P*b on entry, (L L^T)^-1 P*b on exit.
// shortcut: forward pass scatters column by column (no row-major copy of the factor). A row-gather forward pass was within +-20% of it.
DIFFPD_AVX2_TARGET inline void llt_solve_inplace_avx2(const SparseMat& L, const Real* dinv, Real* y)
{
    const Index N = L.cols();
    const int* cp = L.outerIndexPtr(); const int* ri = L.innerIndexPtr(); const Real* val = L.valuePtr();
    for (Index j = 0; j < N; ++j) // L y = b
    {
        const __m256d yj = _mm256_mul_pd(_mm256_loadu_pd(y + 4 * j), _mm256_set1_pd(dinv[j]));
        _mm256_storeu_pd(y + 4 * j, yj);
        for (int p = cp[j] + 1; p < cp[j + 1]; ++p)
        {
            Real* yi = y + 4 * (Index)ri[p];
            _mm256_storeu_pd(yi, _mm256_fnmadd_pd(_mm256_broadcast_sd(val + p), yj, _mm256_loadu_pd(yi)));
        }
    }
    for (Index j = N - 1; j >= 0; --j) // L^T x = y, two accumulators to overlap the FMA latency
    {
        __m256d s0 = _mm256_loadu_pd(y + 4 * j), s1 = _mm256_setzero_pd();
        int p = cp[j] + 1; const int e = cp[j + 1];
        for (; p + 1 < e; p += 2)
        {
            s0 = _mm256_fnmadd_pd(_mm256_broadcast_sd(val + p),     _mm256_loadu_pd(y + 4 * (Index)ri[p]),     s0);
            s1 = _mm256_fnmadd_pd(_mm256_broadcast_sd(val + p + 1), _mm256_loadu_pd(y + 4 * (Index)ri[p + 1]), s1);
        }
        if (p < e) s0 = _mm256_fnmadd_pd(_mm256_broadcast_sd(val + p), _mm256_loadu_pd(y + 4 * (Index)ri[p]), s0);
        _mm256_storeu_pd(y + 4 * j, _mm256_mul_pd(_mm256_add_pd(s0, s1), _mm256_set1_pd(dinv[j])));
    }
}
#endif
