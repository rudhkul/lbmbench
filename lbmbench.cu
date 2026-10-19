// =============================================================================
//  lbmbench.cu  --  Controlled micro-benchmark of core lattice Boltzmann GPU
//                   implementation choices (companion code of the review
//                   "The lattice Boltzmann method on GPUs: a critical review of
//                   the core algorithm").
//
//  Factors that can be varied independently (everything else held fixed):
//     velocity set     : D3Q19, D3Q27
//     propagation      : abpull | abpush | aa | ep (esoteric pull) | mr (moment
//                        representation, regularised, two moment lattices)
//     memory layout    : soa | aos
//     precision        : fp64 | fp32 | mixed (fp64 arithmetic, fp32 storage)
//                        | fp16s (IEEE binary16 storage, fp32 arithmetic)
//                        | fp16c (custom 1-4-11 storage, fp32 arithmetic)
//     collision        : bgk | trt | reg (second-order projective regularised)
//     DDF shifting     : --shift 1 (store f_i - w_i) | --shift 0 (store f_i)
//
//  Modes:
//     info   : print device properties
//     bw     : measure attainable device bandwidth (float4 copy kernel)
//     perf   : periodic box, timed bulk kernel -> MLUPS, effective bandwidth
//     tgv    : 2D Taylor-Green vortex (Nz = 1) -> L2 velocity error vs exact NS
//     verify : cross-check all propagation patterns against abpull (bitwise
//              equality expected), and mr against abpull+reg
//
//  Results are appended as one CSV line per run to --csv (default results.csv).
//
//  Build : make            (default ARCH=sm_120 for RTX 50-series; CUDA >= 12.8)
//  Usage : ./lbmbench --mode perf --q 19 --stream ep --prec fp32 --coll bgk --n 256
//  Author: companion code generated for A. Kulkarni's GPU-LBM review (2026).
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <type_traits>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {            \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
    __LINE__); exit(EXIT_FAILURE);} } while (0)

// ----------------------------------------------------------------------------
// Velocity sets. Ordering: 0 = rest; for odd i, i+1 is the opposite of i.
// This pairing is required by the esoteric-pull and TRT implementations.
// ----------------------------------------------------------------------------
struct D3Q19 {
  static constexpr int Q = 19;
  static constexpr const char* name = "D3Q19";
  __host__ __device__ static constexpr int cx(int i) {
    constexpr int a[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
    return a[i]; }
  __host__ __device__ static constexpr int cy(int i) {
    constexpr int a[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
    return a[i]; }
  __host__ __device__ static constexpr int cz(int i) {
    constexpr int a[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};
    return a[i]; }
  __host__ __device__ static constexpr double w(int i) {
    return i == 0 ? 1.0/3.0 : (i <= 6 ? 1.0/18.0 : 1.0/36.0); }
};

struct D3Q27 {
  static constexpr int Q = 27;
  static constexpr const char* name = "D3Q27";
  __host__ __device__ static constexpr int cx(int i) {
    constexpr int a[27] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,
                           1,-1, 1,-1, 1,-1,-1, 1};
    return a[i]; }
  __host__ __device__ static constexpr int cy(int i) {
    constexpr int a[27] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1,
                           1,-1, 1,-1,-1, 1, 1,-1};
    return a[i]; }
  __host__ __device__ static constexpr int cz(int i) {
    constexpr int a[27] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1,
                           1,-1,-1, 1, 1,-1, 1,-1};
    return a[i]; }
  __host__ __device__ static constexpr double w(int i) {
    return i == 0 ? 8.0/27.0 : (i <= 6 ? 2.0/27.0 : (i <= 18 ? 1.0/54.0 : 1.0/216.0)); }
};

__host__ __device__ constexpr int opp(int i) { return i == 0 ? 0 : ((i & 1) ? i + 1 : i - 1); }

// ----------------------------------------------------------------------------
// Number formats.  A = arithmetic type, S = storage type.
// ----------------------------------------------------------------------------
struct P64 {
  using A = double; using S = double; static constexpr const char* name = "fp64";
  __device__ static A ld(const S* p, size_t i) { return p[i]; }
  __device__ static void st(S* p, size_t i, A v) { p[i] = v; }
};
struct P32 {
  using A = float; using S = float; static constexpr const char* name = "fp32";
  __device__ static A ld(const S* p, size_t i) { return p[i]; }
  __device__ static void st(S* p, size_t i, A v) { p[i] = v; }
};
struct PMX {  // fp64 arithmetic, fp32 storage
  using A = double; using S = float; static constexpr const char* name = "mixed";
  __device__ static A ld(const S* p, size_t i) { return (double)p[i]; }
  __device__ static void st(S* p, size_t i, A v) { p[i] = (float)v; }
};
// IEEE binary16 storage. Values are pre-scaled by 2^15 so that the small
// (shifted) populations fall into the normal range of binary16; |f| < 2 is
// representable, which covers both shifted and unshifted populations.
struct P16S {
  using A = float; using S = unsigned short; static constexpr const char* name = "fp16s";
  __device__ static A ld(const S* p, size_t i) {
    return __half2float(__ushort_as_half(p[i])) * 3.0517578125e-05f; }   // 2^-15
  __device__ static void st(S* p, size_t i, A v) {
    p[i] = __half_as_ushort(__float2half_rn(v * 32768.0f)); }
};
// Custom 16-bit format, 1 sign / 4 exponent / 11 mantissa bits (in the spirit of
// FP16C of Lehmann et al., PRE 2022, but our own definition):
//   e >= 1 : (-1)^s 2^(e-15) (1 + m/2048)      range [2^-14, 2)
//   e == 0 : (-1)^s 2^-25 m                     subnormal, step 2^-25
// Round-to-nearest-even, saturating at the largest finite value.
struct P16C {
  using A = float; using S = unsigned short; static constexpr const char* name = "fp16c";
  __device__ static A ld(const S* p, size_t i) {
    const unsigned h = p[i]; const unsigned mag = h & 0x7FFFu;
    const unsigned e = mag >> 11, m = mag & 0x7FFu;
    float v = (e == 0u) ? (float)m * 2.98023223876953125e-08f            // 2^-25
                        : __uint_as_float(((e + 112u) << 23) | (m << 12));
    return (h & 0x8000u) ? -v : v; }
  __device__ static void st(S* p, size_t i, A v) {
    const unsigned b = __float_as_uint(v); const unsigned s = (b >> 16) & 0x8000u;
    const unsigned a = b & 0x7FFFFFFFu; unsigned h;
    if (a >= (113u << 23)) {                       // normal in target format
      unsigned r = a - (112u << 23);
      r += 0x7FFu + ((r >> 12) & 1u);              // round to nearest even
      h = r >> 12; if (h > 0x7FFFu) h = 0x7FFFu;   // saturate
    } else {                                       // subnormal / zero
      h = __float2uint_rn(__uint_as_float(a) * 33554432.0f);  // * 2^25, exact
      if (h > 0x7FFFu) h = 0x7FFFu;
    }
    p[i] = (S)(s | h); }
};

// ----------------------------------------------------------------------------
enum Coll { BGK = 0, TRT = 1, REG = 2 };
enum Stream { AB_PULL = 0, AB_PUSH = 1, AA = 2, EP = 3, MR = 4 };
enum Layout { SOA = 0, AOS = 1 };

struct Prm {
  int nx, ny, nz; unsigned n;   // number of nodes
  double omega;                 // 1/tau
  double omegam;                // TRT antisymmetric rate
  int shift;                    // 0/1 DDF shifting
  double U;                     // amplitude of the analytic initial field
};

// Analytic initial field used when no host field is supplied (perf mode):
// 2D Taylor-Green vortex in the x-y plane, uniform in z.
__device__ __forceinline__ void field_tgv(unsigned m, const Prm& p, double& r, double& u, double& v, double& w) {
  const int x = m % p.nx, y = (m / p.nx) % p.ny;
  const double X = 6.283185307179586 * (x + 0.5) / p.nx, Y = 6.283185307179586 * (y + 0.5) / p.ny;
  r = 1.0 - 0.75 * p.U * p.U * (cos(2 * X) + cos(2 * Y));
  u = -p.U * cos(X) * sin(Y); v = p.U * sin(X) * cos(Y); w = 0.0;
}

template <int LA, int Q>
__device__ __forceinline__ size_t fidx(unsigned node, int i, unsigned n) {
  if constexpr (LA == SOA) return (size_t)i * n + node;
  else return (size_t)node * Q + i;
}

__device__ __forceinline__ unsigned nbr(int x, int y, int z, int dx, int dy, int dz, const Prm& p) {
  int xx = x + dx, yy = y + dy, zz = z + dz;
  xx = xx < 0 ? p.nx - 1 : (xx >= p.nx ? 0 : xx);
  yy = yy < 0 ? p.ny - 1 : (yy >= p.ny ? 0 : yy);
  zz = zz < 0 ? p.nz - 1 : (zz >= p.nz ? 0 : zz);
  return (unsigned)xx + (unsigned)p.nx * ((unsigned)yy + (unsigned)p.ny * (unsigned)zz);
}

// Shifted equilibrium g_i^eq = f_i^eq - s w_i (second-order Hermite).
template <class V, class A>
__device__ __forceinline__ A geq(int i, A rho, A ux, A uy, A uz, A s) {
  const A cu = (A)V::cx(i) * ux + (A)V::cy(i) * uy + (A)V::cz(i) * uz;
  const A u2 = ux * ux + uy * uy + uz * uz;
  return (A)V::w(i) * ((rho - s) + rho * ((A)3 * cu + (A)4.5 * cu * cu - (A)1.5 * u2));
}

// Macroscopic moments + collision on shifted populations g[] (in place).
template <class V, class A, int COLL>
__device__ __forceinline__ void collide(A* g, const Prm& p, A& rho, A& ux, A& uy, A& uz) {
  constexpr int Q = V::Q;
  const A s = (A)p.shift;
  A r = 0, jx = 0, jy = 0, jz = 0;
#pragma unroll
  for (int i = 0; i < Q; ++i) {
    r += g[i]; jx += (A)V::cx(i) * g[i]; jy += (A)V::cy(i) * g[i]; jz += (A)V::cz(i) * g[i];
  }
  rho = r + s;
  const A ir = (A)1 / rho; ux = jx * ir; uy = jy * ir; uz = jz * ir;
  const A om = (A)p.omega;
  if constexpr (COLL == BGK) {
#pragma unroll
    for (int i = 0; i < Q; ++i) g[i] += om * (geq<V, A>(i, rho, ux, uy, uz, s) - g[i]);
  } else if constexpr (COLL == TRT) {
    const A omm = (A)p.omegam;
    g[0] += om * (geq<V, A>(0, rho, ux, uy, uz, s) - g[0]);
#pragma unroll
    for (int i = 1; i < Q; i += 2) {
      const A ea = geq<V, A>(i, rho, ux, uy, uz, s), eb = geq<V, A>(i + 1, rho, ux, uy, uz, s);
      const A gp = (A)0.5 * (g[i] + g[i + 1]), gm = (A)0.5 * (g[i] - g[i + 1]);
      const A ep = (A)0.5 * (ea + eb), em = (A)0.5 * (ea - eb);
      const A dp = om * (gp - ep), dm = omm * (gm - em);
      g[i] -= dp + dm; g[i + 1] -= dp - dm;
    }
  } else {  // REG: g_i = g_i^eq + (1-omega) w_i/(2 cs^4) Q_i : Pi^neq
    A pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
    A e[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      e[i] = geq<V, A>(i, rho, ux, uy, uz, s);
      const A d = g[i] - e[i];
      const A cx = (A)V::cx(i), cy = (A)V::cy(i), cz = (A)V::cz(i);
      pxx += cx * cx * d; pyy += cy * cy * d; pzz += cz * cz * d;
      pxy += cx * cy * d; pxz += cx * cz * d; pyz += cy * cz * d;
    }
    const A tr = (pxx + pyy + pzz) * (A)(1.0 / 3.0);
    const A fac = ((A)1 - om) * (A)4.5;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      const A cx = (A)V::cx(i), cy = (A)V::cy(i), cz = (A)V::cz(i);
      const A qp = cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                 + (A)2 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz) - tr;
      g[i] = e[i] + fac * (A)V::w(i) * qp;
    }
  }
}

// ----------------------------------------------------------------------------
// Population-based fused stream-collide kernel (one thread per node).
//   AB_PULL : read f_A(x - c_i, i), write f_B(x, i)
//   AB_PUSH : read f_A(x, i),       write f_B(x + c_i, i)
//   AA      : even: read f(x,i), write f(x,opp i);
//             odd : read f(x - c_i, opp i), write f(x + c_i, i)
//   EP      : esoteric pull (Lehmann 2022), single lattice, parity-dependent.
// ----------------------------------------------------------------------------
// The parity of AA / EP is a template parameter (one kernel per parity), which
// keeps register use low; compiling both parities into one kernel with a
// run-time branch raised D3Q27 Esoteric Pull to ~185-230 registers per thread.
template <class V, class P, int COLL, int ST, int LA, int PAR>
__global__ void __launch_bounds__(256)
k_step(typename P::S* __restrict__ fa, typename P::S* __restrict__ fb, const Prm p,
       double* __restrict__ macro) {
  using A = typename P::A; constexpr int Q = V::Q;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= p.n) return;
  const int x = n % p.nx, y = (n / p.nx) % p.ny, z = n / (p.nx * p.ny);
  A g[Q];
  // ---------------- load ----------------
  if constexpr (ST == AB_PULL) {
#pragma unroll
    for (int i = 0; i < Q; ++i)
      g[i] = P::ld(fa, fidx<LA, Q>(nbr(x, y, z, -V::cx(i), -V::cy(i), -V::cz(i), p), i, p.n));
  } else if constexpr (ST == AB_PUSH) {
#pragma unroll
    for (int i = 0; i < Q; ++i) g[i] = P::ld(fa, fidx<LA, Q>(n, i, p.n));
  } else if constexpr (ST == AA) {
    if constexpr (PAR == 0) {
#pragma unroll
      for (int i = 0; i < Q; ++i) g[i] = P::ld(fa, fidx<LA, Q>(n, i, p.n));
    } else {
#pragma unroll
      for (int i = 0; i < Q; ++i)
        g[i] = P::ld(fa, fidx<LA, Q>(nbr(x, y, z, -V::cx(i), -V::cy(i), -V::cz(i), p), opp(i), p.n));
    }
  } else {  // EP
    g[0] = P::ld(fa, fidx<LA, Q>(n, 0, p.n));
#pragma unroll
    for (int i = 1; i < Q; i += 2) {
      const unsigned j = nbr(x, y, z, V::cx(i), V::cy(i), V::cz(i), p);
      if constexpr (PAR == 0) { g[i] = P::ld(fa, fidx<LA, Q>(n, i + 1, p.n)); g[i + 1] = P::ld(fa, fidx<LA, Q>(j, i, p.n)); }
      else             { g[i] = P::ld(fa, fidx<LA, Q>(n, i, p.n));     g[i + 1] = P::ld(fa, fidx<LA, Q>(j, i + 1, p.n)); }
    }
  }
  // ---------------- collide ----------------
  A rho, ux, uy, uz;
  collide<V, A, COLL>(g, p, rho, ux, uy, uz);
  if (macro) {
    macro[n] = (double)rho; macro[(size_t)p.n + n] = (double)ux;
    macro[2 * (size_t)p.n + n] = (double)uy; macro[3 * (size_t)p.n + n] = (double)uz;
  }
  // ---------------- store ----------------
  if constexpr (ST == AB_PULL) {
#pragma unroll
    for (int i = 0; i < Q; ++i) P::st(fb, fidx<LA, Q>(n, i, p.n), g[i]);
  } else if constexpr (ST == AB_PUSH) {
#pragma unroll
    for (int i = 0; i < Q; ++i)
      P::st(fb, fidx<LA, Q>(nbr(x, y, z, V::cx(i), V::cy(i), V::cz(i), p), i, p.n), g[i]);
  } else if constexpr (ST == AA) {
    if constexpr (PAR == 0) {
#pragma unroll
      for (int i = 0; i < Q; ++i) P::st(fa, fidx<LA, Q>(n, opp(i), p.n), g[i]);
    } else {
#pragma unroll
      for (int i = 0; i < Q; ++i)
        P::st(fa, fidx<LA, Q>(nbr(x, y, z, V::cx(i), V::cy(i), V::cz(i), p), i, p.n), g[i]);
    }
  } else {  // EP
    P::st(fa, fidx<LA, Q>(n, 0, p.n), g[0]);
#pragma unroll
    for (int i = 1; i < Q; i += 2) {
      const unsigned j = nbr(x, y, z, V::cx(i), V::cy(i), V::cz(i), p);
      if constexpr (PAR == 0) { P::st(fa, fidx<LA, Q>(j, i, p.n), g[i]);     P::st(fa, fidx<LA, Q>(n, i + 1, p.n), g[i + 1]); }
      else             { P::st(fa, fidx<LA, Q>(j, i + 1, p.n), g[i]); P::st(fa, fidx<LA, Q>(n, i, p.n), g[i + 1]); }
    }
  }
}

// Initialisation: the equilibrium of the initial field is interpreted as the
// post-collision state at t = -1 (so every pattern starts from bitwise
// identical pre-collision populations at its first step).
template <class V, class P, int ST, int LA>
__global__ void k_init(typename P::S* __restrict__ fa, const Prm p, const double* __restrict__ m0) {
  using A = typename P::A; constexpr int Q = V::Q;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= p.n) return;
  const int x = n % p.nx, y = (n / p.nx) % p.ny, z = n / (p.nx * p.ny);
  const A s = (A)p.shift;
  auto eqat = [&](unsigned m, int i) -> A {
    double r, u, v, w;
    if (m0) { r = m0[m]; u = m0[(size_t)p.n + m]; v = m0[2 * (size_t)p.n + m]; w = m0[3 * (size_t)p.n + m]; }
    else field_tgv(m, p, r, u, v, w);
    return geq<V, A>(i, (A)r, (A)u, (A)v, (A)w, s);
  };
  if constexpr (ST == AB_PULL) {
    for (int i = 0; i < Q; ++i) P::st(fa, fidx<LA, Q>(n, i, p.n), eqat(n, i));
  } else if constexpr (ST == AB_PUSH || ST == AA) {
    for (int i = 0; i < Q; ++i)
      P::st(fa, fidx<LA, Q>(n, i, p.n), eqat(nbr(x, y, z, -V::cx(i), -V::cy(i), -V::cz(i), p), i));
  } else {  // EP: "odd-parity store" of feq(n)
    P::st(fa, fidx<LA, Q>(n, 0, p.n), eqat(n, 0));
    for (int i = 1; i < Q; i += 2) {
      const unsigned j = nbr(x, y, z, V::cx(i), V::cy(i), V::cz(i), p);
      P::st(fa, fidx<LA, Q>(j, i + 1, p.n), eqat(n, i));
      P::st(fa, fidx<LA, Q>(n, i, p.n), eqat(n, i + 1));
    }
  }
}

// ----------------------------------------------------------------------------
// Moment representation (Vardhan et al. SC'19 style, regularised BGK):
// store M = (rho, ux, uy, uz, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) per node in SoA,
// two moment lattices (A/B). Each step reconstructs the post-collision
// population f_i(x - c_i) from the neighbour's moments, then accumulates the
// new moments at x. With shifting, rho-s and Pii-s/3 are stored.
// ----------------------------------------------------------------------------
constexpr int NM = 10;
template <class V, class A>
__device__ __forceinline__ A recon(int i, const A* M, const Prm& p) {
  const A s = (A)p.shift, cs2 = (A)(1.0 / 3.0);
  const A rho = M[0] + s, ux = M[1], uy = M[2], uz = M[3];
  // second-order Hermite coefficient A2 = Pi - rho cs2 I (pre-collision)
  const A axx = M[4] + s * cs2 - rho * cs2, ayy = M[5] + s * cs2 - rho * cs2,
          azz = M[6] + s * cs2 - rho * cs2, axy = M[7], axz = M[8], ayz = M[9];
  const A om = (A)p.omega, k = (A)1 - om;
  // post-collision: A2' = rho u u + (1-omega)(A2 - rho u u)
  const A bxx = rho * ux * ux + k * (axx - rho * ux * ux);
  const A byy = rho * uy * uy + k * (ayy - rho * uy * uy);
  const A bzz = rho * uz * uz + k * (azz - rho * uz * uz);
  const A bxy = rho * ux * uy + k * (axy - rho * ux * uy);
  const A bxz = rho * ux * uz + k * (axz - rho * ux * uz);
  const A byz = rho * uy * uz + k * (ayz - rho * uy * uz);
  const A cx = (A)V::cx(i), cy = (A)V::cy(i), cz = (A)V::cz(i);
  const A cu = cx * ux + cy * uy + cz * uz;
  const A qb = cx * cx * bxx + cy * cy * byy + cz * cz * bzz
             + (A)2 * (cx * cy * bxy + cx * cz * bxz + cy * cz * byz) - cs2 * (bxx + byy + bzz);
  return (A)V::w(i) * ((rho - s) + (A)3 * rho * cu + (A)4.5 * qb);
}

template <class V, class P>
__global__ void __launch_bounds__(256)
k_mr(const typename P::S* __restrict__ ma, typename P::S* __restrict__ mb, const Prm p,
     double* __restrict__ macro) {
  using A = typename P::A; constexpr int Q = V::Q;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= p.n) return;
  const int x = n % p.nx, y = (n / p.nx) % p.ny, z = n / (p.nx * p.ny);
  A r = 0, jx = 0, jy = 0, jz = 0, pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
#pragma unroll
  for (int i = 0; i < Q; ++i) {
    const unsigned m = nbr(x, y, z, -V::cx(i), -V::cy(i), -V::cz(i), p);
    A M[NM];
#pragma unroll
    for (int k = 0; k < NM; ++k) M[k] = P::ld(ma, (size_t)k * p.n + m);
    const A g = recon<V, A>(i, M, p);
    const A cx = (A)V::cx(i), cy = (A)V::cy(i), cz = (A)V::cz(i);
    r += g; jx += cx * g; jy += cy * g; jz += cz * g;
    pxx += cx * cx * g; pyy += cy * cy * g; pzz += cz * cz * g;
    pxy += cx * cy * g; pxz += cx * cz * g; pyz += cy * cz * g;
  }
  const A s = (A)p.shift; const A rho = r + s; const A ir = (A)1 / rho;
  const A ux = jx * ir, uy = jy * ir, uz = jz * ir;
  // store shifted moments: rho - s, u, Pi - s cs2 I  (sum_i g_i c c = Pi - s cs2 I)
  P::st(mb, n, r); P::st(mb, (size_t)p.n + n, ux); P::st(mb, 2 * (size_t)p.n + n, uy);
  P::st(mb, 3 * (size_t)p.n + n, uz);
  P::st(mb, 4 * (size_t)p.n + n, pxx); P::st(mb, 5 * (size_t)p.n + n, pyy);
  P::st(mb, 6 * (size_t)p.n + n, pzz); P::st(mb, 7 * (size_t)p.n + n, pxy);
  P::st(mb, 8 * (size_t)p.n + n, pxz); P::st(mb, 9 * (size_t)p.n + n, pyz);
  if (macro) {
    macro[n] = (double)rho; macro[(size_t)p.n + n] = (double)ux;
    macro[2 * (size_t)p.n + n] = (double)uy; macro[3 * (size_t)p.n + n] = (double)uz;
  }
}

template <class P>
__global__ void k_init_mr(typename P::S* __restrict__ ma, const Prm p, const double* __restrict__ m0) {
  using A = typename P::A;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= p.n) return;
  const A s = (A)p.shift, cs2 = (A)(1.0 / 3.0);
  double r0, u0, v0, w0;
  if (m0) { r0 = m0[n]; u0 = m0[(size_t)p.n + n]; v0 = m0[2 * (size_t)p.n + n]; w0 = m0[3 * (size_t)p.n + n]; }
  else field_tgv(n, p, r0, u0, v0, w0);
  const A rho = (A)r0, ux = (A)u0, uy = (A)v0, uz = (A)w0;
  // equilibrium moments: Pi = rho cs2 I + rho u u; stored shifted
  P::st(ma, n, rho - s); P::st(ma, (size_t)p.n + n, ux); P::st(ma, 2 * (size_t)p.n + n, uy);
  P::st(ma, 3 * (size_t)p.n + n, uz);
  P::st(ma, 4 * (size_t)p.n + n, (rho - s) * cs2 + rho * ux * ux);
  P::st(ma, 5 * (size_t)p.n + n, (rho - s) * cs2 + rho * uy * uy);
  P::st(ma, 6 * (size_t)p.n + n, (rho - s) * cs2 + rho * uz * uz);
  P::st(ma, 7 * (size_t)p.n + n, rho * ux * uy); P::st(ma, 8 * (size_t)p.n + n, rho * ux * uz);
  P::st(ma, 9 * (size_t)p.n + n, rho * uy * uz);
}

// ----------------------------------------------------------------------------
// Bandwidth reference: float4 copy
// ----------------------------------------------------------------------------
__global__ void k_copy(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = a[i];
}

// ============================================================================
// Host side
// ============================================================================
struct Cfg {
  std::string mode = "perf", stream = "abpull", prec = "fp32", coll = "bgk", layout = "soa";
  std::string csv = "results.csv", tag = "";
  int q = 19, nx = 128, ny = 128, nz = 128, steps = 200, warm = 20, block = 256, shift = 1;
  double tau = 0.8, U = 0.05, lambda = 3.0 / 16.0;
  int vsteps = 51;
};

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

static void usage() {
  printf("lbmbench --mode {info|bw|perf|tgv|verify} [options]\n"
         "  --q 19|27  --stream abpull|abpush|aa|ep|mr  --layout soa|aos\n"
         "  --prec fp64|fp32|mixed|fp16s|fp16c  --coll bgk|trt|reg  --shift 0|1\n"
         "  --n N (cube) | --nx --ny --nz   --steps S  --warm W  --block B\n"
         "  --tau T  --U U  --lambda L(TRT)  --csv file  --tag text\n");
}

static Cfg parse(int argc, char** argv) {
  Cfg c;
  for (int a = 1; a < argc; ++a) {
    std::string k = argv[a];
    auto nx = [&]() -> std::string { if (a + 1 >= argc) { usage(); exit(1); } return argv[++a]; };
    if (k == "--mode") c.mode = nx(); else if (k == "--stream") c.stream = nx();
    else if (k == "--prec") c.prec = nx(); else if (k == "--coll") c.coll = nx();
    else if (k == "--layout") c.layout = nx(); else if (k == "--csv") c.csv = nx();
    else if (k == "--tag") c.tag = nx(); else if (k == "--q") c.q = atoi(nx().c_str());
    else if (k == "--n") { c.nx = c.ny = c.nz = atoi(nx().c_str()); }
    else if (k == "--nx") c.nx = atoi(nx().c_str()); else if (k == "--ny") c.ny = atoi(nx().c_str());
    else if (k == "--nz") c.nz = atoi(nx().c_str()); else if (k == "--steps") c.steps = atoi(nx().c_str());
    else if (k == "--warm") c.warm = atoi(nx().c_str()); else if (k == "--block") c.block = atoi(nx().c_str());
    else if (k == "--shift") c.shift = atoi(nx().c_str()); else if (k == "--tau") c.tau = atof(nx().c_str());
    else if (k == "--U") c.U = atof(nx().c_str()); else if (k == "--lambda") c.lambda = atof(nx().c_str());
    else if (k == "--vsteps") c.vsteps = atoi(nx().c_str());
    else if (k == "-h" || k == "--help") { usage(); exit(0); }
    else { fprintf(stderr, "unknown option %s\n", k.c_str()); usage(); exit(1); }
  }
  return c;
}

static std::string gpu_name() {
  int d; CK(cudaGetDevice(&d)); cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, d));
  std::string s = pr.name; for (auto& ch : s) if (ch == ',' || ch == ' ') ch = '_';
  return s;
}

static void csv_line(const Cfg& c, const char* line) {
  FILE* f = fopen(c.csv.c_str(), "a");
  if (!f) { perror("csv"); return; }
  fseek(f, 0, SEEK_END);
  if (ftell(f) == 0)
    fprintf(f, "gpu,mode,tag,q,stream,layout,prec,coll,shift,nx,ny,nz,steps,block,tau,U,"
               "time_s,mlups,bytes_cell_step,eff_bw_gbs,mem_bytes_cell,t_start_ms,t_end_ms,"
               "l2err,mass_drift,maxdiff\n");
  fprintf(f, "%s\n", line); fclose(f);
}

// Initial macroscopic field on host. kind 0: 2D Taylor-Green (x-y), kind 1:
// 3D Taylor-Green-like field + small deterministic density noise (verify).
static void init_field(const Cfg& c, std::vector<double>& m0, int kind) {
  const unsigned n = (unsigned)c.nx * c.ny * c.nz; m0.assign(4 * (size_t)n, 0.0);
  const double kx = 2 * M_PI / c.nx, ky = 2 * M_PI / c.ny, kz = 2 * M_PI / c.nz;
  for (int z = 0; z < c.nz; ++z) for (int y = 0; y < c.ny; ++y) for (int x = 0; x < c.nx; ++x) {
    const size_t i = x + (size_t)c.nx * (y + (size_t)c.ny * z);
    if (kind == 0) {
      const double X = kx * (x + 0.5), Y = ky * (y + 0.5);
      // rho = 1 + p'/cs2 with p' = -(U^2/4)(cos2X + cos2Y) for kx=ky
      m0[i] = 1.0 - 3.0 * 0.25 * c.U * c.U * (std::cos(2 * X) + std::cos(2 * Y));
      m0[n + i] = -c.U * std::cos(X) * std::sin(Y);
      m0[2 * (size_t)n + i] = c.U * std::sin(X) * std::cos(Y);
      m0[3 * (size_t)n + i] = 0.0;
    } else {
      const double X = kx * x, Y = ky * y, Z = kz * z;
      uint32_t h = (uint32_t)i * 2654435761u; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
      m0[i] = 1.0 + 1e-3 * ((h & 0xFFFF) / 65535.0 - 0.5);
      m0[n + i] = c.U * std::sin(X) * std::cos(Y) * std::cos(Z);
      m0[2 * (size_t)n + i] = -c.U * std::cos(X) * std::sin(Y) * std::cos(Z);
      m0[3 * (size_t)n + i] = 0.3 * c.U * std::sin(Z + X);
    }
  }
}

struct RunOut { double time_s = 0, mlups = 0, t0 = 0, t1 = 0; std::vector<double> macro; double bcs = 0, mem = 0; };

// Generic runner for population patterns.
template <class V, class P, int COLL, int ST, int LA>
static RunOut run_pop(const Cfg& c, const std::vector<double>* m0h, int steps, bool want_macro) {
  RunOut o; Prm p; p.nx = c.nx; p.ny = c.ny; p.nz = c.nz; p.n = (unsigned)((size_t)c.nx * c.ny * c.nz);
  p.omega = 1.0 / c.tau; p.shift = c.shift; p.U = c.U;
  p.omegam = 1.0 / (c.lambda / (c.tau - 0.5) + 0.5);
  using S = typename P::S; constexpr int Q = V::Q;
  const size_t nf = (size_t)Q * p.n;
  const bool two = (ST == AB_PULL || ST == AB_PUSH);
  S *fa = nullptr, *fb = nullptr; double *m0 = nullptr, *mac = nullptr;
  CK(cudaMalloc(&fa, nf * sizeof(S)));
  if (two) CK(cudaMalloc(&fb, nf * sizeof(S)));
  if (m0h) {
    CK(cudaMalloc(&m0, 4 * (size_t)p.n * sizeof(double)));
    CK(cudaMemcpy(m0, m0h->data(), 4 * (size_t)p.n * sizeof(double), cudaMemcpyHostToDevice));
  }
  const unsigned nb = (p.n + c.block - 1) / c.block;
  k_init<V, P, ST, LA><<<nb, c.block>>>(fa, p, m0); CK(cudaGetLastError());
  CK(cudaDeviceSynchronize()); if (m0) CK(cudaFree(m0));
  if (want_macro) CK(cudaMalloc(&mac, 4 * (size_t)p.n * sizeof(double)));
  int t = 0;
  auto step = [&](double* mptr) {
    if constexpr (ST == AA || ST == EP) {
      if (t & 1) k_step<V, P, COLL, ST, LA, 1><<<nb, c.block>>>(fa, fb, p, mptr);
      else       k_step<V, P, COLL, ST, LA, 0><<<nb, c.block>>>(fa, fb, p, mptr);
    } else {
      k_step<V, P, COLL, ST, LA, 0><<<nb, c.block>>>(fa, fb, p, mptr);
    }
    if (two) std::swap(fa, fb);
    ++t;
  };
  for (int w = 0; w < c.warm && !want_macro; ++w) step(nullptr);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  o.t0 = now_ms(); CK(cudaEventRecord(e0));
  for (int s = 0; s < steps; ++s) step((want_macro && s == steps - 1) ? mac : nullptr);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); o.t1 = now_ms(); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  o.time_s = ms * 1e-3; o.mlups = (double)p.n * steps / o.time_s * 1e-6;
  o.bcs = 2.0 * Q * sizeof(S); o.mem = (two ? 2.0 : 1.0) * Q * sizeof(S);
  if (want_macro) { o.macro.resize(4 * (size_t)p.n);
    CK(cudaMemcpy(o.macro.data(), mac, o.macro.size() * sizeof(double), cudaMemcpyDeviceToHost)); CK(cudaFree(mac)); }
  CK(cudaFree(fa)); if (fb) CK(cudaFree(fb));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return o;
}

template <class V, class P>
static RunOut run_mr(const Cfg& c, const std::vector<double>* m0h, int steps, bool want_macro) {
  RunOut o; Prm p; p.nx = c.nx; p.ny = c.ny; p.nz = c.nz; p.n = (unsigned)((size_t)c.nx * c.ny * c.nz);
  p.omega = 1.0 / c.tau; p.shift = c.shift; p.omegam = 0; p.U = c.U;
  using S = typename P::S;
  S *ma, *mb; double *m0 = nullptr, *mac = nullptr;
  CK(cudaMalloc(&ma, (size_t)NM * p.n * sizeof(S))); CK(cudaMalloc(&mb, (size_t)NM * p.n * sizeof(S)));
  if (m0h) {
    CK(cudaMalloc(&m0, 4 * (size_t)p.n * sizeof(double)));
    CK(cudaMemcpy(m0, m0h->data(), 4 * (size_t)p.n * sizeof(double), cudaMemcpyHostToDevice));
  }
  const unsigned nb = (p.n + c.block - 1) / c.block;
  k_init_mr<P><<<nb, c.block>>>(ma, p, m0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); if (m0) CK(cudaFree(m0));
  if (want_macro) CK(cudaMalloc(&mac, 4 * (size_t)p.n * sizeof(double)));
  auto step = [&](double* mptr) { k_mr<V, P><<<nb, c.block>>>(ma, mb, p, mptr); std::swap(ma, mb); };
  for (int w = 0; w < c.warm && !want_macro; ++w) step(nullptr);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  o.t0 = now_ms(); CK(cudaEventRecord(e0));
  for (int s = 0; s < steps; ++s) step((want_macro && s == steps - 1) ? mac : nullptr);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); o.t1 = now_ms(); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  o.time_s = ms * 1e-3; o.mlups = (double)p.n * steps / o.time_s * 1e-6;
  o.bcs = 2.0 * NM * sizeof(S); o.mem = 2.0 * NM * sizeof(S);
  if (want_macro) { o.macro.resize(4 * (size_t)p.n);
    CK(cudaMemcpy(o.macro.data(), mac, o.macro.size() * sizeof(double), cudaMemcpyDeviceToHost)); CK(cudaFree(mac)); }
  CK(cudaFree(ma)); CK(cudaFree(mb)); CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return o;
}

// ---------------------------- dispatch ---------------------------------------
using Runner = RunOut (*)(const Cfg&, const std::vector<double>*, int, bool);

template <class V, class P, int COLL, int ST>
static Runner pick_layout(const Cfg& c) {
  if (c.layout == "soa") return &run_pop<V, P, COLL, ST, SOA>;
  if constexpr ((std::is_same<P, P32>::value || std::is_same<P, P64>::value) && COLL == BGK) {
    if (c.layout == "aos") return &run_pop<V, P, COLL, ST, AOS>;
  }
  fprintf(stderr, "layout '%s' not built for this combination (aos: fp32/fp64 + bgk only)\n", c.layout.c_str());
  exit(1);
}
template <class V, class P, int COLL>
static Runner pick_stream(const Cfg& c) {
  if (c.stream == "abpull") return pick_layout<V, P, COLL, AB_PULL>(c);
  if (c.stream == "abpush") return pick_layout<V, P, COLL, AB_PUSH>(c);
  if (c.stream == "aa") return pick_layout<V, P, COLL, AA>(c);
  if (c.stream == "ep") return pick_layout<V, P, COLL, EP>(c);
  if (c.stream == "mr") {
    if constexpr (COLL == REG && (std::is_same<P, P32>::value || std::is_same<P, P64>::value))
      return &run_mr<V, P>;
    fprintf(stderr, "mr requires --coll reg and --prec fp32|fp64\n"); exit(1);
  }
  fprintf(stderr, "unknown stream %s\n", c.stream.c_str()); exit(1);
}
template <class V, class P>
static Runner pick_coll(const Cfg& c) {
  if (c.coll == "bgk") return pick_stream<V, P, BGK>(c);
  if (c.coll == "trt") return pick_stream<V, P, TRT>(c);
  if (c.coll == "reg") return pick_stream<V, P, REG>(c);
  fprintf(stderr, "unknown coll %s\n", c.coll.c_str()); exit(1);
}
template <class V>
static Runner pick_prec(const Cfg& c) {
  if (c.prec == "fp64") return pick_coll<V, P64>(c);
  if (c.prec == "fp32") return pick_coll<V, P32>(c);
  if (c.prec == "mixed") return pick_coll<V, PMX>(c);
  if (c.prec == "fp16s") return pick_coll<V, P16S>(c);
  if (c.prec == "fp16c") return pick_coll<V, P16C>(c);
  fprintf(stderr, "unknown prec %s\n", c.prec.c_str()); exit(1);
}
static Runner pick(const Cfg& c) {
  if (c.q == 19) return pick_prec<D3Q19>(c);
#ifndef ONLY_D3Q19
  if (c.q == 27) return pick_prec<D3Q27>(c);
#endif
  fprintf(stderr, "q must be 19 or 27 (27 disabled if built with -DONLY_D3Q19)\n"); exit(1);
}

// ------------------------------ modes ---------------------------------------
static void line_out(const Cfg& c, const RunOut& o, double bw, double l2, double drift, double md, int steps) {
  char buf[1024];
  snprintf(buf, sizeof buf,
           "%s,%s,%s,%d,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%.6g,%.6g,%.6e,%.4f,%.1f,%.3f,%.1f,%.3f,%.3f,%.6e,%.6e,%.6e",
           gpu_name().c_str(), c.mode.c_str(), c.tag.c_str(), c.q, c.stream.c_str(), c.layout.c_str(),
           c.prec.c_str(), c.coll.c_str(), c.shift, c.nx, c.ny, c.nz, steps, c.block, c.tau, c.U,
           o.time_s, o.mlups, o.bcs, bw, o.mem, o.t0, o.t1, l2, drift, md);
  csv_line(c, buf);
}

static void mode_info() {
  int d; CK(cudaGetDevice(&d)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
  int clk = 0, mclk = 0, bus = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d);
  cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, d);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, d);
  printf("device,%s\ncc,%d.%d\nsm_count,%d\nglobal_mem_GiB,%.2f\nl2_MiB,%.1f\n"
         "smem_per_block_KiB,%zu\nregs_per_sm,%d\nsm_clock_MHz,%.0f\nmem_clock_MHz,%.0f\nbus_bits,%d\n"
         "theoretical_bw_GBs,%.1f\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0,
         p.l2CacheSize / 1048576.0, p.sharedMemPerBlock / 1024, p.regsPerMultiprocessor,
         clk / 1e3, mclk / 1e3, bus, 2.0 * mclk * 1e3 * (bus / 8) / 1e9);
  printf("note: theoretical_bw assumes DDR factor 2; for GDDR7 check the vendor value.\n");
}

static void mode_bw(const Cfg& c) {
  const size_t bytes = (size_t)1 << 30;  // 1 GiB per array
  const size_t n4 = bytes / sizeof(float4);
  float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
  CK(cudaMemset(a, 0, bytes)); CK(cudaMemset(b, 0, bytes));
  int d, sms; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d));
  const int grid = sms * 8, reps = 20;
  for (int r = 0; r < 3; ++r) k_copy<<<grid, 512>>>(a, b, n4);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  RunOut o; o.t0 = now_ms(); CK(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) k_copy<<<grid, 512>>>(a, b, n4);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); o.t1 = now_ms(); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); o.time_s = ms * 1e-3;
  const double gbs = 2.0 * bytes * reps / o.time_s / 1e9;
  printf("copy bandwidth (read+write): %.1f GB/s\n", gbs);
  Cfg cc = c; cc.stream = "copy"; cc.prec = "-"; cc.coll = "-"; cc.layout = "-";
  line_out(cc, o, gbs, 0, 0, 0, reps);
  CK(cudaFree(a)); CK(cudaFree(b));
}

static void mode_perf(const Cfg& c) {
  Runner r = pick(c);
  RunOut o = r(c, nullptr, c.steps, false);
  const double bw = o.mlups * 1e6 * o.bcs / 1e9;
  printf("%s %s %s %s %s shift=%d N=%dx%dx%d : %.1f MLUPS, %.1f B/cell/step, eff. BW %.1f GB/s, mem %.0f B/cell\n",
         c.q == 19 ? "D3Q19" : "D3Q27", c.stream.c_str(), c.layout.c_str(), c.prec.c_str(), c.coll.c_str(),
         c.shift, c.nx, c.ny, c.nz, o.mlups, o.bcs, bw, o.mem);
  line_out(c, o, bw, 0, 0, 0, c.steps);
}

// 2D Taylor-Green vortex (Nz = 1 recommended). Run for --steps steps; the
// output of the last step corresponds to t = steps (see k_init comment).
static void mode_tgv(const Cfg& c) {
  Runner r = pick(c);
  std::vector<double> m0; init_field(c, m0, 0);
  RunOut o = r(c, &m0, c.steps, true);
  const unsigned n = (unsigned)((size_t)c.nx * c.ny * c.nz);
  const double nu = (c.tau - 0.5) / 3.0, kx = 2 * M_PI / c.nx, ky = 2 * M_PI / c.ny;
  const double t = (double)c.steps, dec = std::exp(-nu * (kx * kx + ky * ky) * t);
  double num = 0, den = 0, mass = 0;
  for (unsigned i = 0; i < n; ++i) {
    const double ua = m0[n + i] * dec, va = m0[2 * (size_t)n + i] * dec;
    const double du = o.macro[n + i] - ua, dv = o.macro[2 * (size_t)n + i] - va;
    num += du * du + dv * dv; den += ua * ua + va * va; mass += o.macro[i] - m0[i];
  }
  const double l2 = std::sqrt(num / den);
  printf("TGV %s %s %s shift=%d L=%d U=%.3g tau=%.3g t=%d : L2 = %.4e, mass drift = %.3e\n",
         c.stream.c_str(), c.prec.c_str(), c.coll.c_str(), c.shift, c.nx, c.U, c.tau, c.steps, l2, mass / n);
  line_out(c, o, 0, l2, mass / n, 0, c.steps);
}

// Cross-check every propagation pattern against AB-pull.
static void mode_verify(const Cfg& c0) {
  Cfg c = c0; std::vector<double> m0; init_field(c, m0, 1);
  const unsigned n = (unsigned)((size_t)c.nx * c.ny * c.nz);
  Cfg cr = c; cr.stream = "abpull"; cr.layout = "soa";
  RunOut ref = pick(cr)(cr, &m0, c.vsteps, true);
  const char* pats[] = {"abpull", "abpush", "aa", "ep"};
  const char* lays[] = {"soa", "aos"};
  int fails = 0;
  for (auto l : lays) for (auto s : pats) {
    Cfg cc = c; cc.stream = s; cc.layout = l;
    if (std::string(l) == "aos" && !((c.prec == "fp32" || c.prec == "fp64") && c.coll == "bgk")) continue;
    RunOut o = pick(cc)(cc, &m0, c.vsteps, true);
    double md = 0; for (size_t i = 0; i < o.macro.size(); ++i) md = std::fmax(md, std::fabs(o.macro[i] - ref.macro[i]));
    // Tolerance: a few units in the last place of the storage format (EP kernels
    // may be contracted differently by the compiler; differences of 1 ulp are benign).
    const double tolp = (c.prec == "fp64" || c.prec == "mixed") ? 1e-12
                      : (c.prec == "fp32" ? 1e-6 : 1e-4);
    printf("verify %-6s %-3s vs abpull/soa after %d steps: max|diff| = %.3e %s\n", s, l, c.vsteps, md,
           md == 0.0 ? "(bitwise identical)" : (md < tolp ? "(OK, round-off)" : "(CHECK)"));
    if (md >= tolp) ++fails;
    cc.mode = "verify"; line_out(cc, o, 0, 0, 0, md, c.vsteps);
  }
  if (c.coll == "reg" && (c.prec == "fp64" || c.prec == "fp32")) {
    Cfg cc = c; cc.stream = "mr";
    RunOut o = pick(cc)(cc, &m0, c.vsteps, true);
    double md = 0; for (size_t i = 0; i < o.macro.size(); ++i) md = std::fmax(md, std::fabs(o.macro[i] - ref.macro[i]));
    const double tol = c.prec == "fp64" ? 1e-12 : 1e-5;
    printf("verify mr (moment repr.) vs abpull+reg: max|diff| = %.3e %s (round-off level expected)\n", md,
           md < tol ? "(OK)" : "(CHECK)");
    if (md >= tol) ++fails;
    line_out(cc, o, 0, 0, 0, md, c.vsteps);
  }
  double mass = 0; for (unsigned i = 0; i < n; ++i) mass += ref.macro[i] - m0[i];
  printf("reference mass drift per node: %.3e\n", mass / n);
  printf(fails ? "VERIFY: %d CHECK(s) -- inspect above\n" : "VERIFY: all patterns consistent\n", fails);
}

int main(int argc, char** argv) {
  Cfg c = parse(argc, argv);
  if (c.mode == "info") { mode_info(); return 0; }
  if (c.mode == "bw") { mode_bw(c); return 0; }
  if ((size_t)c.nx * c.ny * c.nz >= 4294967295ull) { fprintf(stderr, "grid too large\n"); return 1; }
  if (c.block > 256 || c.block < 32) { fprintf(stderr, "--block must be in [32,256]\n"); return 1; }
  if (c.mode == "perf") mode_perf(c);
  else if (c.mode == "tgv") mode_tgv(c);
  else if (c.mode == "verify") mode_verify(c);
  else { usage(); return 1; }
  return 0;
}
