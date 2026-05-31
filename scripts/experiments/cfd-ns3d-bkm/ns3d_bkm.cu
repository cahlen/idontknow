/*
 * 3D incompressible Navier-Stokes (vorticity form), pseudospectral + cuFFT C2C
 *
 *   ω_t + (u·∇)ω = (ω·∇)u + ν∇²ω,   ω = ∇×u,   ∇·u = 0
 *
 * Periodic [0, 2π)³. 2/3 dealiasing. RK4. BKM: ∫ ||ω||_∞ dt
 *
 * Run: ./ns3d_bkm [N] [nu] [n_steps] [dt] [ic] [out_dir]
 */

#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CERTIFICATE_ERROR: %s: %s\n", #call,             \
                    cudaGetErrorString(err));                                  \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult err = (call);                                              \
        if (err != CUFFT_SUCCESS) {                                            \
            fprintf(stderr, "CERTIFICATE_ERROR: cufft %d\n", (int)err);        \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

struct Grid3 {
    int N;
    size_t n;
    double L;
    double dx;
    double *kx_dev;
    double *ky_dev;
    double *kz_dev;
    double *k2_dev;
    double *dealias_dev;
};

__device__ __forceinline__ void cmul(double ar, double ai, double br, double bi, double *cr,
                                     double *ci) {
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
}

__device__ __forceinline__ void cscale(double s, double *r, double *i) {
    *r *= s;
    *i *= s;
}

__global__ void init_wavenumbers3d(double *kx, double *ky, double *kz, double *k2, int N,
                                 double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N || k >= N) return;
    int idx = k * N * N + j * N + i;
    double dk = TWO_PI / L;
    double kxi = (i <= N / 2) ? i * dk : (i - N) * dk;
    double kyj = (j <= N / 2) ? j * dk : (j - N) * dk;
    double kzi = (k <= N / 2) ? k * dk : (k - N) * dk;
    kx[idx] = kxi;
    ky[idx] = kyj;
    kz[idx] = kzi;
    k2[idx] = kxi * kxi + kyj * kyj + kzi * kzi;
}

__global__ void init_dealias3d(double *mask, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N || k >= N) return;
    int idx = k * N * N + j * N + i;
    double dk = TWO_PI / L;
    double kxi = fabs((i <= N / 2) ? i * dk : (i - N) * dk);
    double kyj = fabs((j <= N / 2) ? j * dk : (j - N) * dk);
    double kzi = fabs((k <= N / 2) ? k * dk : (k - N) * dk);
    double kmax = (N / 3) * dk;
    mask[idx] = (kxi <= kmax && kyj <= kmax && kzi <= kmax) ? 1.0 : 0.0;
}

__global__ void init_taylor_green_vorticity(cufftDoubleComplex *wx, cufftDoubleComplex *wy,
                                            cufftDoubleComplex *wz, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N || k >= N) return;
    int idx = k * N * N + j * N + i;
    double x = i * L / N;
    double y = j * L / N;
    double z = k * L / N;
    wx[idx].x = 2.0 * sin(x) * sin(y) * sin(z);
    wx[idx].y = 0.0;
    wy[idx].x = -2.0 * sin(x) * cos(y) * sin(z);
    wy[idx].y = 0.0;
    wz[idx].x = -4.0 * cos(x) * cos(y) * cos(z);
    wz[idx].y = 0.0;
}

__global__ void init_random_vorticity3d(cufftDoubleComplex *wx, cufftDoubleComplex *wy,
                                        cufftDoubleComplex *wz, int N, double L,
                                        unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= N || j >= N || k >= N) return;
    int idx = k * N * N + j * N + i;
    unsigned long long s = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)idx);
    s ^= s >> 33;
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    double r1 = (s >> 11) * (1.0 / 9007199254740992.0);
    s ^= s >> 17;
    double r2 = (s >> 11) * (1.0 / 9007199254740992.0);
    s ^= s >> 17;
    double r3 = (s >> 11) * (1.0 / 9007199254740992.0);
    double x = i * L / N;
    double y = j * L / N;
    double z = k * L / N;
    double env = exp(-3.0 * ((x - PI) * (x - PI) + (y - PI) * (y - PI) + (z - PI) * (z - PI)));
    wx[idx].x = (r1 - 0.5) * env;
    wy[idx].x = (r2 - 0.5) * env;
    wz[idx].x = (r3 - 0.5) * env;
    wx[idx].y = wy[idx].y = wz[idx].y = 0.0;
}

__global__ void velocity_from_vorticity(cufftDoubleComplex *ux, cufftDoubleComplex *uy,
                                        cufftDoubleComplex *uz, const cufftDoubleComplex *wx,
                                        const cufftDoubleComplex *wy, const cufftDoubleComplex *wz,
                                        const double *kx, const double *ky, const double *kz,
                                        const double *k2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double k2v = k2[idx];
    if (k2v < 1e-30) {
        ux[idx].x = ux[idx].y = uy[idx].x = uy[idx].y = uz[idx].x = uz[idx].y = 0.0;
        return;
    }
    double kxv = kx[idx], kyv = ky[idx], kzv = kz[idx];
    double oxr = wx[idx].x, oxi = wx[idx].y;
    double oyr = wy[idx].x, oyi = wy[idx].y;
    double ozr = wz[idx].x, ozi = wz[idx].y;
    double cxr, cxi, cyr, cyi, czr, czi;
    cmul(kyv, 0.0, ozr, ozi, &cxr, &cxi);
    double tmr, tmi;
    cmul(kzv, 0.0, oyr, oyi, &tmr, &tmi);
    cxr -= tmr;
    cxi -= tmi;
    cmul(kzv, 0.0, oxr, oxi, &cyr, &cyi);
    cmul(kxv, 0.0, ozr, ozi, &tmr, &tmi);
    cyr -= tmr;
    cyi -= tmi;
    cmul(kxv, 0.0, oyr, oyi, &czr, &czi);
    cmul(kyv, 0.0, oxr, oxi, &tmr, &tmi);
    czr -= tmr;
    czi -= tmi;
    double inv = 1.0 / k2v;
    ux[idx].x = -cxi * inv;
    ux[idx].y = cxr * inv;
    uy[idx].x = -cyi * inv;
    uy[idx].y = cyr * inv;
    uz[idx].x = -czi * inv;
    uz[idx].y = czr * inv;
}

__global__ void spectral_deriv3(cufftDoubleComplex *out, const cufftDoubleComplex *in,
                                const double *k, int n, int axis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double kv = k[idx];
    double ir = in[idx].x, ii = in[idx].y;
    if (axis == 0) {
        out[idx].x = -kv * ii;
        out[idx].y = kv * ir;
    } else {
        out[idx].x = kv * ii;
        out[idx].y = -kv * ir;
    }
}

__global__ void vorticity_rhs_physical(cufftDoubleComplex *rwx, cufftDoubleComplex *rwy,
                                       cufftDoubleComplex *rwz, const cufftDoubleComplex *wx,
                                       const cufftDoubleComplex *wy, const cufftDoubleComplex *wz,
                                       const cufftDoubleComplex *ux, const cufftDoubleComplex *uy,
                                       const cufftDoubleComplex *uz, const cufftDoubleComplex *dwx_dx,
                                       const cufftDoubleComplex *dwx_dy,
                                       const cufftDoubleComplex *dwx_dz,
                                       const cufftDoubleComplex *dwy_dx,
                                       const cufftDoubleComplex *dwy_dy,
                                       const cufftDoubleComplex *dwy_dz,
                                       const cufftDoubleComplex *dwz_dx,
                                       const cufftDoubleComplex *dwz_dy,
                                       const cufftDoubleComplex *dwz_dz,
                                       const cufftDoubleComplex *dux_dx,
                                       const cufftDoubleComplex *dux_dy,
                                       const cufftDoubleComplex *dux_dz,
                                       const cufftDoubleComplex *duy_dx,
                                       const cufftDoubleComplex *duy_dy,
                                       const cufftDoubleComplex *duy_dz,
                                       const cufftDoubleComplex *duz_dx,
                                       const cufftDoubleComplex *duz_dy,
                                       const cufftDoubleComplex *duz_dz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double wxr = wx[idx].x, wyr = wy[idx].x, wzr = wz[idx].x;
    double uxr = ux[idx].x, uyr = uy[idx].x, uzr = uz[idx].x;

    double advx = -(uxr * dwx_dx[idx].x + uyr * dwx_dy[idx].x + uzr * dwx_dz[idx].x);
    double advy = -(uxr * dwy_dx[idx].x + uyr * dwy_dy[idx].x + uzr * dwy_dz[idx].x);
    double advz = -(uxr * dwz_dx[idx].x + uyr * dwz_dy[idx].x + uzr * dwz_dz[idx].x);

    double strx = wxr * dux_dx[idx].x + wyr * dux_dy[idx].x + wzr * dux_dz[idx].x;
    double stry = wxr * duy_dx[idx].x + wyr * duy_dy[idx].x + wzr * duy_dz[idx].x;
    double strz = wxr * duz_dx[idx].x + wyr * duz_dy[idx].x + wzr * duz_dz[idx].x;

    rwx[idx].x = advx + strx;
    rwx[idx].y = 0.0;
    rwy[idx].x = advy + stry;
    rwy[idx].y = 0.0;
    rwz[idx].x = advz + strz;
    rwz[idx].y = 0.0;
}

__global__ void apply_dealias3(cufftDoubleComplex *f, const double *mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double m = mask[idx];
    f[idx].x *= m;
    f[idx].y *= m;
}

__global__ void add_viscous(cufftDoubleComplex *rhs, const cufftDoubleComplex *w,
                            const double *k2, double nu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double lap = -k2[idx];
    rhs[idx].x += nu * lap * w[idx].x;
    rhs[idx].y += nu * lap * w[idx].y;
}

__global__ void scale_complex3(cufftDoubleComplex *x, double s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx].x *= s;
    x[idx].y *= s;
}

__global__ void axpy_complex3(cufftDoubleComplex *y, const cufftDoubleComplex *x, double a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx].x += a * x[idx].x;
    y[idx].y += a * x[idx].y;
}

__global__ void lincomb_complex3(cufftDoubleComplex *dst, const cufftDoubleComplex *a, double sa,
                                 const cufftDoubleComplex *b, double sb, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx].x = sa * a[idx].x + sb * b[idx].x;
    dst[idx].y = sa * a[idx].y + sb * b[idx].y;
}

__global__ void max_vorticity_kernel(const cufftDoubleComplex *wx, const cufftDoubleComplex *wy,
                                     const cufftDoubleComplex *wz, int n, double *block_max) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < n) {
        val = sqrt(wx[idx].x * wx[idx].x + wy[idx].x * wy[idx].x + wz[idx].x * wz[idx].x);
    }
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = smem[0];
}

__global__ void enstrophy_kernel(const cufftDoubleComplex *wx, const cufftDoubleComplex *wy,
                                 const cufftDoubleComplex *wz, int n, double *partial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        partial[idx] = 0.0;
        return;
    }
    partial[idx] = wx[idx].x * wx[idx].x + wy[idx].x * wy[idx].x + wz[idx].x * wz[idx].x;
}

struct Fields3 {
    cufftDoubleComplex *wx, *wy, *wz;
    cufftDoubleComplex *ux, *uy, *uz;
    cufftDoubleComplex *dwx_dx, *dwx_dy, *dwx_dz;
    cufftDoubleComplex *dwy_dx, *dwy_dy, *dwy_dz;
    cufftDoubleComplex *dwz_dx, *dwz_dy, *dwz_dz;
    cufftDoubleComplex *dux_dx, *dux_dy, *dux_dz;
    cufftDoubleComplex *duy_dx, *duy_dy, *duy_dz;
    cufftDoubleComplex *duz_dx, *duz_dy, *duz_dz;
    cufftDoubleComplex *rwx, *rwy, *rwz;
    cufftDoubleComplex *k1x, *k1y, *k1z;
    cufftDoubleComplex *k2x, *k2y, *k2z;
    cufftDoubleComplex *k3x, *k3y, *k3z;
    cufftDoubleComplex *k4x, *k4y, *k4z;
    cufftDoubleComplex *tmpx, *tmpy, *tmpz;
    cufftDoubleComplex *wx_phys, *wy_phys, *wz_phys;
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static cufftDoubleComplex *alloc_c(size_t n) {
    cufftDoubleComplex *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(cufftDoubleComplex)));
    return p;
}

static void ifft_scale(cufftHandle inv, cufftDoubleComplex *hat, cufftDoubleComplex *phys,
                       double inv_n3, int nb) {
    CUFFT_CHECK(cufftExecZ2Z(inv, hat, phys, CUFFT_INVERSE));
    scale_complex3<<<nb, 256>>>(phys, inv_n3, nb * 256);
}

static void fft_fwd(cufftHandle fwd, cufftDoubleComplex *phys, cufftDoubleComplex *hat) {
    CUFFT_CHECK(cufftExecZ2Z(fwd, phys, hat, CUFFT_FORWARD));
}

static void compute_rhs3(cufftHandle fwd, cufftHandle inv, const Fields3 *f, const Grid3 *g,
                         double nu, cufftDoubleComplex *rhsx, cufftDoubleComplex *rhsy,
                         cufftDoubleComplex *rhsz, cufftDoubleComplex *wx,
                         cufftDoubleComplex *wy, cufftDoubleComplex *wz) {
    int n = (int)g->n;
    int nb = (n + 255) / 256;
    double inv_n3 = 1.0 / (double)g->n;

    velocity_from_vorticity<<<nb, 256>>>(f->ux, f->uy, f->uz, wx, wy, wz, g->kx_dev, g->ky_dev,
                                         g->kz_dev, g->k2_dev, n);

    spectral_deriv3<<<nb, 256>>>(f->dwx_dx, wx, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->dwx_dy, wx, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dwx_dz, wx, g->kz_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dwy_dx, wy, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->dwy_dy, wy, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dwy_dz, wy, g->kz_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dwz_dx, wz, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->dwz_dy, wz, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dwz_dz, wz, g->kz_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dux_dx, f->ux, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->dux_dy, f->ux, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->dux_dz, f->ux, g->kz_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->duy_dx, f->uy, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->duy_dy, f->uy, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->duy_dz, f->uy, g->kz_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->duz_dx, f->uz, g->kx_dev, n, 0);
    spectral_deriv3<<<nb, 256>>>(f->duz_dy, f->uz, g->ky_dev, n, 1);
    spectral_deriv3<<<nb, 256>>>(f->duz_dz, f->uz, g->kz_dev, n, 1);

    ifft_scale(inv, wx, f->wx_phys, inv_n3, nb);
    ifft_scale(inv, wy, f->wy_phys, inv_n3, nb);
    ifft_scale(inv, wz, f->wz_phys, inv_n3, nb);
    ifft_scale(inv, f->ux, f->tmpx, inv_n3, nb);
    ifft_scale(inv, f->uy, f->tmpy, inv_n3, nb);
    ifft_scale(inv, f->uz, f->tmpz, inv_n3, nb);
    ifft_scale(inv, f->dwx_dx, f->k1x, inv_n3, nb);
    ifft_scale(inv, f->dwx_dy, f->k1y, inv_n3, nb);
    ifft_scale(inv, f->dwx_dz, f->k1z, inv_n3, nb);
    ifft_scale(inv, f->dwy_dx, f->k2x, inv_n3, nb);
    ifft_scale(inv, f->dwy_dy, f->k2y, inv_n3, nb);
    ifft_scale(inv, f->dwy_dz, f->k2z, inv_n3, nb);
    ifft_scale(inv, f->dwz_dx, f->k3x, inv_n3, nb);
    ifft_scale(inv, f->dwz_dy, f->k3y, inv_n3, nb);
    ifft_scale(inv, f->dwz_dz, f->k3z, inv_n3, nb);
    ifft_scale(inv, f->dux_dx, f->k4x, inv_n3, nb);
    ifft_scale(inv, f->dux_dy, f->k4y, inv_n3, nb);
    ifft_scale(inv, f->dux_dz, f->k4z, inv_n3, nb);
    ifft_scale(inv, f->duy_dx, f->rwx, inv_n3, nb);
    ifft_scale(inv, f->duy_dy, f->rwy, inv_n3, nb);
    ifft_scale(inv, f->duy_dz, f->rwz, inv_n3, nb);
    ifft_scale(inv, f->duz_dx, f->dux_dx, inv_n3, nb);
    ifft_scale(inv, f->duz_dy, f->dux_dy, inv_n3, nb);
    ifft_scale(inv, f->duz_dz, f->dux_dz, inv_n3, nb);

    vorticity_rhs_physical<<<nb, 256>>>(
        f->duy_dx, f->duy_dy, f->duz_dx, f->wx_phys, f->wy_phys, f->wz_phys, f->tmpx, f->tmpy,
        f->tmpz, f->k1x, f->k1y, f->k1z, f->k2x, f->k2y, f->k2z, f->k3x, f->k3y, f->k3z, f->k4x,
        f->k4y, f->k4z, f->rwx, f->rwy, f->rwz, f->dux_dx, f->dux_dy, f->dux_dz, n);

    fft_fwd(fwd, f->duy_dx, rhsx);
    fft_fwd(fwd, f->duy_dy, rhsy);
    fft_fwd(fwd, f->duz_dx, rhsz);
    apply_dealias3<<<nb, 256>>>(rhsx, g->dealias_dev, n);
    apply_dealias3<<<nb, 256>>>(rhsy, g->dealias_dev, n);
    apply_dealias3<<<nb, 256>>>(rhsz, g->dealias_dev, n);
    add_viscous<<<nb, 256>>>(rhsx, wx, g->k2_dev, nu, n);
    add_viscous<<<nb, 256>>>(rhsy, wy, g->k2_dev, nu, n);
    add_viscous<<<nb, 256>>>(rhsz, wz, g->k2_dev, nu, n);
}

static void rk4_step3(cufftHandle fwd, cufftHandle inv, Fields3 *f, const Grid3 *g, double nu,
                      double dt, cufftDoubleComplex *wx, cufftDoubleComplex *wy,
                      cufftDoubleComplex *wz) {
    int n = (int)g->n;
    int nb = (n + 255) / 256;

    compute_rhs3(fwd, inv, f, g, nu, f->k1x, f->k1y, f->k1z, wx, wy, wz);
    lincomb_complex3<<<nb, 256>>>(f->tmpx, wx, 1.0, f->k1x, dt * 0.5, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpy, wy, 1.0, f->k1y, dt * 0.5, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpz, wz, 1.0, f->k1z, dt * 0.5, n);
    compute_rhs3(fwd, inv, f, g, nu, f->k2x, f->k2y, f->k2z, f->tmpx, f->tmpy, f->tmpz);
    lincomb_complex3<<<nb, 256>>>(f->tmpx, wx, 1.0, f->k2x, dt * 0.5, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpy, wy, 1.0, f->k2y, dt * 0.5, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpz, wz, 1.0, f->k2z, dt * 0.5, n);
    compute_rhs3(fwd, inv, f, g, nu, f->k3x, f->k3y, f->k3z, f->tmpx, f->tmpy, f->tmpz);
    lincomb_complex3<<<nb, 256>>>(f->tmpx, wx, 1.0, f->k3x, dt, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpy, wy, 1.0, f->k3y, dt, n);
    lincomb_complex3<<<nb, 256>>>(f->tmpz, wz, 1.0, f->k3z, dt, n);
    compute_rhs3(fwd, inv, f, g, nu, f->k4x, f->k4y, f->k4z, f->tmpx, f->tmpy, f->tmpz);

    axpy_complex3<<<nb, 256>>>(wx, f->k1x, dt / 6.0, n);
    axpy_complex3<<<nb, 256>>>(wy, f->k1y, dt / 6.0, n);
    axpy_complex3<<<nb, 256>>>(wz, f->k1z, dt / 6.0, n);
    axpy_complex3<<<nb, 256>>>(wx, f->k2x, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wy, f->k2y, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wz, f->k2z, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wx, f->k3x, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wy, f->k3y, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wz, f->k3z, dt / 3.0, n);
    axpy_complex3<<<nb, 256>>>(wx, f->k4x, dt / 6.0, n);
    axpy_complex3<<<nb, 256>>>(wy, f->k4y, dt / 6.0, n);
    axpy_complex3<<<nb, 256>>>(wz, f->k4z, dt / 6.0, n);
}

static double gpu_max_vorticity3(cufftDoubleComplex *wx, cufftDoubleComplex *wy,
                                 cufftDoubleComplex *wz, int n) {
    int nblocks = (n + 255) / 256;
    double *block_max;
    CUDA_CHECK(cudaMalloc(&block_max, nblocks * sizeof(double)));
    max_vorticity_kernel<<<nblocks, 256>>>(wx, wy, wz, n, block_max);
    double *h = (double *)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h, block_max, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0.0;
    for (int i = 0; i < nblocks; i++) mx = fmax(mx, h[i]);
    free(h);
    cudaFree(block_max);
    return mx;
}

static double gpu_enstrophy3(cufftDoubleComplex *wx, cufftDoubleComplex *wy,
                             cufftDoubleComplex *wz, int n, double dx) {
    double *partial;
    CUDA_CHECK(cudaMalloc(&partial, n * sizeof(double)));
    enstrophy_kernel<<<(n + 255) / 256, 256>>>(wx, wy, wz, n, partial);
    double *h = (double *)malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h, partial, n * sizeof(double), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += h[i];
    free(h);
    cudaFree(partial);
    return sum * dx * dx * dx;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 64;
    double nu = argc > 2 ? atof(argv[2]) : 0.01;
    int n_steps = argc > 3 ? atoi(argv[3]) : 1000;
    double dt = argc > 4 ? atof(argv[4]) : 0.002;
    const char *ic = argc > 5 ? argv[5] : "taylor-green";
    const char *out_dir = argc > 6 ? argv[6] : "scripts/experiments/cfd-ns3d-bkm/results";

    if (N < 16 || (N & (N - 1)) != 0) {
        fprintf(stderr, "N must be power of 2 >= 16\n");
        return 1;
    }

    Grid3 g;
    g.N = N;
    g.n = (size_t)N * N * N;
    g.L = TWO_PI;
    g.dx = g.L / N;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("==========================================\n");
    printf("  CFD Phase 3 — 3D NS pseudospectral BKM\n");
    printf("  Device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Grid: %d³, nu=%.6e, dt=%.4f, steps=%d, IC=%s\n", N, nu, dt, n_steps, ic);
    printf("==========================================\n\n");

    CUDA_CHECK(cudaMalloc(&g.kx_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.ky_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.kz_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.k2_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.dealias_dev, g.n * sizeof(double)));

    dim3 block(8, 8, 8);
    dim3 grid((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);
    init_wavenumbers3d<<<grid, block>>>(g.kx_dev, g.ky_dev, g.kz_dev, g.k2_dev, N, g.L);
    init_dealias3d<<<grid, block>>>(g.dealias_dev, N, g.L);

    Fields3 f;
    f.wx = alloc_c(g.n);
    f.wy = alloc_c(g.n);
    f.wz = alloc_c(g.n);
    f.ux = alloc_c(g.n);
    f.uy = alloc_c(g.n);
    f.uz = alloc_c(g.n);
    f.dwx_dx = alloc_c(g.n);
    f.dwx_dy = alloc_c(g.n);
    f.dwx_dz = alloc_c(g.n);
    f.dwy_dx = alloc_c(g.n);
    f.dwy_dy = alloc_c(g.n);
    f.dwy_dz = alloc_c(g.n);
    f.dwz_dx = alloc_c(g.n);
    f.dwz_dy = alloc_c(g.n);
    f.dwz_dz = alloc_c(g.n);
    f.dux_dx = alloc_c(g.n);
    f.dux_dy = alloc_c(g.n);
    f.dux_dz = alloc_c(g.n);
    f.duy_dx = alloc_c(g.n);
    f.duy_dy = alloc_c(g.n);
    f.duy_dz = alloc_c(g.n);
    f.duz_dx = alloc_c(g.n);
    f.duz_dy = alloc_c(g.n);
    f.duz_dz = alloc_c(g.n);
    f.rwx = alloc_c(g.n);
    f.rwy = alloc_c(g.n);
    f.rwz = alloc_c(g.n);
    f.k1x = alloc_c(g.n);
    f.k1y = alloc_c(g.n);
    f.k1z = alloc_c(g.n);
    f.k2x = alloc_c(g.n);
    f.k2y = alloc_c(g.n);
    f.k2z = alloc_c(g.n);
    f.k3x = alloc_c(g.n);
    f.k3y = alloc_c(g.n);
    f.k3z = alloc_c(g.n);
    f.k4x = alloc_c(g.n);
    f.k4y = alloc_c(g.n);
    f.k4z = alloc_c(g.n);
    f.tmpx = alloc_c(g.n);
    f.tmpy = alloc_c(g.n);
    f.tmpz = alloc_c(g.n);
    f.wx_phys = alloc_c(g.n);
    f.wy_phys = alloc_c(g.n);
    f.wz_phys = alloc_c(g.n);

    if (strcmp(ic, "random") == 0) {
        init_random_vorticity3d<<<grid, block>>>(f.wx, f.wy, f.wz, N, g.L, 0xDEADBEEFULL);
    } else {
        init_taylor_green_vorticity<<<grid, block>>>(f.wx, f.wy, f.wz, N, g.L);
    }

    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan3d(&plan_fwd, N, N, N, CUFFT_Z2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_inv, N, N, N, CUFFT_Z2Z));

    fft_fwd(plan_fwd, f.wx, f.wx);
    fft_fwd(plan_fwd, f.wy, f.wy);
    fft_fwd(plan_fwd, f.wz, f.wz);
    apply_dealias3<<<(g.n + 255) / 256, 256>>>(f.wx, g.dealias_dev, (int)g.n);
    apply_dealias3<<<(g.n + 255) / 256, 256>>>(f.wy, g.dealias_dev, (int)g.n);
    apply_dealias3<<<(g.n + 255) / 256, 256>>>(f.wz, g.dealias_dev, (int)g.n);

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/bkm3d_n%d_nu%.0e_steps%d.csv", out_dir, N, nu,
             n_steps);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "CERTIFICATE_ERROR: cannot open %s\n", csv_path);
        return 2;
    }
    fprintf(csv, "step,time,max_vorticity,enstrophy,bkm_cumulative\n");

    int n = (int)g.n;
    int nb = (n + 255) / 256;
    double inv_n3 = 1.0 / (double)g.n;
    double t0 = now_seconds(), t = 0.0, bkm = 0.0;
    int nan_count = 0;
    int log_stride = n_steps / 100 + 1;

    for (int step = 0; step <= n_steps; step++) {
        ifft_scale(plan_inv, f.wx, f.wx_phys, inv_n3, nb);
        ifft_scale(plan_inv, f.wy, f.wy_phys, inv_n3, nb);
        ifft_scale(plan_inv, f.wz, f.wz_phys, inv_n3, nb);
        double max_w = gpu_max_vorticity3(f.wx_phys, f.wy_phys, f.wz_phys, n);
        double ens = gpu_enstrophy3(f.wx_phys, f.wy_phys, f.wz_phys, n, g.dx);
        if (isnan(max_w) || isinf(max_w) || isnan(ens) || isinf(ens)) nan_count++;

        if (step % log_stride == 0 || step == n_steps) {
            fprintf(csv, "%d,%.6f,%.10f,%.10f,%.10f\n", step, t, max_w, ens, bkm);
        }
        if (step == n_steps) break;

        rk4_step3(plan_fwd, plan_inv, &f, &g, nu, dt, f.wx, f.wy, f.wz);
        apply_dealias3<<<nb, 256>>>(f.wx, g.dealias_dev, n);
        apply_dealias3<<<nb, 256>>>(f.wy, g.dealias_dev, n);
        apply_dealias3<<<nb, 256>>>(f.wz, g.dealias_dev, n);
        bkm += max_w * dt;
        t += dt;
    }
    fclose(csv);

    double elapsed = now_seconds() - t0;
    printf("Wrote %s\n", csv_path);
    printf("Elapsed: %.2f s (%.1f steps/s)\n", elapsed, (n_steps + 1) / elapsed);
    printf("Final BKM integral: %.6f\n", bkm);
    printf("NaN/Inf diagnostics: %d\n", nan_count);
    if (nan_count > 0) {
        fprintf(stderr, "CERTIFICATE_ERROR: numerical failure\n");
        return 2;
    }
    return 0;
}
