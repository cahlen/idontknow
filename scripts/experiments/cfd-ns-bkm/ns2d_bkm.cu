/*
 * 2D incompressible Navier-Stokes (vorticity form), pseudospectral + cuFFT (C2C)
 *
 *   ω_t + u·∇ω = ν ∇²ω,   u = (∂ψ/∂y, -∂ψ/∂x),   ω = -∇²ψ
 *
 * Periodic [0, 2π)². 2/3 dealiasing. RK4.
 * BKM diagnostic: cumulative ∫ ||ω||_∞ dt (2D is globally regular; probe infrastructure).
 *
 * Run: ./ns2d_bkm [N] [nu] [n_steps] [dt] [ic] [out_dir]
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

struct Grid {
    int N;
    size_t n;
    double L;
    double dx;
    double *kx_dev;
    double *ky_dev;
    double *k2_dev;
    double *dealias_dev;
};

__global__ void init_wavenumbers(double *kx, double *ky, double *k2, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    int idx = j * N + i;
    double dk = TWO_PI / L;
    double kxi = (i <= N / 2) ? i * dk : (i - N) * dk;
    double kyj = (j <= N / 2) ? j * dk : (j - N) * dk;
    kx[idx] = kxi;
    ky[idx] = kyj;
    k2[idx] = kxi * kxi + kyj * kyj;
}

__global__ void init_dealias(double *mask, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    int idx = j * N + i;
    double dk = TWO_PI / L;
    double kxi = fabs((i <= N / 2) ? i * dk : (i - N) * dk);
    double kyj = fabs((j <= N / 2) ? j * dk : (j - N) * dk);
    double kmax = (N / 3) * dk;
    mask[idx] = (kxi <= kmax && kyj <= kmax) ? 1.0 : 0.0;
}

__global__ void init_taylor_green(cufftDoubleComplex *omega_hat, int N, double L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    int idx = j * N + i;
    double x = i * L / N;
    double y = j * L / N;
    omega_hat[idx].x = 2.0 * sin(x) * sin(y);
    omega_hat[idx].y = 0.0;
}

__global__ void init_random_vorticity(cufftDoubleComplex *omega_hat, int N, double L,
                                      unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    int idx = j * N + i;
    unsigned long long s = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)idx);
    s ^= s >> 33;
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    double u = (s >> 11) * (1.0 / 9007199254740992.0);
    double x = i * L / N;
    double y = j * L / N;
    double env = exp(-2.0 * ((x - PI) * (x - PI) + (y - PI) * (y - PI)));
    omega_hat[idx].x = (u - 0.5) * env;
    omega_hat[idx].y = 0.0;
}

__global__ void compute_psi_uv(cufftDoubleComplex *psi_hat, cufftDoubleComplex *u_hat,
                               cufftDoubleComplex *v_hat, const cufftDoubleComplex *omega_hat,
                               const double *k2, const double *kx, const double *ky, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    double k2v = k2[idx];
    if (k2v < 1e-30) {
        psi_hat[idx].x = psi_hat[idx].y = 0.0;
        u_hat[idx].x = u_hat[idx].y = 0.0;
        v_hat[idx].x = v_hat[idx].y = 0.0;
        return;
    }
    double omr = omega_hat[idx].x, omi = omega_hat[idx].y;
    double psir = omr / k2v, psii = omi / k2v;
    psi_hat[idx].x = psir;
    psi_hat[idx].y = psii;
    double kxv = kx[idx], kyv = ky[idx];
    u_hat[idx].x = -kyv * psii;
    u_hat[idx].y = kyv * psir;
    v_hat[idx].x = kxv * psii;
    v_hat[idx].y = -kxv * psir;
}

__global__ void spectral_deriv(cufftDoubleComplex *out, const cufftDoubleComplex *in,
                               const double *k, int n, int component) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double kv = k[idx];
    double ir = in[idx].x, ii = in[idx].y;
    if (component == 0) {
        out[idx].x = -kv * ii;
        out[idx].y = kv * ir;
    } else {
        out[idx].x = kv * ii;
        out[idx].y = -kv * ir;
    }
}

__global__ void nonlinear_physical(cufftDoubleComplex *nl, const cufftDoubleComplex *u,
                                   const cufftDoubleComplex *v, const cufftDoubleComplex *dox,
                                   const cufftDoubleComplex *doy, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    nl[idx].x = -(u[idx].x * dox[idx].x - u[idx].y * dox[idx].y + v[idx].x * doy[idx].x -
                  v[idx].y * doy[idx].y);
    nl[idx].y = -(u[idx].x * dox[idx].y + u[idx].y * dox[idx].x + v[idx].x * doy[idx].y +
                  v[idx].y * doy[idx].x);
}

__global__ void apply_dealias(cufftDoubleComplex *fhat, const double *mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double m = mask[idx];
    fhat[idx].x *= m;
    fhat[idx].y *= m;
}

__global__ void rhs_from_nl_visc(cufftDoubleComplex *rhs, const cufftDoubleComplex *nl_hat,
                                 const cufftDoubleComplex *omega_hat, const double *k2,
                                 double nu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double lap = -k2[idx];
    rhs[idx].x = nl_hat[idx].x + nu * lap * omega_hat[idx].x;
    rhs[idx].y = nl_hat[idx].y + nu * lap * omega_hat[idx].y;
}

__global__ void axpy_complex(cufftDoubleComplex *y, const cufftDoubleComplex *x, double a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx].x += a * x[idx].x;
    y[idx].y += a * x[idx].y;
}

__global__ void lincomb_complex(cufftDoubleComplex *dst, const cufftDoubleComplex *a, double sa,
                                const cufftDoubleComplex *b, double sb, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx].x = sa * a[idx].x + sb * b[idx].x;
    dst[idx].y = sa * a[idx].y + sb * b[idx].y;
}

__global__ void scale_complex(cufftDoubleComplex *x, double s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx].x *= s;
    x[idx].y *= s;
}

__global__ void max_abs_kernel(const cufftDoubleComplex *omega, int n, double *block_max) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < n) val = hypot(omega[idx].x, omega[idx].y);
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.x] = smem[0];
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double gpu_max_abs(cufftDoubleComplex *omega_dev, int n) {
    int nblocks = (n + 255) / 256;
    double *block_max;
    CUDA_CHECK(cudaMalloc(&block_max, nblocks * sizeof(double)));
    max_abs_kernel<<<nblocks, 256>>>(omega_dev, n, block_max);
    double *h = (double *)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h, block_max, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0.0;
    for (int i = 0; i < nblocks; i++) mx = fmax(mx, h[i]);
    free(h);
    cudaFree(block_max);
    return mx;
}

static double gpu_enstrophy(cufftDoubleComplex *omega_dev, int n, double dx) {
    cufftDoubleComplex *h =
        (cufftDoubleComplex *)malloc(n * sizeof(cufftDoubleComplex));
    CUDA_CHECK(cudaMemcpy(h, omega_dev, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += h[i].x * h[i].x + h[i].y * h[i].y;
    free(h);
    return sum * dx * dx;
}

static void compute_rhs(cufftHandle plan_fwd, cufftHandle plan_inv,
                        const cufftDoubleComplex *omega_hat, cufftDoubleComplex *psi_hat,
                        cufftDoubleComplex *u_hat, cufftDoubleComplex *v_hat,
                        cufftDoubleComplex *dox_hat, cufftDoubleComplex *doy_hat,
                        cufftDoubleComplex *nl_hat, cufftDoubleComplex *rhs,
                        cufftDoubleComplex *u_phys, cufftDoubleComplex *v_phys,
                        cufftDoubleComplex *dox_phys, cufftDoubleComplex *doy_phys,
                        cufftDoubleComplex *nl_work, const Grid *g, double nu) {
    int n = (int)g->n;
    int nb = (n + 255) / 256;
    double inv_n2 = 1.0 / (double)n;

    compute_psi_uv<<<nb, 256>>>(psi_hat, u_hat, v_hat, omega_hat, g->k2_dev, g->kx_dev,
                                g->ky_dev, g->N);
    spectral_deriv<<<nb, 256>>>(dox_hat, omega_hat, g->kx_dev, n, 0);
    spectral_deriv<<<nb, 256>>>(doy_hat, omega_hat, g->ky_dev, n, 1);

    CUFFT_CHECK(cufftExecZ2Z(plan_inv, u_hat, u_phys, CUFFT_INVERSE));
    scale_complex<<<nb, 256>>>(u_phys, inv_n2, n);
    CUFFT_CHECK(cufftExecZ2Z(plan_inv, v_hat, v_phys, CUFFT_INVERSE));
    scale_complex<<<nb, 256>>>(v_phys, inv_n2, n);
    CUFFT_CHECK(cufftExecZ2Z(plan_inv, dox_hat, dox_phys, CUFFT_INVERSE));
    scale_complex<<<nb, 256>>>(dox_phys, inv_n2, n);
    CUFFT_CHECK(cufftExecZ2Z(plan_inv, doy_hat, doy_phys, CUFFT_INVERSE));
    scale_complex<<<nb, 256>>>(doy_phys, inv_n2, n);

    nonlinear_physical<<<nb, 256>>>(nl_work, u_phys, v_phys, dox_phys, doy_phys, n);
    CUFFT_CHECK(cufftExecZ2Z(plan_fwd, nl_work, nl_hat, CUFFT_FORWARD));
    apply_dealias<<<nb, 256>>>(nl_hat, g->dealias_dev, n);
    rhs_from_nl_visc<<<nb, 256>>>(rhs, nl_hat, omega_hat, g->k2_dev, nu, n);
}

static void rk4_step(cufftHandle plan_fwd, cufftHandle plan_inv, cufftDoubleComplex *omega_hat,
                     cufftDoubleComplex *k1, cufftDoubleComplex *k2, cufftDoubleComplex *k3,
                     cufftDoubleComplex *k4, cufftDoubleComplex *tmp, cufftDoubleComplex *psi_hat,
                     cufftDoubleComplex *u_hat, cufftDoubleComplex *v_hat,
                     cufftDoubleComplex *dox_hat, cufftDoubleComplex *doy_hat,
                     cufftDoubleComplex *nl_hat, cufftDoubleComplex *u_phys,
                     cufftDoubleComplex *v_phys, cufftDoubleComplex *dox_phys,
                     cufftDoubleComplex *doy_phys, cufftDoubleComplex *nl_work, const Grid *g,
                     double nu, double dt) {
    int n = (int)g->n;
    int nb = (n + 255) / 256;

    compute_rhs(plan_fwd, plan_inv, omega_hat, psi_hat, u_hat, v_hat, dox_hat, doy_hat, nl_hat,
                k1, u_phys, v_phys, dox_phys, doy_phys, nl_work, g, nu);
    lincomb_complex<<<nb, 256>>>(tmp, omega_hat, 1.0, k1, dt * 0.5, n);
    compute_rhs(plan_fwd, plan_inv, tmp, psi_hat, u_hat, v_hat, dox_hat, doy_hat, nl_hat, k2,
                u_phys, v_phys, dox_phys, doy_phys, nl_work, g, nu);
    lincomb_complex<<<nb, 256>>>(tmp, omega_hat, 1.0, k2, dt * 0.5, n);
    compute_rhs(plan_fwd, plan_inv, tmp, psi_hat, u_hat, v_hat, dox_hat, doy_hat, nl_hat, k3,
                u_phys, v_phys, dox_phys, doy_phys, nl_work, g, nu);
    lincomb_complex<<<nb, 256>>>(tmp, omega_hat, 1.0, k3, dt, n);
    compute_rhs(plan_fwd, plan_inv, tmp, psi_hat, u_hat, v_hat, dox_hat, doy_hat, nl_hat, k4,
                u_phys, v_phys, dox_phys, doy_phys, nl_work, g, nu);

    axpy_complex<<<nb, 256>>>(omega_hat, k1, dt / 6.0, n);
    axpy_complex<<<nb, 256>>>(omega_hat, k2, dt / 3.0, n);
    axpy_complex<<<nb, 256>>>(omega_hat, k3, dt / 3.0, n);
    axpy_complex<<<nb, 256>>>(omega_hat, k4, dt / 6.0, n);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 256;
    double nu = argc > 2 ? atof(argv[2]) : 0.001;
    int n_steps = argc > 3 ? atoi(argv[3]) : 5000;
    double dt = argc > 4 ? atof(argv[4]) : 0.01;
    const char *ic = argc > 5 ? argv[5] : "taylor-green";
    const char *out_dir = argc > 6 ? argv[6] : "scripts/experiments/cfd-ns-bkm/results";

    if (N < 32 || (N & (N - 1)) != 0) {
        fprintf(stderr, "N must be power of 2 >= 32\n");
        return 1;
    }

    Grid g;
    g.N = N;
    g.n = (size_t)N * N;
    g.L = TWO_PI;
    g.dx = g.L / N;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("==========================================\n");
    printf("  CFD Phase 2 — 2D NS pseudospectral BKM\n");
    printf("  Device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Grid: %d x %d, nu=%.6e, dt=%.4f, steps=%d\n", N, N, nu, dt, n_steps);
    printf("  IC: %s\n", ic);
    printf("==========================================\n\n");

    CUDA_CHECK(cudaMalloc(&g.kx_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.ky_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.k2_dev, g.n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.dealias_dev, g.n * sizeof(double)));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    init_wavenumbers<<<grid, block>>>(g.kx_dev, g.ky_dev, g.k2_dev, N, g.L);
    init_dealias<<<grid, block>>>(g.dealias_dev, N, g.L);

    cufftDoubleComplex *omega_phys, *omega_hat;
    cufftDoubleComplex *psi_hat, *u_hat, *v_hat, *dox_hat, *doy_hat, *nl_hat;
    cufftDoubleComplex *k1, *k2, *k3, *k4, *tmp, *nl_work;
    cufftDoubleComplex *u_phys, *v_phys, *dox_phys, *doy_phys;

    CUDA_CHECK(cudaMalloc(&omega_phys, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&omega_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&psi_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&u_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&v_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&dox_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&doy_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&nl_hat, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&u_phys, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&v_phys, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&dox_phys, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&doy_phys, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&nl_work, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&k1, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&k2, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&k3, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&k4, g.n * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&tmp, g.n * sizeof(cufftDoubleComplex)));

    if (strcmp(ic, "random") == 0) {
        init_random_vorticity<<<grid, block>>>(omega_phys, N, g.L, 0xC0FFEEULL);
    } else {
        init_taylor_green<<<grid, block>>>(omega_phys, N, g.L);
    }

    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, N, N, CUFFT_Z2Z));
    CUFFT_CHECK(cufftPlan2d(&plan_inv, N, N, CUFFT_Z2Z));

    CUFFT_CHECK(cufftExecZ2Z(plan_fwd, omega_phys, omega_hat, CUFFT_FORWARD));
    apply_dealias<<<(g.n + 255) / 256, 256>>>(omega_hat, g.dealias_dev, (int)g.n);

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/bkm_n%d_nu%.0e_steps%d.csv", out_dir, N, nu,
             n_steps);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "CERTIFICATE_ERROR: cannot open %s\n", csv_path);
        return 2;
    }
    fprintf(csv, "step,time,max_vorticity,enstrophy,bkm_cumulative\n");

    double t0 = now_seconds();
    double t = 0.0, bkm = 0.0;
    int nan_count = 0;
    int log_stride = n_steps / 100 + 1;
    int n = (int)g.n;

    for (int step = 0; step <= n_steps; step++) {
        CUFFT_CHECK(cufftExecZ2Z(plan_inv, omega_hat, omega_phys, CUFFT_INVERSE));
        scale_complex<<<(g.n + 255) / 256, 256>>>(omega_phys, 1.0 / (double)g.n, n);
        double max_w = gpu_max_abs(omega_phys, n);
        double ens = gpu_enstrophy(omega_phys, n, g.dx);
        if (isnan(max_w) || isinf(max_w) || isnan(ens) || isinf(ens)) nan_count++;

        if (step % log_stride == 0 || step == n_steps) {
            fprintf(csv, "%d,%.6f,%.10f,%.10f,%.10f\n", step, t, max_w, ens, bkm);
        }

        if (step == n_steps) break;

        rk4_step(plan_fwd, plan_inv, omega_hat, k1, k2, k3, k4, tmp, psi_hat, u_hat, v_hat,
                 dox_hat, doy_hat, nl_hat, u_phys, v_phys, dox_phys, doy_phys, nl_work, &g, nu,
                 dt);
        apply_dealias<<<(g.n + 255) / 256, 256>>>(omega_hat, g.dealias_dev, n);
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

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    return 0;
}
