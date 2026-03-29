#ifndef QD_REAL_H
#define QD_REAL_H

#include <math.h>

/* ================================================================
 * Quad-double arithmetic for CUDA
 *
 * A qd_real is an unevaluated sum of 4 doubles: x = x[0]+x[1]+x[2]+x[3]
 * with |x[1]| <= eps*|x[0]|, |x[2]| <= eps*|x[1]|, etc.
 * This gives ~212 bits (~62 decimal digits) of precision.
 *
 * Based on: Hida, Li, Bailey (2001)
 * "Library for Double-Double and Quad-Double Arithmetic"
 * ================================================================ */

typedef struct { double x[4]; } qd_real;

/* ---- Two-Sum and Two-Prod primitives ---- */

__host__ __device__ inline void two_sum(double a, double b, double *s, double *e) {
    *s = a + b;
    double v = *s - a;
    *e = (a - (*s - v)) + (b - v);
}

__host__ __device__ inline void two_prod(double a, double b, double *p, double *e) {
    *p = a * b;
    *e = fma(a, b, -(*p));
}

/* ---- Double-double addition: (a0+a1) + (b0+b1) = (s0+s1) ---- */

__host__ __device__ inline void dd_add(double a0, double a1, double b0, double b1,
                                        double *s0, double *s1) {
    double t1, t2, e;
    two_sum(a0, b0, &t1, &t2);
    t2 += a1 + b1;
    two_sum(t1, t2, s0, &e);
    *s1 = e;
}

/* ---- qd_real constructors ---- */

__host__ __device__ inline qd_real qd_from_double(double a) {
    qd_real r; r.x[0]=a; r.x[1]=0; r.x[2]=0; r.x[3]=0; return r;
}

__host__ __device__ inline qd_real qd_from_int(long long n) {
    return qd_from_double((double)n);
}

/* ---- Renormalize: ensure non-overlapping property ---- */

__host__ __device__ inline qd_real qd_renorm(double c0, double c1, double c2,
                                              double c3, double c4) {
    double s, t0, t1, t2, t3;
    qd_real r;

    two_sum(c3, c4, &s, &t3);
    two_sum(c2, s, &s, &t2);
    two_sum(c1, s, &s, &t1);
    two_sum(c0, s, &r.x[0], &t0);

    two_sum(t1, t2, &s, &t1);
    two_sum(t0, s, &r.x[1], &t0);

    two_sum(t0, t1, &r.x[2], &t0);
    r.x[3] = t0 + t3;

    return r;
}

/* ---- Addition ---- */

__host__ __device__ inline qd_real qd_add(qd_real a, qd_real b) {
    /* Merge-sort-like addition of 8 components, then renormalize */
    int ia = 0, ib = 0;
    double u[8];
    /* Interleave by magnitude (approximate — use indices) */
    for (int i = 0; i < 4; i++) { u[2*i] = a.x[i]; u[2*i+1] = b.x[i]; }

    /* Cascade two-sum from bottom */
    double s, e;
    double c[5] = {0, 0, 0, 0, 0};

    two_sum(a.x[0], b.x[0], &c[0], &e);
    double t = e;
    two_sum(a.x[1], b.x[1], &s, &e);
    double t2;
    two_sum(t, s, &c[1], &t2);
    t = t2 + e;
    two_sum(a.x[2], b.x[2], &s, &e);
    two_sum(t, s, &c[2], &t2);
    t = t2 + e;
    two_sum(a.x[3], b.x[3], &s, &e);
    two_sum(t, s, &c[3], &t2);
    c[4] = t2 + e;

    return qd_renorm(c[0], c[1], c[2], c[3], c[4]);
}

__host__ __device__ inline qd_real qd_neg(qd_real a) {
    qd_real r;
    r.x[0] = -a.x[0]; r.x[1] = -a.x[1];
    r.x[2] = -a.x[2]; r.x[3] = -a.x[3];
    return r;
}

__host__ __device__ inline qd_real qd_sub(qd_real a, qd_real b) {
    return qd_add(a, qd_neg(b));
}

/* ---- Multiplication ---- */

__host__ __device__ inline qd_real qd_mul(qd_real a, qd_real b) {
    double p0, p1, p2, p3, p4, p5;
    double q0, q1, q2, q3, q4, q5;
    double t0, t1;

    two_prod(a.x[0], b.x[0], &p0, &q0);
    two_prod(a.x[0], b.x[1], &p1, &q1);
    two_prod(a.x[1], b.x[0], &p2, &q2);
    two_prod(a.x[0], b.x[2], &p3, &q3);
    two_prod(a.x[1], b.x[1], &p4, &q4);
    two_prod(a.x[2], b.x[0], &p5, &q5);

    /* Accumulate from bottom */
    two_sum(p1, p2, &p1, &p2);
    two_sum(q0, p1, &t0, &t1);

    double r1 = t0;
    double c2 = t1 + p2;

    two_sum(p3, p4, &t0, &t1);
    double t2 = t1;
    two_sum(t0, p5, &t0, &t1);
    t2 += t1;
    two_sum(c2, t0, &c2, &t0);
    t2 += t0;

    double c3 = t2 + q1 + q2 + q3 + q4 + q5
                + a.x[0]*b.x[3] + a.x[1]*b.x[2]
                + a.x[2]*b.x[1] + a.x[3]*b.x[0];

    return qd_renorm(p0, r1, c2, c3, 0.0);
}

/* ---- Division: a / b using Newton iteration ---- */

__host__ __device__ inline qd_real qd_div(qd_real a, qd_real b) {
    /* Compute q = a/b using long division */
    double q0 = a.x[0] / b.x[0];
    qd_real r = qd_sub(a, qd_mul(qd_from_double(q0), b));

    double q1 = r.x[0] / b.x[0];
    r = qd_sub(r, qd_mul(qd_from_double(q1), b));

    double q2 = r.x[0] / b.x[0];
    r = qd_sub(r, qd_mul(qd_from_double(q2), b));

    double q3 = r.x[0] / b.x[0];

    return qd_renorm(q0, q1, q2, q3, 0.0);
}

/* ---- Comparison ---- */

__host__ __device__ inline int qd_gt(qd_real a, qd_real b) {
    if (a.x[0] != b.x[0]) return a.x[0] > b.x[0];
    if (a.x[1] != b.x[1]) return a.x[1] > b.x[1];
    if (a.x[2] != b.x[2]) return a.x[2] > b.x[2];
    return a.x[3] > b.x[3];
}

__host__ __device__ inline int qd_lt_zero(qd_real a) { return a.x[0] < 0.0; }

__host__ __device__ inline double qd_to_double(qd_real a) { return a.x[0] + a.x[1]; }

/* ---- Absolute value ---- */

__host__ __device__ inline qd_real qd_abs(qd_real a) {
    return qd_lt_zero(a) ? qd_neg(a) : a;
}

/* ---- Constants ---- */

/* π to ~62 decimal digits as a quad-double.
 * These are the exact double decomposition of:
 * 3.14159265358979323846264338327950288419716939937510...
 */
__host__ __device__ inline qd_real qd_pi() {
    qd_real r;
    r.x[0] = 3.141592653589793116e+00;
    r.x[1] = 1.224646799147353207e-16;
    r.x[2] = -2.994769809718339666e-33;
    r.x[3] = 1.112454220863365282e-49;
    return r;
}

/* 2π */
__host__ __device__ inline qd_real qd_two_pi() {
    qd_real r;
    r.x[0] = 6.283185307179586232e+00;
    r.x[1] = 2.449293598294706414e-16;
    r.x[2] = -5.989539619436679332e-33;
    r.x[3] = 2.224908441726730563e-49;
    return r;
}

/* ---- Multiply qd by integer ---- */

__host__ __device__ inline qd_real qd_mul_int(qd_real a, long long n) {
    return qd_mul(a, qd_from_double((double)n));
}

/* ---- sin via argument reduction + Taylor series ---- */

__host__ __device__ inline qd_real qd_sin(qd_real a) {
    /* Argument reduction: compute a mod 2π, then reduce to [-π, π] */
    qd_real two_pi = qd_two_pi();
    qd_real pi = qd_pi();

    /* k = round(a / (2π)) */
    double k_d = round(a.x[0] / two_pi.x[0]);
    long long k = (long long)k_d;

    /* r = a - k * 2π */
    qd_real r = qd_sub(a, qd_mul_int(two_pi, k));

    /* Further reduce: if r > π, r -= 2π; if r < -π, r += 2π */
    if (qd_gt(r, pi)) r = qd_sub(r, two_pi);
    if (qd_lt_zero(qd_add(r, pi))) r = qd_add(r, two_pi);

    /* Now |r| <= π. Use range reduction to |r| <= π/4 via identities:
     * For simplicity, just use Taylor series directly (r is usually small
     * for our use case since we're evaluating at integers near multiples of π).
     */

    /* Taylor series: sin(r) = r - r³/3! + r⁵/5! - r⁷/7! + ...
     * Converges fast when |r| < π. We need ~20 terms for 62-digit precision.
     */
    qd_real r2 = qd_mul(r, r);
    qd_real term = r;
    qd_real sum = r;

    for (int i = 1; i <= 25; i++) {
        double denom = -(2.0*i) * (2.0*i + 1.0);
        term = qd_mul(term, r2);
        term = qd_div(term, qd_from_double(denom));
        sum = qd_add(sum, term);

        /* Early termination if term is negligible */
        if (fabs(term.x[0]) < 1e-60 * fabs(sum.x[0])) break;
    }

    return sum;
}

#endif /* QD_REAL_H */
