/* dwln.S - AArch64 NEON depthwise conv row + LayerNorm(+residual) over channel-major blocks.
 *
 * void pt_dw_row(float *y, const float *x, long n, const float *w, long K, float b)
 *     y[t] = b + sum_k w[k] x[t+k]   (x points at column -pad), n % 4 == 0; 16 outputs per iteration.
 * void pt_ln_res(float *x, const float *r, float *y, long C, long ld, long n, const float *g, const float *b,
 *                float eps)
 *     per column t: if r: x[c][t] += r[c][t];  y[c][t] = (x - mean) / sqrt(var + eps) * g[c] + b[c]
 *     4 columns per iteration, three passes down the C rows (2.7 KB per column group stays in L1).
 */
    .text
    .align 4
    .global pt_dw_row
    .type   pt_dw_row, %function
/* x0=y x1=x x2=n x3=w x4=K s0=b */
pt_dw_row:
    cbz     x2, 9f
    dup     v31.4s, v0.s[0]
1:  cmp     x2, #16
    b.lt    3f
    mov     v16.16b, v31.16b
    mov     v17.16b, v31.16b
    mov     v18.16b, v31.16b
    mov     v19.16b, v31.16b
    mov     x5, x3
    mov     x6, x1
    mov     x7, x4
2:  ld1r    {v4.4s}, [x5], #4
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x6]
    add     x6, x6, #4
    fmla    v16.4s, v0.4s, v4.4s
    fmla    v17.4s, v1.4s, v4.4s
    fmla    v18.4s, v2.4s, v4.4s
    fmla    v19.4s, v3.4s, v4.4s
    subs    x7, x7, #1
    b.ne    2b
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add     x1, x1, #64
    sub     x2, x2, #16
    cbnz    x2, 1b
    ret
3:  mov     v16.16b, v31.16b
    mov     x5, x3
    mov     x6, x1
    mov     x7, x4
4:  ld1r    {v4.4s}, [x5], #4
    ld1     {v0.4s}, [x6]
    add     x6, x6, #4
    fmla    v16.4s, v0.4s, v4.4s
    subs    x7, x7, #1
    b.ne    4b
    st1     {v16.4s}, [x0], #16
    add     x1, x1, #16
    subs    x2, x2, #4
    b.gt    3b
9:  ret
    .size   pt_dw_row, .-pt_dw_row

    .align 4
    .global pt_ln_res
    .type   pt_ln_res, %function
/* x0=x x1=r x2=y x3=C x4=ld x5=n x6=g x7=b s0=eps */
pt_ln_res:
    cbz     x5, 9f
    cbz     x3, 9f
    dup     v31.4s, v0.s[0]          /* eps */
    scvtf   s30, x3
    dup     v30.4s, v30.s[0]         /* (float)C */
    fmov    v29.4s, #1.0
    lsl     x4, x4, #2               /* row stride bytes */
1:  /* pass 1: x += r ; sum */
    movi    v16.4s, #0
    mov     x9, x0
    mov     x10, x1
    mov     x11, x3
2:  ld1     {v0.4s}, [x9]
    cbz     x1, 3f
    ld1     {v1.4s}, [x10]
    add     x10, x10, x4
    fadd    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x9]
3:  fadd    v16.4s, v16.4s, v0.4s
    add     x9, x9, x4
    subs    x11, x11, #1
    b.ne    2b
    fdiv    v16.4s, v16.4s, v30.4s   /* mean */
    /* pass 2: var */
    movi    v17.4s, #0
    mov     x9, x0
    mov     x11, x3
4:  ld1     {v0.4s}, [x9]
    add     x9, x9, x4
    fsub    v0.4s, v0.4s, v16.4s
    fmla    v17.4s, v0.4s, v0.4s
    subs    x11, x11, #1
    b.ne    4b
    fdiv    v17.4s, v17.4s, v30.4s
    fadd    v17.4s, v17.4s, v31.4s
    fsqrt   v17.4s, v17.4s
    fdiv    v17.4s, v29.4s, v17.4s   /* inv */
    /* pass 3: y = (x - mean) * inv * g + b */
    mov     x9, x0
    mov     x12, x2
    mov     x13, x6
    mov     x14, x7
    mov     x11, x3
5:  ld1     {v0.4s}, [x9]
    add     x9, x9, x4
    ld1r    {v2.4s}, [x13], #4
    ld1r    {v3.4s}, [x14], #4
    fsub    v0.4s, v0.4s, v16.4s
    fmul    v0.4s, v0.4s, v17.4s
    fmul    v0.4s, v0.4s, v2.4s
    fadd    v0.4s, v0.4s, v3.4s
    st1     {v0.4s}, [x12]
    add     x12, x12, x4
    subs    x11, x11, #1
    b.ne    5b
    add     x0, x0, #16
    cbz     x1, 6f
    add     x1, x1, #16
6:  add     x2, x2, #16
    subs    x5, x5, #4
    b.gt    1b
9:  ret
    .size   pt_ln_res, .-pt_ln_res
