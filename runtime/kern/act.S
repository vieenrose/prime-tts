/* act.S - AArch64 NEON fp32 activation kernels (in place on one row segment, n % 4 == 0).
 *
 *   void pt_snake_row(float *y, long n, float a, float ia);   y += sin(a*y)^2 * ia
 *   void pt_silu_row(float *y, long n);                       y  = y / (1 + exp(-y))
 *   void pt_tanh_row(float *y, long n);                       y  = sign(y) (1 - e) / (1 + e),  e = exp(-2|y|)
 *   void pt_sigmul_row(float *y, const float *g, long n);     y  = y / (1 + exp(-g))           (GLU gate)
 *
 * sin: Cody-Waite reduction by pi (3-part constant, fused FMLS) to r in [-pi/2, pi/2], odd Taylor
 * polynomial to r^11 (|err| < 6e-8 on the interval). Because only sin^2 is needed the parity of k is
 * irrelevant (sin(r + k pi)^2 == sin(r)^2), which removes the sign fix-up entirely.
 * exp: Cephes expf scheme: clamp to [-87.3, 88.3], n = round(x log2 e), 2-part ln2 reduction,
 * degree-5 polynomial, scale by integer add into the exponent field.
 * Two 4-lane vectors per iteration when n >= 8 to give the out-of-order core independent chains.
 */
    .section .rodata
    .align 4
.Lsin_c:
    .float 0.31830988618379067      /* 1/pi */
    .float 3.140625                 /* pi = PA + PB + PC (PA, PB exact in few bits) */
    .float 9.67502593994140625e-4
    .float 1.509957990978376432e-7
    .float -1.6666666666666666e-1   /* -1/3! */
    .float 8.3333333333333333e-3    /*  1/5! */
    .float -1.9841269841269841e-4   /* -1/7! */
    .float 2.7557319223985893e-6    /*  1/9! */
    .float -2.5052108385441720e-8   /* -1/11! */
.Lexp_c:
    .float 88.3762626647949         /* hi clamp */
    .float -87.3365447504019        /* lo clamp */
    .float 1.44269504088896341      /* log2 e */
    .float 0.693359375              /* C1 */
    .float -2.12194440e-4           /* C2 */
    .float 1.9875691500e-4          /* p0 */
    .float 1.3981999507e-3
    .float 8.3334519073e-3
    .float 4.1665795894e-2
    .float 1.6666665459e-1
    .float 5.0000001201e-1          /* p5 */
    .float 1.0

    .text
/* ---------------------------------------------------------------- exp constants -> v16..v27 */
.macro LOAD_EXP_CONSTS
    adr     x9, .Lexp_c
    ld1r    {v16.4s}, [x9], #4
    ld1r    {v17.4s}, [x9], #4
    ld1r    {v18.4s}, [x9], #4
    ld1r    {v19.4s}, [x9], #4
    ld1r    {v20.4s}, [x9], #4
    ld1r    {v21.4s}, [x9], #4
    ld1r    {v22.4s}, [x9], #4
    ld1r    {v23.4s}, [x9], #4
    ld1r    {v24.4s}, [x9], #4
    ld1r    {v25.4s}, [x9], #4
    ld1r    {v26.4s}, [x9], #4
    ld1r    {v27.4s}, [x9], #4
.endm

/* d = exp(s); clobbers t0, t1, t2 (s preserved unless d == s) */
.macro EXPV d, s, t0, t1, t2
    fmin    \t0\().4s, \s\().4s, v16.4s
    fmax    \t0\().4s, \t0\().4s, v17.4s          /* x */
    fmul    \t1\().4s, \t0\().4s, v18.4s
    frintn  \t1\().4s, \t1\().4s                  /* k */
    fmls    \t0\().4s, \t1\().4s, v19.4s
    fmls    \t0\().4s, \t1\().4s, v20.4s          /* r */
    fcvtzs  \t1\().4s, \t1\().4s
    shl     \t1\().4s, \t1\().4s, #23             /* k << 23 */
    mov     \t2\().16b, v22.16b
    fmla    \t2\().4s, \t0\().4s, v21.4s          /* p0 r + p1 */
    mov     \d\().16b, v23.16b
    fmla    \d\().4s, \t2\().4s, \t0\().4s
    mov     \t2\().16b, v24.16b
    fmla    \t2\().4s, \d\().4s, \t0\().4s
    mov     \d\().16b, v25.16b
    fmla    \d\().4s, \t2\().4s, \t0\().4s
    mov     \t2\().16b, v26.16b
    fmla    \t2\().4s, \d\().4s, \t0\().4s        /* poly */
    fmul    \d\().4s, \t0\().4s, \t0\().4s        /* r^2 */
    fmul    \d\().4s, \d\().4s, \t2\().4s         /* poly r^2 */
    fadd    \d\().4s, \d\().4s, \t0\().4s
    fadd    \d\().4s, \d\().4s, v27.4s            /* + r + 1 */
    add     \d\().4s, \d\().4s, \t1\().4s         /* * 2^k */
.endm

/* ---------------------------------------------------------------- snake */
.macro SNAKE y, z, k, r2, p, q
    fmul    \z\().4s, \y\().4s, v30.4s            /* z = a y */
    fmul    \k\().4s, \z\().4s, v20.4s
    frintn  \k\().4s, \k\().4s
    fmls    \z\().4s, \k\().4s, v21.4s
    fmls    \z\().4s, \k\().4s, v22.4s
    fmls    \z\().4s, \k\().4s, v23.4s            /* r */
    fmul    \r2\().4s, \z\().4s, \z\().4s
    mov     \p\().16b, v27.16b
    fmla    \p\().4s, \r2\().4s, v28.4s
    mov     \q\().16b, v26.16b
    fmla    \q\().4s, \r2\().4s, \p\().4s
    mov     \p\().16b, v25.16b
    fmla    \p\().4s, \r2\().4s, \q\().4s
    mov     \q\().16b, v24.16b
    fmla    \q\().4s, \r2\().4s, \p\().4s         /* c3 + r2(...) */
    fmul    \q\().4s, \q\().4s, \r2\().4s
    fmla    \z\().4s, \z\().4s, \q\().4s          /* s = r + r r2 P */
    fmul    \z\().4s, \z\().4s, \z\().4s          /* s^2 */
    fmla    \y\().4s, \z\().4s, v31.4s            /* y += s^2 ia */
.endm


/* four independent 4-lane snakes interleaved op by op: y = v0..v3, z = v4..v7, k = v16..v19, r2/p/q reuse */
.macro SNAKE4
    fmul    v4.4s, v0.4s, v30.4s
    fmul    v5.4s, v1.4s, v30.4s
    fmul    v6.4s, v2.4s, v30.4s
    fmul    v7.4s, v3.4s, v30.4s
    fmul    v16.4s, v4.4s, v20.4s
    fmul    v17.4s, v5.4s, v20.4s
    fmul    v18.4s, v6.4s, v20.4s
    fmul    v19.4s, v7.4s, v20.4s
    frintn  v16.4s, v16.4s
    frintn  v17.4s, v17.4s
    frintn  v18.4s, v18.4s
    frintn  v19.4s, v19.4s
    fmls    v4.4s, v16.4s, v21.4s
    fmls    v5.4s, v17.4s, v21.4s
    fmls    v6.4s, v18.4s, v21.4s
    fmls    v7.4s, v19.4s, v21.4s
    fmls    v4.4s, v16.4s, v22.4s
    fmls    v5.4s, v17.4s, v22.4s
    fmls    v6.4s, v18.4s, v22.4s
    fmls    v7.4s, v19.4s, v22.4s
    fmls    v4.4s, v16.4s, v23.4s
    fmls    v5.4s, v17.4s, v23.4s
    fmls    v6.4s, v18.4s, v23.4s
    fmls    v7.4s, v19.4s, v23.4s           /* r */
    fmul    v16.4s, v4.4s, v4.4s
    fmul    v17.4s, v5.4s, v5.4s
    fmul    v18.4s, v6.4s, v6.4s
    fmul    v19.4s, v7.4s, v7.4s            /* r2 */
    mov     v8.16b, v27.16b
    mov     v9.16b, v27.16b
    mov     v10.16b, v27.16b
    mov     v11.16b, v27.16b
    fmla    v8.4s, v16.4s, v28.4s
    fmla    v9.4s, v17.4s, v28.4s
    fmla    v10.4s, v18.4s, v28.4s
    fmla    v11.4s, v19.4s, v28.4s
    mov     v12.16b, v26.16b
    mov     v13.16b, v26.16b
    mov     v14.16b, v26.16b
    mov     v15.16b, v26.16b
    fmla    v12.4s, v16.4s, v8.4s
    fmla    v13.4s, v17.4s, v9.4s
    fmla    v14.4s, v18.4s, v10.4s
    fmla    v15.4s, v19.4s, v11.4s
    mov     v8.16b, v25.16b
    mov     v9.16b, v25.16b
    mov     v10.16b, v25.16b
    mov     v11.16b, v25.16b
    fmla    v8.4s, v16.4s, v12.4s
    fmla    v9.4s, v17.4s, v13.4s
    fmla    v10.4s, v18.4s, v14.4s
    fmla    v11.4s, v19.4s, v15.4s
    mov     v12.16b, v24.16b
    mov     v13.16b, v24.16b
    mov     v14.16b, v24.16b
    mov     v15.16b, v24.16b
    fmla    v12.4s, v16.4s, v8.4s
    fmla    v13.4s, v17.4s, v9.4s
    fmla    v14.4s, v18.4s, v10.4s
    fmla    v15.4s, v19.4s, v11.4s
    fmul    v12.4s, v12.4s, v16.4s
    fmul    v13.4s, v13.4s, v17.4s
    fmul    v14.4s, v14.4s, v18.4s
    fmul    v15.4s, v15.4s, v19.4s
    fmla    v4.4s, v4.4s, v12.4s
    fmla    v5.4s, v5.4s, v13.4s
    fmla    v6.4s, v6.4s, v14.4s
    fmla    v7.4s, v7.4s, v15.4s            /* s */
    fmul    v4.4s, v4.4s, v4.4s
    fmul    v5.4s, v5.4s, v5.4s
    fmul    v6.4s, v6.4s, v6.4s
    fmul    v7.4s, v7.4s, v7.4s
    fmla    v0.4s, v4.4s, v31.4s
    fmla    v1.4s, v5.4s, v31.4s
    fmla    v2.4s, v6.4s, v31.4s
    fmla    v3.4s, v7.4s, v31.4s
.endm

    .align 4
    .global pt_snake_row
    .type   pt_snake_row, %function
pt_snake_row:
    cbz     x1, 10f
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    dup     v30.4s, v0.s[0]
    dup     v31.4s, v1.s[0]
    adr     x9, .Lsin_c
    ld1r    {v20.4s}, [x9], #4
    ld1r    {v21.4s}, [x9], #4
    ld1r    {v22.4s}, [x9], #4
    ld1r    {v23.4s}, [x9], #4
    ld1r    {v24.4s}, [x9], #4
    ld1r    {v25.4s}, [x9], #4
    ld1r    {v26.4s}, [x9], #4
    ld1r    {v27.4s}, [x9], #4
    ld1r    {v28.4s}, [x9], #4
1:  cmp     x1, #16
    b.lt    2f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
    SNAKE4
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    sub     x1, x1, #16
    cbnz    x1, 1b
    b       9f
2:  ld1     {v0.4s}, [x0]
    SNAKE   v0, v4, v5, v6, v7, v16
    st1     {v0.4s}, [x0], #16
    subs    x1, x1, #4
    b.gt    2b
9:  ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
10: ret
    .size   pt_snake_row, .-pt_snake_row

/* ---------------------------------------------------------------- silu */
    .align 4
    .global pt_silu_row
    .type   pt_silu_row, %function
pt_silu_row:
    cbz     x1, 9f
    LOAD_EXP_CONSTS
1:  cmp     x1, #8
    b.lt    2f
    ld1     {v0.4s, v1.4s}, [x0]
    fneg    v2.4s, v0.4s
    fneg    v3.4s, v1.4s
    EXPV    v4, v2, v5, v6, v7
    EXPV    v28, v3, v29, v30, v31
    fadd    v4.4s, v4.4s, v27.4s
    fadd    v28.4s, v28.4s, v27.4s
    fdiv    v0.4s, v0.4s, v4.4s
    fdiv    v1.4s, v1.4s, v28.4s
    st1     {v0.4s, v1.4s}, [x0], #32
    sub     x1, x1, #8
    cbnz    x1, 1b
    ret
2:  ld1     {v0.4s}, [x0]
    fneg    v2.4s, v0.4s
    EXPV    v4, v2, v5, v6, v7
    fadd    v4.4s, v4.4s, v27.4s
    fdiv    v0.4s, v0.4s, v4.4s
    st1     {v0.4s}, [x0], #16
    subs    x1, x1, #4
    b.gt    1b
9:  ret
    .size   pt_silu_row, .-pt_silu_row

/* ---------------------------------------------------------------- sigmul (GLU) */
    .align 4
    .global pt_sigmul_row
    .type   pt_sigmul_row, %function
pt_sigmul_row:
    cbz     x2, 9f
    LOAD_EXP_CONSTS
1:  ld1     {v0.4s}, [x0]
    ld1     {v1.4s}, [x1], #16
    fneg    v2.4s, v1.4s
    EXPV    v4, v2, v5, v6, v7
    fadd    v4.4s, v4.4s, v27.4s
    fdiv    v0.4s, v0.4s, v4.4s
    st1     {v0.4s}, [x0], #16
    subs    x2, x2, #4
    b.gt    1b
9:  ret
    .size   pt_sigmul_row, .-pt_sigmul_row

/* ---------------------------------------------------------------- tanh */
    .align 4
    .global pt_tanh_row
    .type   pt_tanh_row, %function
pt_tanh_row:
    cbz     x1, 9f
    LOAD_EXP_CONSTS
    movi    v28.4s, #0x80, lsl #24   /* sign mask */
1:  ld1     {v0.4s}, [x0]
    fabs    v1.4s, v0.4s
    fadd    v1.4s, v1.4s, v1.4s
    fneg    v1.4s, v1.4s             /* -2|y| */
    EXPV    v2, v1, v5, v6, v7       /* e */
    fsub    v3.4s, v27.4s, v2.4s     /* 1 - e */
    fadd    v4.4s, v27.4s, v2.4s     /* 1 + e */
    fdiv    v3.4s, v3.4s, v4.4s
    and     v0.16b, v0.16b, v28.16b
    orr     v0.16b, v0.16b, v3.16b
    st1     {v0.4s}, [x0], #16
    subs    x1, x1, #4
    b.gt    1b
9:  ret
    .size   pt_tanh_row, .-pt_tanh_row
