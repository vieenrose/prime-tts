/* k8x12.S - AArch64 NEON fp32 conv/GEMM micro-kernel, 8 output rows x 12 columns per tile.
 *
 * void pt_k8x12(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
 *               const float *bias)
 * Same contract as pt_k4x16 but two consecutive packed 4-row blocks (A0 = A, A1 = A + 4*nk floats) and
 * 12 columns per tile.  Per j: 2 A loads + 3 X loads feed 24 FMLA (192 flops / 5 loads, vs 128/5 for
 * 4x16).  acc[row i][column-vector c] = v(8 + 8c + i): 24 accumulators (v8-v31, d8-d15 spilled);
 * register renaming covers the reuse of v0-v4 across iterations.
 * x0=Y x1=ldy(bytes) x2=A x3=X x4=off x5=nk x6=ntile x7=bias
 */
    .text
    .align 4
    .global pt_k8x12
    .type   pt_k8x12, %function
pt_k8x12:
    cbz     x6, 9f
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    mov     x11, x2                 /* A0 base */
    add     x10, x2, x5, lsl #4     /* A1 base = A + nk*16 bytes */
    mov     x12, x4
    mov     x15, x10
1:  cbz     x7, 2f
    add     x9, x7, #16
    ld4r    {v8.4s, v9.4s, v10.4s, v11.4s}, [x7]
    ld4r    {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    b       3f
2:
    movi    v8.4s, #0
    movi    v9.4s, #0
    movi    v10.4s, #0
    movi    v11.4s, #0
    movi    v12.4s, #0
    movi    v13.4s, #0
    movi    v14.4s, #0
    movi    v15.4s, #0
3:
    mov     v16.16b, v8.16b
    mov     v24.16b, v8.16b
    mov     v17.16b, v9.16b
    mov     v25.16b, v9.16b
    mov     v18.16b, v10.16b
    mov     v26.16b, v10.16b
    mov     v19.16b, v11.16b
    mov     v27.16b, v11.16b
    mov     v20.16b, v12.16b
    mov     v28.16b, v12.16b
    mov     v21.16b, v13.16b
    mov     v29.16b, v13.16b
    mov     v22.16b, v14.16b
    mov     v30.16b, v14.16b
    mov     v23.16b, v15.16b
    mov     v31.16b, v15.16b
    mov     x2, x11
    mov     x10, x15
    mov     x4, x12
    mov     x9, x5
4:  ldr     x16, [x4], #8
    add     x16, x3, x16
    ld1     {v3.4s}, [x2], #16
    ld1     {v4.4s}, [x10], #16
    ld1     {v0.4s, v1.4s, v2.4s}, [x16]
    fmla    v8.4s, v0.4s, v3.s[0]
    fmla    v9.4s, v0.4s, v3.s[1]
    fmla    v10.4s, v0.4s, v3.s[2]
    fmla    v11.4s, v0.4s, v3.s[3]
    fmla    v12.4s, v0.4s, v4.s[0]
    fmla    v13.4s, v0.4s, v4.s[1]
    fmla    v14.4s, v0.4s, v4.s[2]
    fmla    v15.4s, v0.4s, v4.s[3]
    fmla    v16.4s, v1.4s, v3.s[0]
    fmla    v17.4s, v1.4s, v3.s[1]
    fmla    v18.4s, v1.4s, v3.s[2]
    fmla    v19.4s, v1.4s, v3.s[3]
    fmla    v20.4s, v1.4s, v4.s[0]
    fmla    v21.4s, v1.4s, v4.s[1]
    fmla    v22.4s, v1.4s, v4.s[2]
    fmla    v23.4s, v1.4s, v4.s[3]
    fmla    v24.4s, v2.4s, v3.s[0]
    fmla    v25.4s, v2.4s, v3.s[1]
    fmla    v26.4s, v2.4s, v3.s[2]
    fmla    v27.4s, v2.4s, v3.s[3]
    fmla    v28.4s, v2.4s, v4.s[0]
    fmla    v29.4s, v2.4s, v4.s[1]
    fmla    v30.4s, v2.4s, v4.s[2]
    fmla    v31.4s, v2.4s, v4.s[3]
    subs    x9, x9, #1
    b.ne    4b
    mov     x13, x0
    stp     q8, q16, [x13]
    str     q24, [x13, #32]
    add     x13, x13, x1
    stp     q9, q17, [x13]
    str     q25, [x13, #32]
    add     x13, x13, x1
    stp     q10, q18, [x13]
    str     q26, [x13, #32]
    add     x13, x13, x1
    stp     q11, q19, [x13]
    str     q27, [x13, #32]
    add     x13, x13, x1
    stp     q12, q20, [x13]
    str     q28, [x13, #32]
    add     x13, x13, x1
    stp     q13, q21, [x13]
    str     q29, [x13, #32]
    add     x13, x13, x1
    stp     q14, q22, [x13]
    str     q30, [x13, #32]
    add     x13, x13, x1
    stp     q15, q23, [x13]
    str     q31, [x13, #32]
    add     x0, x0, #48
    add     x3, x3, #48
    subs    x6, x6, #1
    b.ne    1b
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
9:  ret
    .size   pt_k8x12, .-pt_k8x12
