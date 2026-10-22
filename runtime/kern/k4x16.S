/* k4x16.S - AArch64 NEON fp32 conv/GEMM micro-kernel, 4 output rows x 16 columns per tile.
 *
 * void pt_k4x16(float *Y, long ldy, const float *A, const float *X, const long *off, long nk, long ntile,
 *               const float *bias)
 *   Y[i][t] = bias[i] + sum_{j<nk} A[4j+i] * X[off[j]/4 + t]     i<4, t<16*ntile   (stores, no accumulate)
 *
 * Tap-major implicit GEMM: j enumerates (tap k, input channel ci); off[j] = (ci*ldx + k*dil)*4 bytes, so
 * dilation/padding/strided channel slices cost nothing (no im2col). A is the packed [K][Cin][4] weight block.
 * Per j: 1 x 16-byte A load + 4 x 16-byte X loads feed 16 FMLA (vector x lane), i.e. 128 flops / 5 loads.
 * Accumulator acc[row i][column-vector c] lives in v(16 + 4c + i) so LD4R initialises all bias lanes at once.
 * The j loop is unrolled x2 (two independent load sets: v0-v4 and v5-v9) to hide the off[] -> address
 * dependency; d8/d9 are callee-saved and are spilled.
 *
 * x0=Y x1=ldy(bytes) x2=A x3=X x4=off x5=nk x6=ntile x7=bias
 */
    .text
    .align 4
    .global pt_k4x16
    .type   pt_k4x16, %function
pt_k4x16:
    cbz     x6, 9f
    stp     d8, d9, [sp, #-16]!
    mov     x11, x2                 /* A base */
    mov     x12, x4                 /* off base */
    add     x13, x0, x1             /* Y row 1 */
    add     x14, x13, x1            /* Y row 2 */
    add     x15, x14, x1            /* Y row 3 */
1:  /* ---- tile: init accumulators with bias */
    cbz     x7, 2f
    ld4r    {v16.4s, v17.4s, v18.4s, v19.4s}, [x7]
    b       3f
2:  movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
3:  mov     v20.16b, v16.16b
    mov     v21.16b, v17.16b
    mov     v22.16b, v18.16b
    mov     v23.16b, v19.16b
    mov     v24.16b, v16.16b
    mov     v25.16b, v17.16b
    mov     v26.16b, v18.16b
    mov     v27.16b, v19.16b
    mov     v28.16b, v16.16b
    mov     v29.16b, v17.16b
    mov     v30.16b, v18.16b
    mov     v31.16b, v19.16b
    mov     x2, x11
    mov     x4, x12
    lsr     x9, x5, #1              /* pairs */
    cbz     x9, 5f
4:  /* ---- two j per iteration */
    ldp     x16, x17, [x4], #16
    add     x16, x3, x16
    add     x17, x3, x17
    ld1     {v4.4s, v5.4s}, [x2], #32
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x16]
    ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x17]
    fmla    v16.4s, v0.4s, v4.s[0]
    fmla    v17.4s, v0.4s, v4.s[1]
    fmla    v18.4s, v0.4s, v4.s[2]
    fmla    v19.4s, v0.4s, v4.s[3]
    fmla    v20.4s, v1.4s, v4.s[0]
    fmla    v21.4s, v1.4s, v4.s[1]
    fmla    v22.4s, v1.4s, v4.s[2]
    fmla    v23.4s, v1.4s, v4.s[3]
    fmla    v24.4s, v2.4s, v4.s[0]
    fmla    v25.4s, v2.4s, v4.s[1]
    fmla    v26.4s, v2.4s, v4.s[2]
    fmla    v27.4s, v2.4s, v4.s[3]
    fmla    v28.4s, v3.4s, v4.s[0]
    fmla    v29.4s, v3.4s, v4.s[1]
    fmla    v30.4s, v3.4s, v4.s[2]
    fmla    v31.4s, v3.4s, v4.s[3]
    fmla    v16.4s, v6.4s, v5.s[0]
    fmla    v17.4s, v6.4s, v5.s[1]
    fmla    v18.4s, v6.4s, v5.s[2]
    fmla    v19.4s, v6.4s, v5.s[3]
    fmla    v20.4s, v7.4s, v5.s[0]
    fmla    v21.4s, v7.4s, v5.s[1]
    fmla    v22.4s, v7.4s, v5.s[2]
    fmla    v23.4s, v7.4s, v5.s[3]
    fmla    v24.4s, v8.4s, v5.s[0]
    fmla    v25.4s, v8.4s, v5.s[1]
    fmla    v26.4s, v8.4s, v5.s[2]
    fmla    v27.4s, v8.4s, v5.s[3]
    fmla    v28.4s, v9.4s, v5.s[0]
    fmla    v29.4s, v9.4s, v5.s[1]
    fmla    v30.4s, v9.4s, v5.s[2]
    fmla    v31.4s, v9.4s, v5.s[3]
    subs    x9, x9, #1
    b.ne    4b
5:  tbz     x5, #0, 6f              /* odd nk: one more j */
    ldr     x16, [x4], #8
    add     x16, x3, x16
    ld1     {v4.4s}, [x2], #16
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x16]
    fmla    v16.4s, v0.4s, v4.s[0]
    fmla    v17.4s, v0.4s, v4.s[1]
    fmla    v18.4s, v0.4s, v4.s[2]
    fmla    v19.4s, v0.4s, v4.s[3]
    fmla    v20.4s, v1.4s, v4.s[0]
    fmla    v21.4s, v1.4s, v4.s[1]
    fmla    v22.4s, v1.4s, v4.s[2]
    fmla    v23.4s, v1.4s, v4.s[3]
    fmla    v24.4s, v2.4s, v4.s[0]
    fmla    v25.4s, v2.4s, v4.s[1]
    fmla    v26.4s, v2.4s, v4.s[2]
    fmla    v27.4s, v2.4s, v4.s[3]
    fmla    v28.4s, v3.4s, v4.s[0]
    fmla    v29.4s, v3.4s, v4.s[1]
    fmla    v30.4s, v3.4s, v4.s[2]
    fmla    v31.4s, v3.4s, v4.s[3]
6:  /* ---- store row i = {v16+i, v20+i, v24+i, v28+i} */
    stp     q16, q20, [x0]
    stp     q24, q28, [x0, #32]
    stp     q17, q21, [x13]
    stp     q25, q29, [x13, #32]
    stp     q18, q22, [x14]
    stp     q26, q30, [x14, #32]
    stp     q19, q23, [x15]
    stp     q27, q31, [x15, #32]
    add     x0, x0, #64
    add     x13, x13, #64
    add     x14, x14, #64
    add     x15, x15, #64
    add     x3, x3, #64
    subs    x6, x6, #1
    b.ne    1b
    ldp     d8, d9, [sp], #16
9:  ret
    .size   pt_k4x16, .-pt_k4x16
