use crate::utils::ele_ij;

use core::arch::x86_64::{
    __m256d, _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd,
};

#[target_feature(enable = "avx2")]
#[allow(non_snake_case)]
pub unsafe fn gemm_4x4_kernel(
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    // Declare vector registers to hold 4x4 Columns and load them
    let mut c_0123_0: __m256d = _mm256_loadu_pd(&c[ele_ij(0, 0, ld_c)]);
    let mut c_0123_1: __m256d = _mm256_loadu_pd(&c[ele_ij(0, 1, ld_c)]);
    let mut c_0123_2: __m256d = _mm256_loadu_pd(&c[ele_ij(0, 2, ld_c)]);
    let mut c_0123_3: __m256d = _mm256_loadu_pd(&c[ele_ij(0, 3, ld_c)]);

    for p in 0..k {
        // declare vector register for loading/broadcasting b[p, j]
        let mut b_pj: __m256d;

        // declare a vector register to hold the current column of a and load it with four elements of that column
        let a_0123_p: __m256d = _mm256_loadu_pd(&a[ele_ij(0, p, ld_a)]);

        // Load/broadcast beta( p,0 )
        b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 0, ld_b)]);

        // update the first column of c with the current element of a times b[p, 0]
        c_0123_0 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_0);

        // Load/broadcast b[p,1]
        b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 1, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_1 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_1);

        // Load/broadcast b[p,2]
        b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 2, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_2 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_2);

        // Load/broadcast b[p,3]
        b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 3, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_3 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_3);
    }

    //store the updated results
    _mm256_storeu_pd(&mut c[ele_ij(0, 0, ld_c)], c_0123_0);
    _mm256_storeu_pd(&mut c[ele_ij(0, 1, ld_c)], c_0123_1);
    _mm256_storeu_pd(&mut c[ele_ij(0, 2, ld_c)], c_0123_2);
    _mm256_storeu_pd(&mut c[ele_ij(0, 3, ld_c)], c_0123_3);
}
