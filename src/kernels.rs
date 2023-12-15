use crate::utils::at;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256d, _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd,
};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(non_snake_case)]
/// # Safety
///
/// This is a kernel used for FMA (Fused multiply add) operaton to compute matrix product
/// command to list features: rustc --print=cfg -C target-cpu=native
pub unsafe fn gemm_4x4_kernel_arm(
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &[f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    // Declare vector registers to hold 4x4 Columns and load them

    use std::{
        arch::aarch64::{
            float64x1_t, float64x1x4_t, vfma_f64, vfma_lane_f64, vfma_n_f64, vld4_f64, vmla_f64,
        },
        simd::Simd,
    };
    let mut c_0123_0: float64x1x4_t = vld4_f64(&c[at(0, 0, ld_c)]);
    // let mut c_0123_1: float64x1x4_t = vld4_f64(&c[at(0, 1, ld_c)]);
    // let mut c_0123_2: float64x1x4_t = vld4_f64(&c[at(0, 2, ld_c)]);
    // let mut c_0123_3: float64x1x4_t = vld4_f64(&c[at(0, 3, ld_c)]);

    for p in 0..k {
        // declare a vector register to hold the current column of a and load it with four elements of that column
        let a_0123_p: float64x1x4_t = vld4_f64(&a[at(0, p, ld_a)]);

        // declare vector register for loading/broadcasting b[p, j]
        // Load b(p,0)
        let lane: float64x1_t = float64x1_t::from(Simd::from([b[at(p, 0, ld_b)]]));
        // broadcast b(p, 0)
        let b_pj = float64x1x4_t(lane, lane, lane, lane);

        // update the first column of c with the current element of a times b[p, 0]
        println!(
            "c = {:?}\na = {:?}\nb = {:?}",
            c_0123_0.0, a_0123_p.0, b_pj.0
        );
        let caca = vmla_f64(c_0123_0.0, a_0123_p.0, b_pj.0);
        println!("fma : {:?}", caca);
        // println!("c -->: {:?}", c_0123_0.0);

        //     // Load/broadcast b[p,1]
        //     b_pj = _mm256_broadcast_sd(&b[at(p, 1, ld_b)]);

        //     // update the second column of C with the current column of A times  b[p,1]
        //     c_0123_1 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_1);

        //     // Load/broadcast b[p,2]
        //     b_pj = _mm256_broadcast_sd(&b[at(p, 2, ld_b)]);

        //     // update the second column of C with the current column of A times  b[p,1]
        //     c_0123_2 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_2);

        //     // Load/broadcast b[p,3]
        //     b_pj = _mm256_broadcast_sd(&b[at(p, 3, ld_b)]);

        //     // update the second column of C with the current column of A times  b[p,1]
        //     c_0123_3 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_3);
    }

    // //store the updated results
    // _mm256_storeu_pd(&mut c[at(0, 0, ld_c)], c_0123_0);
    // _mm256_storeu_pd(&mut c[at(0, 1, ld_c)], c_0123_1);
    // _mm256_storeu_pd(&mut c[at(0, 2, ld_c)], c_0123_2);
    // _mm256_storeu_pd(&mut c[at(0, 3, ld_c)], c_0123_3);
}

#[cfg(any(
    all(target_arch = "x86", target_feature = "avx2"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]
#[allow(non_snake_case)]
/// # Safety
///
/// This is a kernel used for FMA (Fused multiply add) operaton to compute matrix product
pub unsafe fn gemm_4x4_kernel(
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &[f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    // Declare vector registers to hold 4x4 Columns and load them
    let mut c_0123_0: __m256d = _mm256_loadu_pd(&c[at(0, 0, ld_c)]);
    let mut c_0123_1: __m256d = _mm256_loadu_pd(&c[at(0, 1, ld_c)]);
    let mut c_0123_2: __m256d = _mm256_loadu_pd(&c[at(0, 2, ld_c)]);
    let mut c_0123_3: __m256d = _mm256_loadu_pd(&c[at(0, 3, ld_c)]);

    for p in 0..k {
        // declare vector register for loading/broadcasting b[p, j]
        let mut b_pj: __m256d;

        // declare a vector register to hold the current column of a and load it with four elements of that column
        let a_0123_p: __m256d = _mm256_loadu_pd(&a[at(0, p, ld_a)]);

        // Load/broadcast beta( p,0 )
        b_pj = _mm256_broadcast_sd(&b[at(p, 0, ld_b)]);

        // update the first column of c with the current element of a times b[p, 0]
        c_0123_0 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_0);

        // Load/broadcast b[p,1]
        b_pj = _mm256_broadcast_sd(&b[at(p, 1, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_1 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_1);

        // Load/broadcast b[p,2]
        b_pj = _mm256_broadcast_sd(&b[at(p, 2, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_2 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_2);

        // Load/broadcast b[p,3]
        b_pj = _mm256_broadcast_sd(&b[at(p, 3, ld_b)]);

        // update the second column of C with the current column of A times  b[p,1]
        c_0123_3 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_3);
    }

    //store the updated results
    _mm256_storeu_pd(&mut c[at(0, 0, ld_c)], c_0123_0);
    _mm256_storeu_pd(&mut c[at(0, 1, ld_c)], c_0123_1);
    _mm256_storeu_pd(&mut c[at(0, 2, ld_c)], c_0123_2);
    _mm256_storeu_pd(&mut c[at(0, 3, ld_c)], c_0123_3);
}
