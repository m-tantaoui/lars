extern crate test;

use crate::utils::ele_ij;

use self::GemmRoutines::{Loop5Gemm, NaiveGemm};
use std::cmp::min;
use std::fmt;
use std::slice::Iter;

use core::arch::x86_64::*;

const NR: usize = 4;
const MR: usize = 4;

const NC: usize = 96;
const MC: usize = 96;
const KC: usize = 96;

#[allow(clippy::too_many_arguments)]
pub fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &[f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                c[ele_ij(i, j, ld_c)] += a[ele_ij(i, p, ld_a)] * b[ele_ij(p, j, ld_b)];
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(non_snake_case)]
unsafe fn gemm_MRxNR_kernel(
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    // Declare vector registers to hold 4x4 Columns and load them
    let mut c_0123_0: __m256d;
    let mut c_0123_1: __m256d;
    let mut c_0123_2: __m256d;
    let mut c_0123_3: __m256d;
    c_0123_0 = _mm256_loadu_pd(&c[ele_ij(0, 0, ld_c)]);
    c_0123_1 = _mm256_loadu_pd(&c[ele_ij(0, 1, ld_c)]);
    c_0123_2 = _mm256_loadu_pd(&c[ele_ij(0, 2, ld_c)]);
    c_0123_3 = _mm256_loadu_pd(&c[ele_ij(0, 3, ld_c)]);

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

#[allow(clippy::too_many_arguments)]
pub fn gemm_5_loops(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    if m % MR != 0 || MC % MR != 0 {
        panic!("m and MC must be multiples of MR\n")
    }

    if n % NR != 0 || NC % NR != 0 {
        panic!("n and NC must be multiples of NR\n")
    }

    loop5(m, n, k, a, ld_a, b, ld_b, c, ld_c);
}

#[allow(clippy::too_many_arguments)]
fn loop5(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for j in (0..n).step_by(NC) {
        let jb = min(NC, n - j);
        loop4(
            m,
            jb,
            k,
            a,
            ld_a,
            &mut b[ele_ij(0, j, ld_b)..],
            ld_b,
            &mut c[ele_ij(0, j, ld_c)..],
            ld_c,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn loop4(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for p in (0..k).step_by(KC) {
        let pb = min(KC, k - p);
        loop3(
            m,
            n,
            pb,
            &a[ele_ij(0, p, ld_a)..],
            ld_a,
            &mut b[ele_ij(p, 0, ld_b)..],
            ld_b,
            c,
            ld_c,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn loop3(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for i in (0..m).step_by(MC) {
        let ib = min(MC, m - i);
        loop2(
            ib,
            n,
            k,
            &a[ele_ij(i, 0, ld_a)..],
            ld_a,
            b,
            ld_b,
            &mut c[ele_ij(i, 0, ld_c)..],
            ld_c,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn loop2(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for j in (0..n).step_by(NR) {
        let jb = min(NR, n - j);
        loop1(
            m,
            jb,
            k,
            a,
            ld_a,
            &mut b[ele_ij(0, j, ld_b)..],
            ld_b,
            &mut c[ele_ij(0, j, ld_c)..],
            ld_c,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn loop1(
    m: usize,
    _: usize,
    k: usize,
    a: &[f64],
    ld_a: usize,
    b: &mut [f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for i in (0..m).step_by(MR) {
        unsafe {
            gemm_MRxNR_kernel(
                k,
                &a[ele_ij(i, 0, ld_a)..],
                ld_a,
                b,
                ld_b,
                &mut c[ele_ij(i, 0, ld_c)..],
                ld_c,
            )
        }
    }
}

#[derive(Debug)]
pub enum GemmRoutines {
    NaiveGemm,
    Loop5Gemm,
}

impl GemmRoutines {
    pub fn iterator() -> Iter<'static, GemmRoutines> {
        static ROUTINES: [GemmRoutines; 2] = [NaiveGemm, Loop5Gemm];
        ROUTINES.iter()
    }
}

impl fmt::Display for GemmRoutines {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::display;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use test::Bencher;

    #[bench]
    fn benchmark_naive_gemm(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 3; //number of rows of A
        let n = 2; // number of columns of B
        let k = 3; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a = vec![1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 3.0, 1.0];
        let ld_a = 3; // leading dimension of A

        let b = vec![-2.0, 1.0, -1.0, 1.0, 3.0, 2.0];
        let ld_b = 3; // leading dimension of B

        let mut c = vec![1.0, -1.0, -2.0, 0.0, 2.0, 1.0];
        let ld_c = 3; // leading dimension of C

        bencher.iter(|| {
            naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_naive_gemm() {
        // defining matrices as vectors (column major matrices)
        let m = 3; //number of rows of A
        let n = 2; // number of columns of B
        let k = 3; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a = vec![1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 3.0, 1.0];
        let ld_a = 3; // leading dimension of A

        let b = vec![-2.0, 1.0, -1.0, 1.0, 3.0, 2.0];
        let ld_b = 3; // leading dimension of B

        let mut c = vec![1.0, -1.0, -2.0, 0.0, 2.0, 1.0];
        let ld_c = 3; // leading dimension of C

        // computing C:=AB + C
        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

        display(m, n, ld_c, &c);

        assert_eq!(c, vec![-5.0, -5.0, 3.0, -1.0, 12.0, 7.0]);
    }

    #[bench]
    fn benchmark_gemm_5_loops(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 48; //number of rows of A
        let n = 4 * 48; // number of columns of B
        let k = 4 * 48; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 48; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 48; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 48; // leading dimension of C (number of rows)

        bencher.iter(|| {
            gemm_5_loops(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_gemm_5_loops() {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 2; //number of rows of A
        let n = 4 * 2; // number of columns of B
        let k = 4 * 2; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 2; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 2; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 2; // leading dimension of C (number of rows)

        // computing C:=AB + C
        gemm_5_loops(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }
}
