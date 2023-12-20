extern crate test;

use crate::kernels::gemm_4x4_kernel;
use crate::utils::at;

use self::GemmRoutines::{Loop5Gemm, NaiveGemm};
use std::cmp::min;
use std::fmt;
use std::slice::Iter;

#[cfg(any(
    all(target_arch = "x86", target_feature = "avx2"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]
use crate::kernels::gemm_4x4_kernel;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::kernels::gemm_4x4_kernel_arm;

// TODO: define a function that can get maximum values for MR and NR
const NR: usize = 4;
const MR: usize = 4;

// Write some documentation that explains how we choose those parameters
// based on L1, L2 and L3 cache
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
                c[at(i, j, ld_c)] += a[at(i, p, ld_a)] * b[at(p, j, ld_b)];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn parallelized_gemm(
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
    if m % MR != 0 || MC % MR != 0 {
        panic!("m and MC must be multiples of MR\n")
    }

    if n % NR != 0 || NC % NR != 0 {
        panic!("n and NC must be multiples of NR\n")
    }

    // parallelize using rayon
    todo!();

    // // parallelizing the first loop
    // c.par_chunks_mut(m * NC)
    //     .enumerate()
    //     .for_each(|(j, c_chunk)| {
    //         let b_chunk_start_index = j * k * NC;
    //         let b_chunk_stop_index = min((j + 1) * (NC * k), b.len());
    //         let b_chunk = &b[b_chunk_start_index..b_chunk_stop_index];

    //         for p in (0..k).step_by(KC) {
    //             let a_p_start_index = p * m * KC;
    //             let a_p_stop_index = min((p + 1) * m * KC, a.len());
    //             let a_p = &a[a_p_start_index..a_p_stop_index];
    //         }
    //     });
}

#[allow(clippy::too_many_arguments)]
pub fn gemm(
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
    b: &[f64],
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
            &b[at(0, j, ld_b)..],
            ld_b,
            &mut c[at(0, j, ld_c)..],
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
    b: &[f64],
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
            &a[at(0, p, ld_a)..],
            ld_a,
            &b[at(p, 0, ld_b)..],
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
    b: &[f64],
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
            &a[at(i, 0, ld_a)..],
            ld_a,
            b,
            ld_b,
            &mut c[at(i, 0, ld_c)..],
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
    b: &[f64],
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
            &b[at(0, j, ld_b)..],
            ld_b,
            &mut c[at(0, j, ld_c)..],
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
    b: &[f64],
    ld_b: usize,
    c: &mut [f64],
    ld_c: usize,
) {
    for i in (0..m).step_by(MR) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                gemm_4x4_kernel(
                    k,
                    &a[at(i, 0, ld_a)..],
                    ld_a,
                    b,
                    ld_b,
                    &mut c[at(i, 0, ld_c)..],
                    ld_c,
                )
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                gemm_4x4_kernel_arm(
                    k,
                    &a[at(i, 0, ld_a)..],
                    ld_a,
                    b,
                    ld_b,
                    &mut c[at(i, 0, ld_c)..],
                    ld_c,
                )
            }
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
        let m = 800; //number of rows of A
        let n = 800; // number of columns of B
        let k = 16; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        bencher.iter(|| {
            naive_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
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
        let m = 800; //number of rows of A
        let n = 800; // number of columns of B
        let k = 16; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        bencher.iter(|| {
            gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_gemm_5_loops() {
        // defining matrices as vectors (column major matrices)
        let m = 4; //number of rows of A
        let n = 4; // number of columns of B
        let k = 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = m; // leading dimension of A (number of rows)

        let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = k; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = m; // leading dimension of C (number of rows)

        // // display(m, k, ld_a, &a);
        // // println!();
        // // display(k, n, ld_b, &b);
        // // println!();
        // // display(m, n, ld_c, &c);

        // c.par_chunks_mut(m * 4)
        //     .enumerate()
        //     .for_each(|(j, mut c_column)| {
        //         println!("inside closure calling gemm");
        //         gemm_5_loops(m, n, k, &a, ld_a, &b, ld_b, &mut c_column, ld_c);
        //     });

        // computing C:=AB + C
        gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }
}
