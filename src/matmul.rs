extern crate test;

use crate::dot::{axpy, dots};
use crate::utils::{ele_i, ele_ij};

use self::GemmRoutines::{AxPyGemm, AxPyGerGemm, DotsGemm, NaiveGemm};
use std::cmp::min;
use std::fmt;
use std::slice::Iter;

use core::arch::x86_64::*;

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

#[allow(clippy::too_many_arguments)]
pub fn dots_gemv(
    m: usize,
    n: usize,
    a: &[f64],
    ld_a: usize,
    x: &[f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
) {
    for i in 0..m {
        dots(
            n,
            &a[ele_ij(i, 0, ld_a)..],
            ld_a,
            x,
            incx,
            &mut y[ele_i(i, incy)],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn axpy_gemv(
    m: usize,
    n: usize,
    a: &[f64],
    ld_a: usize,
    x: &[f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
) {
    for j in 0..n {
        axpy(
            m,
            x[ele_i(j, incx)],
            &a[ele_ij(0, j, ld_a)..],
            incx,
            y,
            incy,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dots_gemm(
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
        dots_gemv(
            m,
            k,
            a,
            ld_a,
            &b[ele_ij(0, j, ld_b)..],
            1,
            &mut c[ele_ij(0, j, ld_c)..],
            1,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn axpy_gemm(
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
        axpy_gemv(
            m,
            k,
            a,
            ld_a,
            &b[ele_ij(0, j, ld_b)..],
            1,
            &mut c[ele_ij(0, j, ld_c)..],
            1,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn axpy_ger(
    m: usize,
    n: usize,
    x: &[f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
    a: &mut [f64],
    ld_a: usize,
) {
    for i in 0..m {
        axpy(
            n,
            x[ele_i(i, incx)],
            y,
            incy,
            &mut a[ele_ij(i, 0, ld_a)..],
            ld_a,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn axpy_ger_gemm(
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
    for p in 0..k {
        axpy_ger(
            m,
            n,
            &a[ele_ij(0, p, ld_a)..],
            1,
            &mut b[ele_ij(p, 0, ld_b)..],
            ld_b,
            c,
            ld_c,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn block_naive_gemm(
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
    let nb = 4;
    let mb = 4;
    let kb = 4;
    for j in (0..n).step_by(nb) {
        let jb = min(n - j, nb);
        for i in (0..m).step_by(mb) {
            let ib = min(m - i, mb);
            for p in (0..k).step_by(kb) {
                let pb = min(k - p, kb);
                naive_gemm(
                    ib,
                    jb,
                    pb,
                    &a[ele_ij(i, p, ld_a)..],
                    ld_a,
                    &b[ele_ij(p, j, ld_b)..],
                    ld_b,
                    &mut c[ele_ij(i, j, ld_c)..],
                    ld_c,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]

pub fn block_axpy_ger_gemm(
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
    let nr = 4;
    let mr = 4;
    let kr = 4;
    for i in (0..m).step_by(mr) {
        for j in (0..n).step_by(nr) {
            for p in (0..k).step_by(kr) {
                axpy_ger_gemm(
                    mr,
                    nr,
                    kr,
                    &a[ele_ij(i, p, ld_a)..],
                    ld_a,
                    &mut b[ele_ij(p, j, ld_b)..],
                    ld_b,
                    &mut c[ele_ij(i, j, ld_c)..],
                    ld_c,
                )
            }
        }
    }
}

fn gemm_4x4_kernel(
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
    unsafe {
        c_0123_0 = _mm256_loadu_pd(&c[ele_ij(0, 0, ld_c)]);
        c_0123_1 = _mm256_loadu_pd(&c[ele_ij(0, 1, ld_c)]);
        c_0123_2 = _mm256_loadu_pd(&c[ele_ij(0, 2, ld_c)]);
        c_0123_3 = _mm256_loadu_pd(&c[ele_ij(0, 3, ld_c)]);
    }

    for p in 0..k {
        // declare vector register for loading/broadcasting b[p, j]
        let mut b_pj: __m256d;

        // declare a vector register to hold the current column of a and load it with four elements of that column
        let a_0123_p: __m256d;
        unsafe {
            a_0123_p = _mm256_loadu_pd(&a[ele_ij(0, p, ld_a)]);
        }

        // Load/broadcast beta( p,0 )
        unsafe {
            b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 0, ld_b)]);
        }

        // update the first column of c with the current element of a times b[p, 0]
        unsafe { c_0123_0 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_0) }

        // Load/broadcast b[p,1]
        unsafe {
            b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 1, ld_b)]);
        }

        // update the second column of C with the current column of A times  b[p,1]
        unsafe {
            c_0123_1 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_1);
        }

        // Load/broadcast b[p,2]
        unsafe {
            b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 2, ld_b)]);
        }

        // update the second column of C with the current column of A times  b[p,1]
        unsafe {
            c_0123_2 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_2);
        }

        // Load/broadcast b[p,3]
        unsafe {
            b_pj = _mm256_broadcast_sd(&b[ele_ij(p, 3, ld_b)]);
        }

        // update the second column of C with the current column of A times  b[p,1]
        unsafe {
            c_0123_3 = _mm256_fmadd_pd(a_0123_p, b_pj, c_0123_3);
        }
    }

    //store the updated results
    unsafe {
        _mm256_storeu_pd(&mut c[ele_ij(0, 0, ld_c)], c_0123_0);
        _mm256_storeu_pd(&mut c[ele_ij(0, 1, ld_c)], c_0123_1);
        _mm256_storeu_pd(&mut c[ele_ij(0, 2, ld_c)], c_0123_2);
        _mm256_storeu_pd(&mut c[ele_ij(0, 3, ld_c)], c_0123_3);
    }

    // println!("{:#?}", gamma_0123_0)
}

#[allow(clippy::too_many_arguments)]
pub fn gemm(
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
    let mr = 4;
    let nr = 4;

    if m % mr != 0 || n % nr != 0 {
        panic!("m and n must be multiples of `mr` and `nr`, respectively \n");
    }

    for j in (0..n).step_by(nr) {
        for i in (0..m).step_by(mr) {
            gemm_4x4_kernel(
                k,
                &a[ele_ij(i, 0, ld_a)..],
                ld_a,
                &mut b[ele_ij(0, j, ld_b)..],
                ld_b,
                &mut c[ele_ij(i, j, ld_c)..],
                ld_c,
            )
        }
    }
}

#[derive(Debug)]
pub enum GemmRoutines {
    NaiveGemm,
    DotsGemm,
    AxPyGemm,
    AxPyGerGemm,
}

impl GemmRoutines {
    pub fn iterator() -> Iter<'static, GemmRoutines> {
        static ROUTINES: [GemmRoutines; 4] = [NaiveGemm, DotsGemm, AxPyGemm, AxPyGerGemm];
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
    fn benchmark_block_naive_gemm(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k * m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let b: Vec<f64> = (1..=(k * n * k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(k * n * k * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        bencher.iter(|| {
            block_naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_block_naive_gemm() {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        // computing C:=AB + C
        block_naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }

    #[bench]
    fn benchmark_dots_gemm(bencher: &mut Bencher) {
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
            dots_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_dots_gemm() {
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
        dots_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

        assert_eq!(c, vec![-5.0, -5.0, 3.0, -1.0, 12.0, 7.0]);
    }

    #[bench]
    fn benchmark_axpy_gemm(bencher: &mut Bencher) {
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
            axpy_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_axpy_gemm() {
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
        axpy_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

        assert_eq!(c, vec![-5.0, -5.0, 3.0, -1.0, 12.0, 7.0]);
    }

    #[bench]
    fn benchmark_dots_gemv(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m: usize = 3; //number of rows of A
        let n: usize = 4; // number of columns of A

        let a = vec![
            2.0, 2.0, -2.0, 2.0, 1.0, -2.0, -1.0, 0.0, 2.0, 2.0, -2.0, 2.0,
        ];
        let ld_a: usize = 3; // leading dimension of A

        let x = vec![2.0, -1.0, 0.0, -1.0];
        let inc_x: usize = 1;

        let mut y = vec![0.0; 4];
        let inc_y: usize = 1;

        bencher.iter(|| dots_gemv(m, n, &a, ld_a, &x, inc_x, &mut y, inc_y));
    }

    #[test]
    fn test_dots_gemv() {
        // defining matrices as vectors (column major matrices)
        let m: usize = 3; //number of rows of A
        let n: usize = 4; // number of columns of A

        let a = vec![
            2.0, 2.0, -2.0, 2.0, 1.0, -2.0, -1.0, 0.0, 2.0, 2.0, -2.0, 2.0,
        ];
        let ld_a: usize = 3; // leading dimension of A

        let x = vec![2.0, -1.0, 0.0, -1.0];
        let inc_x: usize = 1;

        let mut y = vec![0.0; 3];
        let inc_y: usize = 1;

        dots_gemv(m, n, &a, ld_a, &x, inc_x, &mut y, inc_y);

        assert_eq!(y, vec![0.0, 5.0, -4.0]);
    }

    #[bench]
    fn benchmark_axpy_gemv(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m: usize = 3; //number of rows of A
        let n: usize = 4; // number of columns of A

        let a = vec![
            2.0, 2.0, -2.0, 2.0, 1.0, -2.0, -1.0, 0.0, 2.0, 2.0, -2.0, 2.0,
        ];
        let ld_a: usize = 3; // leading dimension of A

        let x = vec![2.0, -1.0, 0.0, -1.0];
        let inc_x: usize = 1;

        let mut y = vec![0.0; 4];
        let inc_y: usize = 1;

        bencher.iter(|| axpy_gemv(m, n, &a, ld_a, &x, inc_x, &mut y, inc_y));
    }

    #[test]
    fn test_axpy_gemv() {
        // defining matrices as vectors (column major matrices)
        let m: usize = 3; //number of rows of A
        let n: usize = 4; // number of columns of A

        let a = vec![
            2.0, 2.0, -2.0, 2.0, 1.0, -2.0, -1.0, 0.0, 2.0, 2.0, -2.0, 2.0,
        ]; // column-major indexing
           // let a = vec![
           //     2.0, 2.0, -1.0, 2.0, 2.0, 1.0, 0.0, -2.0, -2.0, -2.0, 2.0, 2.0,
           // ];// row major indexing
        let ld_a: usize = 3; // leading dimension of A

        let x = vec![2.0, -1.0, 0.0, -1.0];
        let inc_x: usize = 1;

        let mut y = vec![0.0; 3];
        let inc_y: usize = 1;

        axpy_gemv(m, n, &a, ld_a, &x, inc_x, &mut y, inc_y);

        assert_eq!(y, vec![0.0, 5.0, -4.0]);
    }

    #[bench]
    fn benchmark_axpy_ger_gemm(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 3; //number of rows of A
        let n = 2; // number of columns of B
        let k = 3; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a = vec![1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 3.0, 1.0];
        let ld_a = 3; // leading dimension of A

        let mut b = vec![-2.0, 1.0, -1.0, 1.0, 3.0, 2.0];
        let ld_b = 3; // leading dimension of B

        let mut c = vec![1.0, -1.0, -2.0, 0.0, 2.0, 1.0];
        let ld_c = 3; // leading dimension of C

        bencher.iter(|| {
            axpy_ger_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_axpy_ger_gemm() {
        // defining matrices as vectors (column major matrices)
        let m = 3; //number of rows of A
        let n = 2; // number of columns of B
        let k = 3; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a = vec![1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 3.0, 1.0];
        let ld_a = 3; // leading dimension of A

        let mut b = vec![-2.0, 1.0, -1.0, 1.0, 3.0, 2.0];
        let ld_b = 3; // leading dimension of B

        let mut c = vec![1.0, -1.0, -2.0, 0.0, 2.0, 1.0];
        let ld_c = 3; // leading dimension of C

        // computing C:=AB + C
        axpy_ger_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);

        assert_eq!(c, vec![-5.0, -5.0, 3.0, -1.0, 12.0, 7.0]);
    }

    #[bench]
    fn benchmark_block_axpy_ger_gemm(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        bencher.iter(|| {
            block_axpy_ger_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_block_axpy_ger_gemm() {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        // computing C:=AB + C
        block_axpy_ger_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);

        naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }

    #[bench]
    fn benchmark_block_kernel_gemm(bencher: &mut Bencher) {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        bencher.iter(|| {
            gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);
        });
    }

    #[test]
    fn test_block_kernel_gemm() {
        // defining matrices as vectors (column major matrices)
        let m = 4 * 4; //number of rows of A
        let n = 4 * 4; // number of columns of B
        let k = 4 * 4; // number of columns of A and number of rows of B , they must be equal!!!!!

        let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
        let ld_a = 4 * 4; // leading dimension of A (number of rows)

        let mut b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
        let ld_b = 4 * 4; // leading dimension of B (number of rows)

        let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
        let ld_c = 4 * 4; // leading dimension of C (number of rows)

        // computing C:=AB + C
        naive_gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c, ld_c);

        gemm(m, n, k, &a, ld_a, &mut b, ld_b, &mut c_ref, ld_c);

        assert_eq!(c, c_ref);
    }
}
