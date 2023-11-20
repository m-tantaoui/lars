extern crate test;

use crate::dot::{axpy, dots};
use crate::utils::{ele_i, ele_ij};

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

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use test::Bencher;

    #[bench]
    fn benchmark_gemm(bencher: &mut Bencher) {
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
    fn test_gemm() {
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

        assert_eq!(c, vec![-5.0, -5.0, 3.0, -1.0, 12.0, 7.0]);
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
}
