extern crate test;
use crate::utils::ele_i;

pub fn dots(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize, gamma: &mut f64) {
    for i in 0..n {
        *gamma += x[ele_i(i, incx)] * y[ele_i(i, incy)];
    }
}

pub fn axpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    for i in 0..n {
        y[ele_i(i, incy)] += alpha * x[ele_i(i, incx)]; // Fused Multiply-Add
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use test::Bencher;

    #[test]
    fn test_dots() {
        //  defining a vector
        let a = [-1.0, 0.0, 2.0, 3.0, -1.0, 1.0, 4.0, -1.0, 3.0, 5.0];
        let mut gamma = 1.0;

        dots(3, &a[1..], 4, &a[3..], 1, &mut gamma);

        assert_eq!(gamma, 5.0)
    }

    #[test]
    fn test_axpy() {
        let n: usize = 3;
        let alpha: f64 = -2.0;
        let x: Vec<f64> = vec![2.0, -1.0, 3.0];
        let mut y: Vec<f64> = vec![2.0, 1.0, 0.0];

        axpy(n, alpha, &x, 1, &mut y, 1);

        assert_eq!(y, vec![-2.0, 3.0, -6.0])
    }

    #[bench]
    fn benchmark_dots(bencher: &mut Bencher) {
        //  defining a vector
        let a = [-1.0, 0.0, 2.0, 3.0, -1.0, 1.0, 4.0, -1.0, 3.0, 5.0];
        let mut gamma = 1.0;

        bencher.iter(|| dots(3, &a[1..], 4, &a[3..], 1, &mut gamma))
    }

    #[bench]
    fn benchmark_axpy(bencher: &mut Bencher) {
        let n: usize = 3;
        let alpha: f64 = -2.0;
        let x: Vec<f64> = vec![2.0, -1.0, 3.0];
        let mut y: Vec<f64> = vec![2.0, 1.0, 0.0];

        bencher.iter(|| axpy(n, alpha, &x, 1, &mut y, 1));
    }
}
