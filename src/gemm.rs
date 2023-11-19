use crate::utils::element;

pub fn naive_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &Vec<f64>,
    ld_a: usize,
    b: &Vec<f64>,
    ld_b: usize,
    c: &mut Vec<f64>,
    ld_c: usize,
) {
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[element(i, j, ld_c)] += a[element(i, p, ld_a)] * b[element(p, j, ld_b)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

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
}
