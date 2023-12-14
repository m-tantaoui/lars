use lars::matmul::{naive_gemm, parallelized_gemm};

fn main() {
    // defining matrices as vectors (column major matrices)
    let m = 120; //number of rows of A
    let n = 120; // number of columns of B
    let k = 4; // number of columns of A and number of rows of B , they must be equal!!!!!

    let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
    let ld_a = m; // leading dimension of A (number of rows)

    let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
    let ld_b = k; // leading dimension of B (number of rows)

    let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    let ld_c = m; // leading dimension of C (number of rows)

    parallelized_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

    // naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

    // assert_eq!(c, c_ref);
}
