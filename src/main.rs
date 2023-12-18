use std::ffi::{c_float, c_int};

use lars::{
    matmul::{gemm, naive_gemm},
    utils::{display, display_f32},
};

extern "C" {

    fn matmul(m: c_int, n: c_int, k: c_int, a: *mut c_float, b: *mut c_float, c: *mut c_float);

}

fn cuda() {
    let m = 4;
    let n = 4;
    let k = 4;

    let mut a: Vec<f32> = Vec::new();
    let mut b: Vec<f32> = Vec::new();
    let mut c: Vec<f32> = Vec::new();

    for i in 1..=(m * n) {
        a.push(i as f32);
        b.push(i as f32);
        c.push(i as f32);
    }

    unsafe { matmul(m, n, k, a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr()) }

    display_f32(m, n, m, a.as_slice());
    println!();
    display_f32(m, n, m, b.as_slice());
    println!();
    display_f32(m, n, m, c.as_slice());
}

fn main() {
    cuda();

    // defining matrices as vectors (column major matrices)
    let m = 4; //number of rows of A
    let n = 4; // number of columns of B
    let k = 4; // number of columns of A and number of rows of B , they must be equal!!!!!

    let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
    let ld_a = m; // leading dimension of A (number of rows)

    let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
    let ld_b = k; // leading dimension of B (number of rows)

    let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    // let mut c: Vec<f64> = vec![0.0; (m * n)];
    let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    let ld_c = m; // leading dimension of C (number of rows)

    // parallelized_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

    display(m, n, ld_c, &c);
    println!();
    display(m, n, ld_c, &c_ref);
}
