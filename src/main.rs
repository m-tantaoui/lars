use std::ffi::{c_float, c_int};

use lars::{
    matmul::{gemm, naive_gemm},
    utils::display,
};

// use libc::{c_float, c_int};

const VEC_SIZE: c_int = 10;

extern "C" {
    fn dot(x: *mut c_float, y: *mut c_float, N: c_int) -> c_float;
}

fn cuda() {
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();

    for _ in 0..VEC_SIZE {
        x.push(5.0);
        y.push(3.0);
    }

    unsafe {
        let gpu_res = dot(x.as_mut_ptr(), y.as_mut_ptr(), VEC_SIZE);
    }
}

fn main() {
    cuda();
    // // defining matrices as vectors (column major matrices)
    // let m = 4; //number of rows of A
    // let n = 4; // number of columns of B
    // let k = 4; // number of columns of A and number of rows of B , they must be equal!!!!!

    // let a: Vec<f64> = (1..=(m * k)).map(|a| a as f64).collect();
    // let ld_a = m; // leading dimension of A (number of rows)

    // let b: Vec<f64> = (1..=(k * n)).map(|a| a as f64).collect();
    // let ld_b = k; // leading dimension of B (number of rows)

    // let mut c: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    // let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    // let ld_c = m; // leading dimension of C (number of rows)

    // // parallelized_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    // gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    // naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

    // display(m, n, ld_c, &c);
    // println!();
    // display(m, n, ld_c, &c_ref);
}
