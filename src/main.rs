use lars::{
    matmul::{gemm, naive_gemm},
    utils::display,
};

use libc::{c_float, size_t};

const VEC_SIZE: usize = 10;

extern "C" {
    fn dot(v1: *mut c_float, v2: *mut c_float, N: size_t) -> c_float;
}

fn cpu_dot(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
    let mut res: f32 = 0.;
    for i in 0..v1.len() {
        res += v1[i] * v2[i];
    }
    return res;
}

fn cuda() {
    let mut v1: Vec<f32> = Vec::new();
    let mut v2: Vec<f32> = Vec::new();
    let mut gpu_res: c_float;
    let mut cpu_res: f32 = 0.;

    for _ in 0..VEC_SIZE {
        v1.push(1.0);
        v2.push(2.0);
    }

    println!("{:?}", v1);
    println!("{:?}", v2);

    println!("GPU computing started");
    unsafe {
        gpu_res = dot(v1.as_mut_ptr(), v2.as_mut_ptr(), VEC_SIZE);
    }
    println!("GPU computing finished");
    println!("GPU dot product result: {}", gpu_res);

    cpu_res = cpu_dot(v1, v2);
    println!("CPU dot product result: {}", cpu_res);
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
    let mut c_ref: Vec<f64> = (1..=(m * n)).map(|a| a as f64).collect();
    let ld_c = m; // leading dimension of C (number of rows)

    // parallelized_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);
    naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c_ref, ld_c);

    display(m, n, ld_c, &c);
    println!();
    display(m, n, ld_c, &c_ref);
}
