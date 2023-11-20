use std::time::{Duration, SystemTime};

use lars::gemm::naive_gemm;

fn timeit<F: FnMut() -> T, T>(mut f: F, repeat: usize) -> Vec<Duration> {
    let durations: Vec<Duration> = (0..repeat)
        .map(|i| {
            let start = SystemTime::now();
            f();
            let end = SystemTime::now();
            end.duration_since(start).unwrap()
        })
        .collect();
    durations
}

fn compute_gflops() {
    // defining matrices as vectors (column major matrices)
    let m: usize = 10; //number of rows of A
    let n = 10; // number of columns of B
    let k = 10; // number of columns of A and number of rows of B , they must be equal!!!!!

    let vec_a: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let a = vec_a.as_slice();
    let ld_a = 10; // leading dimension of A

    let vec_b: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let b = vec_b.as_slice();
    let ld_b = 10; // leading dimension of B

    let mut vec_c: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let c = vec_c.as_mut_slice();
    let ld_c = 10; // leading dimension of C

    // computing C:=AB + C
    // naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

    println!("{}", 10_f32.powi(-9));

    let flops: f64 = (2 * m * n * k) as f64 * 10_f64.powi(-9); // number of floating points operations
    let durations = timeit(|| naive_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c), 10);
    let durations_in_seconds: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_secs_f64())
        .collect();
    let gflops: Vec<f64> = durations_in_seconds
        .iter()
        .map(|seconds| flops / seconds)
        .collect();
    println!("GigaFlops {:?} ", gflops);
    // println!("duration in micro seconds {}", duration.as_micros())
}

fn main() {
    compute_gflops()
}
