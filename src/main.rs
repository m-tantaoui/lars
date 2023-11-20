use std::time::{Duration, SystemTime};

use lars::matmul::naive_gemm;

fn timeit<F: FnMut() -> T, T>(mut f: F, repeat: usize) -> Vec<Duration> {
    let durations: Vec<Duration> = (0..repeat)
        .map(|_| {
            let start = SystemTime::now();
            f();
            let end = SystemTime::now();
            end.duration_since(start).unwrap()
        })
        .collect();
    durations
}

fn compute_gflops() {
    let dim: &usize = &500;
    // defining matrices as vectors (column major matrices)
    let m: usize = *dim; //number of rows of A
    let n: usize = *dim; // number of columns of B
    let k: usize = *dim; // number of columns of A and number of rows of B , they must be equal!!!!!

    let vec_a: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
    let a = vec_a.as_slice();
    let ld_a = *dim; // leading dimension of A

    let vec_b: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
    let b = vec_b.as_slice();
    let ld_b = *dim; // leading dimension of B

    let mut vec_c: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
    let c = vec_c.as_mut_slice();
    let ld_c = *dim; // leading dimension of C

    let flops: f64 = (2 * m * n * k) as f64 * 10_f64.powi(-9); // number of floating points operations (in giga)
    let durations = timeit(|| naive_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c), 3);

    let durations_in_seconds: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_secs_f64())
        .collect();

    let gflops: Vec<f64> = durations_in_seconds
        .iter()
        .map(|seconds| flops / seconds)
        .collect();

    let gflops_mean: f64 = gflops.iter().sum::<f64>() / gflops.len() as f64;
    let duration = durations_in_seconds.iter().sum::<f64>() / durations_in_seconds.len() as f64;

    println!("GigaFlops: \n {:#?} ", gflops_mean);
    println!("duration in seconds {}", duration);
}

fn main() {
    compute_gflops()
}
