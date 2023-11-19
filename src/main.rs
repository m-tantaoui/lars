use std::time::{Duration, SystemTime};

use lars::gemm::naive_gemm;

fn timeit<F: FnMut() -> T, T>(mut f: F) -> Duration {
    let start = SystemTime::now();
    f();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    return duration;
}

fn compute_gflops() {
    // defining matrices as vectors (column major matrices)
    let m: usize = 10; //number of rows of A
    let n = 10; // number of columns of B
    let k = 10; // number of columns of A and number of rows of B , they must be equal!!!!!

    let a = (0..100).into_iter().map(|i| i as f64).collect();
    let ld_a = 10; // leading dimension of A

    let b = (100..200).into_iter().map(|i| i as f64).collect();
    let ld_b = 10; // leading dimension of B

    let mut c = (0..100).into_iter().map(|i| i as f64).collect();
    let ld_c = 10; // leading dimension of C

    // computing C:=AB + C
    // naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

    let flops: f32 = ((2 * m * n * k) as f32 * 1_f32.powi(-9)) as f32; // number of floating points operations
    let duration = timeit(|| naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c));
    let seconds = duration.as_secs() as f32;
    let gflops = flops / seconds;
    println!("GigaFlops {:.3} ", gflops);
    println!("duration in micro seconds {}", duration.as_micros())
}

fn main() {
    compute_gflops()
    // // defining matrices as vectors (column major matrices)
    // let m = 3; //number of rows of A
    // let n = 2; // number of columns of B
    // let k = 3; // number of columns of A and number of rows of B , they must be equal!!!!!

    // let a = vec![1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 3.0, 1.0];
    // let ld_a = 3; // leading dimension of A

    // let b = vec![-2.0, 1.0, -1.0, 1.0, 3.0, 2.0];
    // let ld_b = 3; // leading dimension of B

    // let mut c = vec![1.0, -1.0, -2.0, 0.0, 2.0, 1.0];
    // let ld_c = 3; // leading dimension of C

    // // computing C:=AB + C
    // naive_gemm(m, n, k, &a, ld_a, &b, ld_b, &mut c, ld_c);

    // display(m, n, ld_c, &c);
}
