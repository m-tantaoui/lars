use std::{
    error::Error,
    io::{self, Stdout},
    process,
};

use csv::Writer;
use serde::Serialize;
use std::time::{Duration, SystemTime};

use lars::matmul::{axpy_gemm, axpy_ger_gemm, dots_gemm, naive_gemm, GemmRoutines};

extern crate csv;

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

fn measure_performance(
    wtr: &mut Writer<Stdout>,
    dimensions: Vec<usize>,
) -> Result<(), Box<dyn Error>> {
    for dim in dimensions {
        for routine in GemmRoutines::iterator() {
            // let dim: &usize = &500;
            // defining matrices as vectors (column major matrices)
            let m: usize = dim; //number of rows of A
            let n: usize = dim; // number of columns of B
            let k: usize = dim; // number of columns of A and number of rows of B , they must be equal!!!!!

            let vec_a: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
            let a = vec_a.as_slice();
            let ld_a = dim; // leading dimension of A

            let mut vec_b: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
            let b = vec_b.as_mut_slice();
            let ld_b = dim; // leading dimension of B

            let mut vec_c: Vec<f64> = (0..dim.pow(2)).map(|i| i as f64).collect();
            let c = vec_c.as_mut_slice();
            let ld_c = dim; // leading dimension of C

            let flops: f64 = (2 * m * n * k) as f64 * 10_f64.powi(-9); // number of floating points operations (in giga)

            // let gemm = gemm_routine(*routine);
            let durations = timeit(
                || match routine {
                    GemmRoutines::NaiveGemm => naive_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c),
                    GemmRoutines::DotsGemm => dots_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c),
                    GemmRoutines::AxPyGemm => axpy_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c),
                    GemmRoutines::AxPyGerGemm => axpy_ger_gemm(m, n, k, a, ld_a, b, ld_b, c, ld_c),
                },
                1,
            );

            let durations_in_seconds: Vec<f64> = durations
                .iter()
                .map(|duration| duration.as_secs_f64())
                .collect();

            let gflops_vec: Vec<f64> = durations_in_seconds
                .iter()
                .map(|seconds| flops / seconds)
                .collect();

            let gflops_mean: f64 = gflops_vec.iter().sum::<f64>() / gflops_vec.len() as f64;
            let seconds_mean =
                durations_in_seconds.iter().sum::<f64>() / durations_in_seconds.len() as f64;

            wtr.serialize(Record {
                dimension: dim,
                gflops: gflops_mean,
                seconds: seconds_mean,
                routine: routine.to_string(),
            })
            .unwrap();
        }
    }

    Ok(())
}

// Note that structs can derive both Serialize and Deserialize!
#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Record {
    dimension: usize,
    gflops: f64,
    seconds: f64,
    routine: String,
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_writer(io::stdout());

    let dimensions = vec![50, 100, 200, 300, 500, 700, 1_000, 2_000, 3_000, 5000];

    measure_performance(&mut wtr, dimensions)?;

    wtr.flush()?;
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}
