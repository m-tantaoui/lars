extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-lcublas")
        .files(&["src/matmul.cpp", "src/kernels.cu"])
        .compile("dot.a");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-link-search=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
}
