extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .files(&["src/dot.cpp", "src/dot_gpu.cu"])
        .compile("dot.a");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-link-search=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

