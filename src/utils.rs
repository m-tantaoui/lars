pub fn ele_ij(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

pub fn ele_i(i: usize, inc: usize) -> usize {
    inc * i
}

pub fn display(m: usize, n: usize, ld: usize, a: &[f64]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[ele_ij(i, j, ld)]);
        }
        println!()
    }
}
