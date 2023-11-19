pub fn element(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}
pub fn display(m: usize, n: usize, ld: usize, a: &Vec<f64>) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[element(i, j, ld)]);
        }
        print!("\n")
    }
}
