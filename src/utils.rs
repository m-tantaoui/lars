pub fn element(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}
pub fn display(m: usize, n: usize, ld_c: usize, c: &Vec<f64>) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", c[element(i, j, ld_c)]);
        }
        print!("\n")
    }
}
