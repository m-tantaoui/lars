pub fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

pub fn ele_i(i: usize, inc: usize) -> usize {
    inc * i
}

pub fn display(m: usize, n: usize, ld: usize, a: &[f64]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[at(i, j, ld)]);
        }
        println!()
    }
}

pub fn display_f32(m: i32, n: i32, ld: i32, a: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            print!("{} \t", a[((j * ld) + i) as usize]);
        }
        println!()
    }
}
