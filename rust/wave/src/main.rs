extern crate accelerate_src;

use std::io::Write;

use num_complex::*;
use num_traits::zero;

mod linalg;
mod sch;

use linalg::*;

fn main() {
    let n = 40;
    let model = sch::ModelS1::new(sch::PLANCK_DEFAULT, sch::MASS_DEFAULT, sch::LENGTH_DEFAULT);
    let m = model.hamiltonian(&NVector::row_from_vec(vec![0.0; n]));
    let tstep = 1.0 / 128.0;
    
    let timeevol = sch::make_time_evol(&m, tstep / sch::PLANCK_DEFAULT);

    let mut times = Vec::new();
    let mut psis = Vec::new();

    let mut psi: NVector<Complex64, Col> = NVector::col_from_vec(vec![zero(); n]);

    for i in 0..n {
        let x = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        psi[i] = Complex64::from(x.sin());
    }
    psi.scale(Complex64::from(psi.norm2().recip()));
    
    for i in 0..1024 {
        psi = timeevol.mmulv(&psi);
        psis.push(psi.clone());
        times.push(i as f64 * tstep);
    }

    let mut fout = std::fs::File::create("res.csv").unwrap();
    write!(fout, "x").unwrap();
    for i in 0..psis.len() {
        write!(fout, ",a{:05},ph{:05}", i, i).unwrap();
    }
    write!(fout, "\n").unwrap();
    for j in 0..n {
        write!(fout, "{}", j).unwrap();
        for i in 0..psis.len() {
            let x = psis[i][j];
            write!(fout, ",{:6.3},{:6.3}", x.norm_sqr(), x.arg()).unwrap();
        }
        write!(fout, "\n").unwrap();
    }
}
