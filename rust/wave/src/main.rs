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

    let ssa = sch::stationary_states(&m);
    sch::write_stationary("psi-a.csv", "e-a.csv", &m);

    let xsin: Vec<f64> = (0..n).map(|k| 2.0 * std::f64::consts::PI * (k as f64) / (n as f64)).collect();
    let vsin: Vec<f64> = xsin.iter().map(|x| 2.0 * x.sin()).collect();
    let mb = model.hamiltonian(&NVector::row_from_vec(vsin));
    let ssb = sch::stationary_states(&mb);
    sch::write_stationary("psi-b.csv", "e-b.csv", &mb);

    for (i, (ebi, vbi)) in ssb.iter().enumerate() {
        println!("{:02}\t{:0.4}\t{:0.4}", i, ebi, ssa[0].1.dagger().dot(vbi));
    }

    let a0b = ssa[0].1.dagger().dot(&ssb[0].1);
    let a2b = ssa[0].1.dagger().dot(&ssb[2].1);
    // let sum = 

    let egap = ssb[2].0 - ssb[0].0;
    let time = 0.5 * std::f64::consts::PI / egap;
    println!("egap = {:0.3}, time = {:0.3}", egap, time);
    let tegap = sch::make_time_evol(&mb, time);
    let mut psi: NVector<Complex64,Col> = ssa[0].1.clone().into();
    println!("{:0.4}", psi.dagger());
    psi = tegap.mmulv(&psi);
    println!("{:0.4}", psi.dagger());

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
