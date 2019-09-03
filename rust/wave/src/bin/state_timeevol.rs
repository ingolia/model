use std::io::Write;

use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::{convert,convert_ref};
use num_complex::*;

use wave::sch;

pub const HSIZE: usize = 10;
pub const NSTATES: usize = 5;
pub const TSTEP: f64 = 0.01;
pub const TFINAL: f64 = 100.0;

fn main() {
    let model = sch::ModelS1::new(sch::PLANCK_DEFAULT, sch::MASS_DEFAULT, sch::LENGTH_DEFAULT, HSIZE);

    // a is the state with zero potential
    let ma = model.hamiltonian_v0_real();
    let ssa = sch::stationary_states(&ma);
    sch::write_stationary("psi-a.csv", "e-a.csv", &ma).unwrap();

    // b is the state with sinusoidal potential
    let xsin: DVector<f64> = DVector::from_fn(HSIZE, |k, _| 2.0 * std::f64::consts::PI * (k as f64) / (HSIZE as f64));
    let vsin = xsin * 2.0;
    let mb = model.hamiltonian_real(&vsin);
    let ssb = sch::stationary_states(&mb);
    sch::write_stationary("psi-b.csv", "e-b.csv", &mb).unwrap();

    // Observables
    let xhat = model.position();
    let qhat = model.momentum();
    let hahat: DMatrix<Complex<f64>> = convert_ref(&ma);
    let hbhat: DMatrix<Complex<f64>> = convert_ref(&mb);

    let state_obs = |psi: &DVector<Complex<f64>>| {
        let x = sch::herm_exp(psi, &xhat);
        let q = sch::herm_exp(psi, &qhat);
        let ea = sch::herm_exp(psi, &hahat);
        let eb = sch::herm_exp(psi, &hbhat);
        format!("{:0.3}\t{:0.3}\t{:0.3}\t{:0.3}", x, q, ea, eb)
    };

    let timeevol_a = sch::make_time_evol(&ma, TSTEP / sch::PLANCK_DEFAULT);
    let timeevol_b = sch::make_time_evol(&mb, TSTEP / sch::PLANCK_DEFAULT);
    
    let psi_a0 = ssa[0].1.clone();
    let psi_a1 = ssa[1].1.clone();
    let psi_a2 = ssa[2].1.clone();
    
    let mut times = Vec::new();
    let mut psis = Vec::new();
    let mut xs = Vec::new();
    let mut qs = Vec::new();

    let mut psi = psi_a0.clone();
    psi = (psi_a1 + psi_a2 * Complex::i()).scale(std::f64::consts::FRAC_1_SQRT_2);
    
    let mut psi_new = psi.clone();

    let mut t = 0.0;
    while t < TFINAL {
// //        let psi_new = (if t < 16.88 { &timeevol_b } else { &timeevol_a }) * &psi;
        
        times.push(t);
        psis.push(psi.clone());
        xs.push(sch::herm_exp(&psi, &xhat));
        qs.push(sch::herm_exp(&psi, &qhat));

        psi_new.gemv(convert(1.0), &timeevol_a, &psi, convert(0.0));
        std::mem::swap(&mut psi, &mut psi_new);
        t += TSTEP;
    }

    let mut fout = std::fs::File::create("te.csv").unwrap();
    write!(fout, "x\tt\txhat\tqhat").unwrap();
    for (i, (e_a, _psi_a)) in ssa.iter().enumerate() {
        write!(fout, "\ta{}_{:0.3}", i, e_a).unwrap();
    }
    write!(fout, "\n").unwrap();
    
    for i in 0..times.len() {
        write!(fout, "{}\t{:0.4}\t{:0.4}\t{:0.6}",
               i, times[i], xs[i], qs[i]).unwrap();

        for (_e_a, psi_a) in ssa.iter() {
            write!(fout, "\t{:0.4}", psi_a.dotc(&psis[i])).unwrap();
        }

        write!(fout, "\n").unwrap();
    }
    
    let mut fout = std::fs::File::create("res.csv").unwrap();
    write!(fout, "x").unwrap();
    for i in 0..psis.len() {
        write!(fout, ",a{:05},ph{:05}", i, i).unwrap();
    }
    write!(fout, "\n").unwrap();
    for j in 0..HSIZE {
        write!(fout, "{}", j).unwrap();
        for i in 0..psis.len() {
            let x = psis[i][j];
            write!(fout, ",{:6.3},{:6.3}", x.norm_sqr(), x.arg()).unwrap();
        }
        write!(fout, "\n").unwrap();
    }
}
