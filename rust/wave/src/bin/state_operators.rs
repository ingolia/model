//use std::io::Write;

use nalgebra::DVector;
use nalgebra::DMatrix;
use nalgebra::{convert,convert_ref};
use num_complex::*;

use wave::sch;

pub const HSIZE: usize = 10;
pub const NSTATES: usize = 5;

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

    // Enumerate stationary states of a
    println!("Stationary states of a:");
    for i in 0..NSTATES {
        let psi_ai: DVector<Complex<f64>> = ssa[i].1.clone();
        println!("{:02}\ta\t{:0.3}\t{}", i, ssa[i].0, state_obs(&psi_ai));
    }

    // Construct a mixed state
    let psi_a1 = ssa[1].1.clone();
    let psi_a2 = ssa[2].1.clone();

    let psiq =  (psi_a1 + psi_a2 * Complex::i()).scale(std::f64::consts::FRAC_1_SQRT_2);
    println!("psi_1 = 1/sqrt(2) (a_1 + i a_2)");
    println!("{:13.3}\nNorm psi_q* psi_q = {:13.3}", psiq, psiq.dotc(&psiq));
    println!("psiq\t{}", state_obs(&psiq));

    // Enumerate stationary state of b
    println!("Stationary states of b:");
    for i in 0..NSTATES {
        let psi_bi = ssb[i].1.clone();
        println!("{:02}\tb\t{:0.3}\t{}", i, ssb[i].0, state_obs(&psi_bi));
    }

    // Components of a_0 in basis of b:
    println!("Components of a_0 in basis of b:");
    for (i, (ebi, vbi)) in ssb.iter().enumerate() {
        println!("{:02}\t{:0.3}\t{:0.3}", i, ebi, ssa[0].1.dotc(vbi));
    }


    // for (i, (eai, vai)) in ssa.iter().enumerate() {
    //     let psi: NVector<Complex64, Col> = NVector::from(vai.clone());
    //     println!("{:02}\t{:+15.4}", i, psi.dagger());
    //     println!("\t{:+15.4}", (&q * &psi).dagger());
    //     println!("\t{:7.4}", psi.dagger().dot(&(&q * &psi)));
    // }

    // let h = model.hamiltonian_V0();
    // println!("H =\n{:0.2}", h);
    // for (i, (eai, vai)) in ssa.iter().enumerate() {
    //     let psi: NVector<Complex64, Col> = NVector::from(vai.clone());
    //     println!("{:02}\t{:+15.4}", i, psi.dagger());
    //     println!("\t{:+15.4}", (&h * &psi).dagger());
    //     println!("\t{:7.4} vs {:7.4}", psi.dagger().dot(&(&h * &psi)), eai);
    // }

    // let egap = ssb[2].0 - ssb[0].0;
    // let time = 0.5 * std::f64::consts::PI / egap;
    // println!("egap = {:0.3}, time = {:0.3}", egap, time);
    // let tegap = sch::make_time_evol(&mb, time);
    // let mut psi: NVector<Complex64,Col> = ssa[0].1.clone().into();
    // println!("{:0.4}", psi.dagger());
    // psi = tegap.mmulv(&psi);
    // println!("{:0.4}", psi.dagger());

    // let timeevol = sch::make_time_evol(&m, tstep / sch::PLANCK_DEFAULT);

    // let mut times = Vec::new();
    // let mut psis = Vec::new();

    // let mut psi: NVector<Complex64, Col> = NVector::col_from_vec(vec![zero(); n]);

    // for i in 0..n {
    //     let x = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
    //     psi[i] = Complex64::from(x.sin());
    // }
    // psi.scale(Complex64::from(psi.norm2().recip()));
    
    // for i in 0..1024 {
    //     psi = timeevol.mmulv(&psi);
    //     psis.push(psi.clone());
    //     times.push(i as f64 * tstep);
    // }

    // let mut fout = std::fs::File::create("res.csv").unwrap();
    // write!(fout, "x").unwrap();
    // for i in 0..psis.len() {
    //     write!(fout, ",a{:05},ph{:05}", i, i).unwrap();
    // }
    // write!(fout, "\n").unwrap();
    // for j in 0..n {
    //     write!(fout, "{}", j).unwrap();
    //     for i in 0..psis.len() {
    //         let x = psis[i][j];
    //         write!(fout, ",{:6.3},{:6.3}", x.norm_sqr(), x.arg()).unwrap();
    //     }
    //     write!(fout, "\n").unwrap();
    // }
}
