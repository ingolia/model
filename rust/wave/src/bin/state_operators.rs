//use std::io::Write;

use nalgebra::DVector;
use nalgebra::DMatrix;
use nalgebra::convert_ref;
use num_complex::*;

use wave::sch;

pub const HSIZE: usize = 10;
pub const NSTATES: usize = 5;

fn main() {
    let model = sch::ModelS1::new(sch::PLANCK_DEFAULT, sch::MASS_DEFAULT, sch::LENGTH_DEFAULT, HSIZE);
    let ma = model.hamiltonian_v0_real();
    let ssa = sch::stationary_states(&ma);
    sch::write_stationary("psi-a.csv", "e-a.csv", &ma).unwrap();

    let xsin: DVector<f64> = DVector::from_fn(HSIZE, |k, _| 2.0 * std::f64::consts::PI * (k as f64) / (HSIZE as f64));
    let vsin = xsin * 2.0;
    let mb = model.hamiltonian_real(&vsin);
    let ssb = sch::stationary_states(&mb);
    sch::write_stationary("psi-b.csv", "e-b.csv", &mb).unwrap();

    let xhat = model.position();
    let qhat = model.momentum();
    let hahat: DMatrix<Complex<f64>> = convert_ref(&ma);
    let hbhat: DMatrix<Complex<f64>> = convert_ref(&mb);

    let state_obs = |psi: &DVector<Complex<f64>>| {
        let x = psi.adjoint() * &xhat * psi;
        let q = psi.adjoint() * &qhat * psi;
        let ea = psi.adjoint() * &hahat * psi;
        let eb = psi.adjoint() * &hbhat * psi;
        format!("{:0.3}\t{:0.3}\t{:0.3}\t{:0.3}", x, q, ea, eb)
    };
    
    for i in 0..NSTATES {
        let psi_ai: DVector<Complex<f64>> = ssa[i].1.clone();
        println!("{:02}\ta\t{:0.3}\t{}", i, ssa[i].0, state_obs(&psi_ai));
    }
    
    let psi_a1 = ssa[1].1.clone();
    let psi_a2 = ssa[2].1.clone();

    let psiq =  (psi_a1 + psi_a2 * Complex::i()).scale(std::f64::consts::FRAC_1_SQRT_2);
    println!("psiq\t{:13.3}\t{:13.3}", psiq, psiq.adjoint() * &psiq);
    println!("psiq\t{}", state_obs(&psiq));
    
    for i in 0..NSTATES {
        let psi_bi = ssb[i].1.clone();
        println!("{:02}\tb\t{:0.3}\t{}", i, ssb[i].0, state_obs(&psi_bi));
    }

    for (i, (ebi, vbi)) in ssb.iter().enumerate() {
        println!("{:02}\t{:0.3}\t{:0.3}", i, ebi, ssa[0].1.adjoint() * vbi);
    }

    // let a0b = Complex64::from(ssa[0].1.dagger() * &ssb[0].1);
    // let a2b = Complex64::from(ssa[0].1.dagger() * &ssb[2].1);
    // let psi_b0 = CVector::from(&ssb[0].1);
    // let psi_b2 = CVector::from(&ssb[2].1);
    // let t0 = &psi_b0 * a0b + &psi_b2 * a2b;
    // let t1 = &psi_b0 * a0b + &psi_b2 * a2b * Complex::i();
    // let t2 = &psi_b0 * a0b - &psi_b2 * a2b;

    // println!("t0\t{}", state_obs(&t0));
    // println!("t1\t{}", state_obs(&t1));
    // println!("t2\t{}", state_obs(&t2));

    // println!("t0 =\t{:0.3}", t0);
    // println!("t1 =\t{:0.3}", t1);
    // println!("t2 =\t{:0.3}", t2);
    // println!("t1' t1=\t{:0.3}", t1.dagger() * &t1);

    // for (i, (eai, rvai)) in ssa.iter().enumerate() {
    //     let vai = CVector::from(rvai);
    //     println!("{:0.2}\t{:0.3}\t{:0.3}\t{:0.3}\t{:0.3}", i, eai, t0.dagger() * &vai, t1.dagger() * &vai, t2.dagger() * &vai);
    // }

    // let psi_a0: CVector<Complex64> = CVector::from(&ssa[0].1);
    // let psi_a1: CVector<Complex64> = CVector::from(&ssa[1].1);
    // let psi_a2: CVector<Complex64> = CVector::from(&ssa[2].1);

    // let t1a0 = t1.dagger() * &psi_a0;
    // let t1a1 = t1.dagger() * &psi_a1;
    // let t1a2 = t1.dagger() * &psi_a2;
    // let t1_0 = t1a0 * &psi_a0 + (t1a1 * &psi_a1 + t1a2 * &psi_a2);
    // let t1_1 = t1a0 * &psi_a0 + Complex::i() * (t1a1 * &psi_a1 + t1a2 * &psi_a2);
    // let t1_2 = t1a0 * &psi_a0 + -1.0 * (t1a1 * &psi_a1 + t1a2 * &psi_a2);

    // println!("t1_0\t{}", state_obs(&t1_0));
    // println!("t1_1\t{}", state_obs(&t1_1));
    // println!("t1_2\t{}", state_obs(&t1_2));

    // println!("psi_a0\t{:13.3}\t{:13.3}", psi_a0, t1a0);
    // println!("psi_a1\t{:13.3}\t{:13.3}", psi_a1, t1a1);
    // println!("psi_a2\t{:13.3}\t{:13.3}", psi_a2, t1a2);
    
    // println!("t1_0\t{:13.3}\t{:13.3}", t1_0, t1_0.dagger() * &t1_0);
    // println!("t1_1\t{:13.3}\t{:13.3}", t1_1, t1_1.dagger() * &t1_1);
    // println!("t1_2\t{:13.3}\t{:13.3}", t1_2, t1_2.dagger() * &t1_2);
    
    // println!("{:0.4}", sum.dagger());

    // let isum = &ssb[0].1 * a0b + &ssb[2].1 * -a2b;
    // println!("{:0.4}", isum.dagger());

    // println!("x =\n{:0.2}", x);

    // for (i, (eai, vai)) in ssa.iter().enumerate() {
    //     let psi: NVector<Complex64, Col> = NVector::from(vai.clone());
    //     println!("{:02}\t{:+15.4}", i, psi.dagger());
    //     println!("\t{:+15.4}", (&x * &psi).dagger());
    //     println!("\t{:.4}", psi.dagger().dot(&(&x * &psi)));
    // }
    
    // let q = model.momentum();

    // println!("q =\n{:0.2}", q);

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
