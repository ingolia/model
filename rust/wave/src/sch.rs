use std::io::Write;
use std::path::Path;

use num_complex::*;

use crate::linalg::*;

pub const PLANCK_DEFAULT: f64 = 4.0;
pub const MASS_DEFAULT: f64 = 1.0;
pub const LENGTH_DEFAULT: f64 = (2.0 * std::f64::consts::PI);

pub struct ModelS1 {
    planck: f64,
    mass: f64,
    length: f64,
}

impl ModelS1 {
    pub fn new(planck: f64, mass: f64, length: f64) -> Self {
        ModelS1 { planck: planck, mass: mass, length: length }
    }

    pub fn position(&self, n: usize) -> MatrixSquare<Complex64> {
        let mut pos = MatrixSquare::zeros(n);

        for j in 0..n {
            pos[(j,j)] = Complex64::new(self.length * ((j as f64 + 0.5) / (n as f64)), 0.0);
        }

        pos
    }

    pub fn momentum(&self, n: usize) -> MatrixSquare<Complex64> {
        let mut q = MatrixSquare::zeros(n);
        
        for j in 0..n {
            // - i h d/dx = - i h (psi_{j+1} - psi_{j-1}) = i h psi_{j-1} - i h psi_{j+1}
            q[(j, (j+n-1)%n)] = Complex64::i() * self.planck;
            q[(j, (j+n+1)%n)] = -Complex64::i() * self.planck;
        }

        q
    }
    
    pub fn hamiltonian<T>(&self, v: &NVector<f64, T>) -> MatrixSquare<f64> {
        let n = v.len();
        let hstep = self.length / (n as f64);

        let mut hamil = MatrixSquare::zeros(n);

        let pfact = -0.5 * self.planck * self.planck / self.mass;
        let hstep2 = 1.0 / (hstep * hstep);

        for j in 0..n {
            hamil[(j, (j+n-1)%n)] = pfact * hstep2;
            hamil[(j, (j+n+1)%n)] = pfact * hstep2;
            hamil[(j, j)] = -2.0 * pfact * hstep2 + v[j];
        }

        hamil
    }

    pub fn hamiltonian_spinor<T>(&self, v: &NVector<f64, T>) -> MatrixSquare<f64> {
        let mut hamil = self.hamiltonian(v);
        let n = v.len();
        
        hamil[(0,n-1)] = -hamil[(0,n-1)];
        hamil[(n-1,0)] = -hamil[(n-1,0)];

        hamil
    }    
}

pub fn stationary_states(hamil: &MatrixSquare<f64>) -> Vec<(f64,NVector<f64,Col>)> {
    let (eigvals, eigvecs) = hamil.dsyev();
    eigvals.into_iter().zip(eigvecs.cols().into_iter()).collect()
}

pub fn write_stationary<P: AsRef<Path>, Q: AsRef<Path>>(vec_file: P, e_file: Q, hamil: &MatrixSquare<f64>) -> std::io::Result<()> {
    let mut vec_out = std::fs::File::create(vec_file)?;
    let mut e_out = std::fs::File::create(e_file)?;

    let ss = stationary_states(hamil);

    write!(vec_out, "n")?;
    write!(e_out, "n")?;
    for j in 0..ss.len() {
        write!(vec_out, ",n{:03}", j)?;
        write!(e_out, ",n{:03}", j)?;
    }
    write!(vec_out, "\n")?;
    write!(e_out, "\n")?;

    write!(e_out, "E")?;
    for j in 0..ss.len() {
        write!(e_out, ",{:0.4}", ss[j].0)?;
    }
    write!(e_out, "\n")?;

    for i in 0..ss.len() {
        write!(vec_out, "x{:03}", i)?;
        for j in 0..ss.len() {
            write!(vec_out, ",{:0.4}", ss[j].1[i])?;
        }
        write!(vec_out, "\n")?;
    }
    Ok(())
}
    
pub fn make_time_evol(hamil: &MatrixSquare<f64>, tstep_over_planck: f64) -> MatrixSquare<Complex64> {
    // println!("H =\n{:6.3}", hamil);

    let (eigvals, eigvecs) = hamil.dsyev();

    let eigvecinv: MatrixSquare<Complex64> = eigvecs.into();
    
    let eigvecfwd = eigvecinv.dagger();

    // println!("eigvecfwd =\n{:13.3}", eigvecfwd);
    // println!("eigvecinv =\n{:13.3}", eigvecinv);

    let mut eigvalmat = MatrixSquare::zeros(eigvals.len());
    for (i, eigval) in eigvals.iter().enumerate() {
        eigvalmat[(i, i)] = (-Complex64::i() * f64::from(eigval * tstep_over_planck)).exp();
    }
    // println!("eigvalmat =\n{:13.3}", eigvalmat);

    let timeevol = eigvecinv.mmulm(&eigvalmat).mmulm(&eigvecfwd);

    // println!("timeevol =\n{:13.3}", timeevol);

    // println!("timeevol^H timeevol =\n{:13.3}", timeevol.dagger().mmulm(&timeevol));

    timeevol
}
