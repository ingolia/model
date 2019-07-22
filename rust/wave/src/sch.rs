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
