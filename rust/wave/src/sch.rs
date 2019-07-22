use crate::linalg::*;

pub const H_DEFAULT: f64 = 4.0;
pub const MASS_DEFAULT: f64 = 1.0;
pub const LENGTH_DEFAULT: f64 = (2.0 * std::f64::consts::PI);

pub struct ModelS1 {
    h: f64,
    mass: f64,
    length: f64,
}

impl ModelS1 {
    pub fn new(h: f64, mass: f64, length: f64) -> Self {
        ModelS1 { h: h, mass: mass, length: length }
    }

    pub fn hamiltonian(&self, n: usize) -> MatrixSquare<f64> {
        let hstep = self.length / (n as f64);

        let mut hamil = MatrixSquare::zeros(n);

        let pfact = -0.5 * self.h * self.h / self.mass;
        let hstep2 = 1.0 / (hstep * hstep);

        for j in 0..n {
            hamil[(j, (j+n-1)%n)] = pfact * hstep2;
            hamil[(j, (j+n+1)%n)] = pfact * hstep2;
            hamil[(j, j)] = -2.0 * pfact * hstep2;
        }

        hamil
    }

    pub fn hamiltonian_spinor(&self, n: usize) -> MatrixSquare<f64> {
        let mut hamil = self.hamiltonian(n);

        hamil[(0,n-1)] = -hamil[(0,n-1)];
        hamil[(n-1,0)] = -hamil[(n-1,0)];

        hamil
    }    
}
