use std::io::Write;
use std::path::Path;

use nalgebra::Complex;
use nalgebra::ComplexField;
use nalgebra::allocator::Allocator;
use nalgebra::base::*;
use nalgebra::base::dimension::*;
use nalgebra::linalg::SymmetricEigen;
use num_traits::*;

pub const PLANCK_DEFAULT: f64 = 4.0;
pub const MASS_DEFAULT: f64 = 1.0;
pub const LENGTH_DEFAULT: f64 = (2.0 * std::f64::consts::PI);

pub struct ModelS1 {
    planck: f64,
    mass: f64,
    length: f64,
    hsize: usize
}

impl ModelS1
{
    fn hsize(&self) -> usize {
        self.hsize
    }

    fn hstep(&self) -> f64 {
        self.length / (self.hsize() as f64)
    }
    
    pub fn new(planck: f64, mass: f64, length: f64, hsize: usize) -> Self {
        ModelS1 { planck: planck, mass: mass, length: length, hsize: hsize }
    }
    
    pub fn position(&self) -> DMatrix<Complex<f64>> {
        let mut pos: DMatrix<Complex<f64>> = DMatrix::from_element(self.hsize, self.hsize, Zero::zero());
        for j in 0..pos.nrows() {
            pos[(j, j)] = Complex::new((j as f64 + 0.5) * self.hstep(), 0.0);
        }
        pos
    }

    pub fn momentum(&self) -> DMatrix<Complex<f64>> {
        let mut q: DMatrix<Complex<f64>> = DMatrix::from_element(self.hsize, self.hsize, Zero::zero());
        let nrow = q.nrows();
            
        for j in 0..nrow {
            // - i h d/dx = - i h (psi_{j+1} - psi_{j-1}) = i h psi_{j-1} - i h psi_{j+1}
            q[(j, (j+nrow-1)%nrow)] = Complex::i() * self.planck;
            q[(j, (j+nrow+1)%nrow)] = -Complex::i() * self.planck;
        }

        q
    }

    pub fn hamiltonian_v0_real(&self) -> DMatrix<f64> {
        let mut hamil: DMatrix<f64> = DMatrix::from_element(self.hsize, self.hsize, 0.0);

        let pfact = -0.5 * self.planck * self.planck / self.mass;
        let hstep2 = 1.0 / (self.hstep() * self.hstep());

        let nrow = hamil.nrows();
        
        for j in 0..nrow {
            hamil[(j, (j+nrow-1)%nrow)] = pfact * hstep2;
            hamil[(j, (j+nrow+1)%nrow)] = pfact * hstep2;
            hamil[(j, j)] = -2.0 * pfact * hstep2;
        }

        hamil
    }
    
    pub fn hamiltonian_v0(&self) -> DMatrix<Complex<f64>> {
        DMatrix::from_iterator(self.hsize, self.hsize, self.hamiltonian_v0_real().into_iter().map(|&x| Complex::from_real(x)))
    }
    
    pub fn hamiltonian_real(&self, v: &DVector<f64>) -> DMatrix<f64> {
        if v.len() != self.hsize {
            panic!("hamiltonian V.len() {} != model hsize {}", v.len(), self.hsize);
        }

        let mut hamil = self.hamiltonian_v0_real();
        
        for j in 0..self.hsize {
            hamil[(j, j)] += v[j];
        }

        hamil
    }

    pub fn hamiltonian(&self, v: &DVector<f64>) -> DMatrix<Complex<f64>> {
        DMatrix::from_iterator(self.hsize, self.hsize, self.hamiltonian_real(v).into_iter().map(|&x| Complex::new(x, 0.0)))
    }

    // pub fn hamiltonian_spinor<T>(&self, v: &NVector<f64, T>) -> MatrixSquare<f64> {
    //     let mut hamil = self.hamiltonian(v);
    //     let n = v.len();
        
    //     hamil[(0,n-1)] = -hamil[(0,n-1)];
    //     hamil[(n-1,0)] = -hamil[(n-1,0)];

    //     hamil
    // }    
}

pub fn stationary_states(hamil: &DMatrix<f64>) -> Vec<(f64,DVector<Complex<f64>>)> {
    let eig = SymmetricEigen::new(hamil.clone());

    let mut states: Vec<(f64,DVector<Complex<f64>>)> = eig.eigenvalues.iter()
        .zip(eig.eigenvectors.column_iter())
        .map(|(&val, vec)| (val, DVector::from_iterator(hamil.nrows(), vec.into_iter().map(|&x| Complex::from_real(x)))))
        .collect();
    states.sort_by(|(va, _), (vb, _)| va.partial_cmp(vb).unwrap());
    states
}

pub fn write_stationary<P: AsRef<Path>, Q: AsRef<Path>>(vec_file: P, e_file: Q, hamil: &DMatrix<f64>) -> std::io::Result<()> {
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
    
// pub fn make_time_evol(hamil: &MatrixSquare<f64>, tstep_over_planck: f64) -> MatrixSquare<Complex64> {
//     // println!("H =\n{:6.3}", hamil);

//     let (eigvals, eigvecs) = hamil.dsyev();

//     let eigvecinv: MatrixSquare<Complex64> = eigvecs.into();
    
//     let eigvecfwd = eigvecinv.dagger();

//     // println!("eigvecfwd =\n{:13.3}", eigvecfwd);
//     // println!("eigvecinv =\n{:13.3}", eigvecinv);

//     let mut eigvalmat = MatrixSquare::zeros(eigvals.len());
//     for (i, eigval) in eigvals.iter().enumerate() {
//         eigvalmat[(i, i)] = (-Complex64::i() * f64::from(eigval * tstep_over_planck)).exp();
//     }
//     // println!("eigvalmat =\n{:13.3}", eigvalmat);

//     let timeevol = &eigvecinv * &eigvalmat * &eigvecfwd;

//     // println!("timeevol =\n{:13.3}", timeevol);

//     // println!("timeevol^H timeevol =\n{:13.3}", timeevol.dagger().mmulm(&timeevol));

//     timeevol
// }
