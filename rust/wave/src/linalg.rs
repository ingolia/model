use std::fmt::{Display,Formatter};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use blas::{ddot, dgemv, dnrm2, dznrm2, dscal, zdotu, zgemv, zscal};
use lapack::dsyev;
use num_complex::*;
use num_traits::identities::*;

pub trait Conjugate {
    type Output;
    fn dagger(&self) -> Self::Output;
}

impl Conjugate for f64 {
    type Output = f64;
    fn dagger(&self) -> Self::Output {
        *self
    }
}

impl Conjugate for Complex<f64> {
    type Output = Complex<f64>;
    fn dagger(&self) -> Self::Output {
        self.conj()
    }
}

pub trait Blastype
    where Self: std::marker::Sized {

    fn xnrm2(x: &[Self]) -> f64;
    fn xscal(x: &mut [Self], a: Self) -> ();
    fn xdot(x: &[Self], y: &[Self]) -> Self;

    fn xgemv_simple(a: &[Self], x: &[Self]) -> Vec<Self>;
}

impl Blastype for f64 {
    fn xnrm2(x: &[Self]) -> f64 {
        unsafe {
            dnrm2(x.len() as i32, x, 1)
        }
    }

    fn xscal(x: &mut [Self], a: Self) -> () {
        unsafe {
            dscal(x.len() as i32, a, x, 1);
        }
    }

    fn xdot(x: &[Self], y: &[Self]) -> Self {
        if x.len() != y.len() {
            panic!("f64 dot product length {} != {}", x.len(), y.len());
        }
        unsafe {
            ddot(x.len() as i32, x, 1, y, 1)
        }
    }

    fn xgemv_simple(a: &[Self], x: &[Self]) -> Vec<Self> {
        let n = x.len();
        if a.len() != n * n {
            panic!("f64 matrix-vector mismatch {} vs {}", x.len(), a.len());
        }
        let mut y = vec![0.0; n];
        unsafe {
            dgemv(b'N', n as i32, n as i32, 1.0, a, n as i32, x, 1, 0.0, &mut y, 1);
        }
        y
    }
}

impl Blastype for Complex64 {
    fn xnrm2(x: &[Self]) -> f64 {
        unsafe {
            dznrm2(x.len() as i32, x, 1)
        }
    }

    fn xscal(x: &mut [Self], a: Self) -> () {
        unsafe {
            zscal(x.len() as i32, a, x, 1);
        }
    }

    fn xdot(x: &[Self], y: &[Self]) -> Self {
        if x.len() != y.len() {
            panic!("Complex64 dot product length {} != {}", x.len(), y.len());
        }
        let mut pres: [Complex64; 1] = [Complex64::zero()];
        unsafe {
            zdotu(&mut pres, x.len() as i32, x, 1, y, 1);
        }
        pres[0]
    }

    fn xgemv_simple(a: &[Self], x: &[Self]) -> Vec<Self> {
        let n = x.len();
        if a.len() != n * n {
            panic!("c64 matrix-vector mismatch {} vs {}", x.len(), a.len());
        }
        let mut y = vec![zero(); n];
        unsafe {
            zgemv(b'N', n as i32, n as i32, one(), a, n as i32, x, 1, zero(), &mut y, 1);
        }
        y
    }
}

pub trait VType {}

pub struct Row {}
impl VType for Row {}
pub struct Col {}
impl VType for Col {}

pub struct NVector<E, T> {
    data: Vec<E>,
    phantom: PhantomData<T>,
}

impl<E, T: VType> NVector<E, T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<E> NVector<E, Row> {
    pub fn row_from_vec(data: Vec<E>) -> Self {
        NVector {
            data: data,
            phantom: PhantomData,
        }
    }
}

impl<E> NVector<E, Col> {
    pub fn col_from_vec(data: Vec<E>) -> Self {
        NVector {
            data: data,
            phantom: PhantomData,
        }
    }
}

impl <E: Blastype, T> NVector<E, T> {
    pub fn norm2(&self) -> f64 {
        Blastype::xnrm2(&self.data)
    }

    pub fn scale(&mut self, a: E) -> () {
        Blastype::xscal(&mut self.data, a);
    }

    pub fn dot(&self, y: &NVector<E, Col>) -> E {
        Blastype::xdot(&self.data, &y.data)
    }
}

impl <T> From<NVector<f64, T>> for NVector<Complex64, T> {
    fn from(v: NVector<f64, T>) -> Self {
        NVector { data: v.data.iter().map(Complex64::from).collect(), phantom: PhantomData }
    }
}

impl Display for NVector<f64, Row> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let mut iter = self.data.iter();

        write!(f, "[")?;
        if let Some(x) = iter.next() {
            x.fmt(f)?;
        }
        for x in iter {
            write!(f, ", ")?;
            x.fmt(f)?;
        }
        write!(f, "]")
    }
}

impl<E, T> Index<usize> for NVector<E, T> {
    type Output = E;

    fn index(&self, idx: usize) -> &E {
        &self.data[idx]
    }
}

impl<E, T> IndexMut<usize> for NVector<E, T> {
    fn index_mut(&mut self, idx: usize) -> &mut E {
        &mut self.data[idx]
    }
}

impl<E: Conjugate> Conjugate for NVector<E, Row> {
    type Output = NVector<E::Output, Col>;
    fn dagger(&self) -> Self::Output {
        NVector::col_from_vec(self.data.iter().map(Conjugate::dagger).collect())
    }
}

impl<E: Conjugate> Conjugate for NVector<E, Col> {
    type Output = NVector<E::Output, Row>;
    fn dagger(&self) -> Self::Output {
        NVector::row_from_vec(self.data.iter().map(Conjugate::dagger).collect())
    }
}

pub struct MatrixSquare<E> {
    n: usize,
    data: Vec<E>,
}

impl<E> MatrixSquare<E> {
    fn data_index(&self, row_col: (usize, usize)) -> usize {
        let (row, col) = row_col;
        if row >= self.n || col >= self.n {
            panic!(
                "Index ({}, {}) for {}x{} MatrixSquare",
                row, col, self.n, self.n
            );
        }
        row * self.n + col
    }
}

impl <E: Copy> MatrixSquare<E> {
    pub fn from_rows(rows: &[NVector<E, Row>]) -> Self {
        let n = rows.len();
        let mut data = Vec::with_capacity(n*n);
        for i in 0..n {
            if rows[i].len() != n {
                panic!("from_rows row {} length {} != {}", i, rows[i].len(), n);
            }
        }
        for j in 0..n {
            for i in 0..n {
                data.push(rows[i][j]);
            }
        }
        MatrixSquare { n: n, data: data }
    }
}

impl MatrixSquare<f64> {
    pub fn dsyev(&self) -> (Vec<f64>, Vec<NVector<f64, Row>>) {
        let mut m = self.data.clone();
        let mut eigvals = vec![0.0; self.n];
        let mut work_len = vec![0.0];
        let mut info = 0;
        let n = self.n as i32;

        unsafe {
            dsyev(
                b'V', b'U', n, &mut m, n, &mut eigvals, &mut work_len, -1, &mut info,
            );
        }

        if info != 0 {
            panic!("dsyev: Failed getting work length: {}", info);
        }

        let lwork: i32 = work_len[0] as i32;
        let mut work = vec![0.0; lwork as usize];

        unsafe {
            dsyev(
                b'V', b'U', n, &mut m, n, &mut eigvals, &mut work, lwork, &mut info,
            );
        }

        if info < 0 {
            panic!("dsyev: Illegal argument {}", -info);
        } else if info > 0 {
            panic!("dsyev: Convergence failure {}", info);
        }

        let mut eigvecs = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut eigvec = Vec::with_capacity(self.n);
            for j in 0..self.n {
                eigvec.push(m[self.data_index((i, j))]);
            }
            eigvecs.push(NVector::row_from_vec(eigvec));
        }

        (eigvals, eigvecs)
    }
}

impl<E: Zero + Clone> MatrixSquare<E> {
    pub fn zeros(n: usize) -> Self {
        MatrixSquare {
            n: n,
            data: vec![zero(); n * n],
        }
    }
}

impl<E: Zero + One + Clone> MatrixSquare<E> {
    pub fn one(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                data.push(if i == j { one() } else { zero() });
            }
        }
        MatrixSquare { n: n, data: data }
    }
}

impl From<MatrixSquare<f64>> for MatrixSquare<Complex64> {
    fn from(v: MatrixSquare<f64>) -> Self {
        MatrixSquare { n: v.n, data: v.data.iter().map(Complex64::from).collect() }
    }
    
}

impl<E> Index<(usize, usize)> for MatrixSquare<E> {
    type Output = E;

    fn index(&self, row_col: (usize, usize)) -> &E {
        &self.data[self.data_index(row_col)]
    }
}

impl<E> IndexMut<(usize, usize)> for MatrixSquare<E> {
    fn index_mut(&mut self, row_col: (usize, usize)) -> &mut E {
        let idx = self.data_index(row_col);
        &mut self.data[idx]
    }
}

impl<E: Conjugate> Conjugate for MatrixSquare<E> {
    type Output = MatrixSquare<E::Output>;

    fn dagger(&self) -> Self::Output {
        let mut data = Vec::with_capacity(self.n * self.n);
        for j in 0..self.n {
            for i in 0..self.n {
                data.push(self[(i, j)].dagger());
            }
        }
        MatrixSquare {
            n: self.n,
            data: data,
        }
    }
}

impl <E: Display> Display for MatrixSquare<E> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        for i in 0..self.n {
            if i == 0 { write!(f, "[")?; } else { write!(f, " ")?; }
            
            if self.n > 0 {
                self[(i, 0)].fmt(f)?;
                for j in 1..self.n {
                    write!(f, ", ")?;
                    self[(i, j)].fmt(f)?;
                }
            }
            
            if i+1 == self.n {
                write!(f, "]")?;
            } else {
                write!(f, "\n")?;
            }
        }

        Ok(())
    }
}

