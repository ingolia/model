#![recursion_limit="16"]
use std::borrow::Borrow;
use std::fmt::{Display,Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use blas::{ddot, dgemm, dgemv, dnrm2, dznrm2, dscal, zdotc, zgemm, zgemv, zscal};
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

    fn xgemv_simple(trans: bool, a: &[Self], x: &[Self]) -> Vec<Self>;
    fn xgemm_simple(n: usize, a: &[Self], b: &[Self]) -> Vec<Self>;
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

    fn xgemv_simple(trans: bool, a: &[Self], x: &[Self]) -> Vec<Self> {
        let n = x.len();
        if a.len() != n * n {
            panic!("f64 matrix-vector mismatch {} vs {}", x.len(), a.len());
        }
        let mut y = vec![0.0; n];
        unsafe {
            dgemv(if trans { b'T' } else { b'N' }, n as i32, n as i32, 1.0, a, n as i32, x, 1, 0.0, &mut y, 1);
        }
        y
    }

    fn xgemm_simple(n: usize, a: &[Self], b: &[Self]) -> Vec<Self> {
        if a.len() != n * n || b.len() != n * n {
            panic!("f64 matrix-vector mismatch {} vs {}", a.len(), b.len());
        }
        let mut c = vec![0.0; n * n];
        unsafe {
            dgemm(b'N', b'N', n as i32, n as i32, n as i32, 1.0, a, n as i32, b, n as i32, 0.0, &mut c, n as i32);
        }
        c
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
            zdotc(&mut pres, x.len() as i32, x, 1, y, 1);
        }
        pres[0]
    }

    fn xgemv_simple(trans: bool, a: &[Self], x: &[Self]) -> Vec<Self> {
        let n = x.len();
        if a.len() != n * n {
            panic!("c64 matrix-vector mismatch {} vs {}", x.len(), a.len());
        }
        let mut y = vec![zero(); n];
        unsafe {
            zgemv(if trans { b'T' } else { b'N' }, n as i32, n as i32, one(), a, n as i32, x, 1, zero(), &mut y, 1);
        }
        y
    }

    fn xgemm_simple(n: usize, a: &[Self], b: &[Self]) -> Vec<Self> {
        if a.len() != n * n || b.len() != n * n {
            panic!("c64 matrix-vector mismatch {} vs {}", a.len(), b.len());
        }
        let mut c = vec![zero(); n * n];
        unsafe {
            zgemm(b'N', b'N', n as i32, n as i32, n as i32, one(), a, n as i32, b, n as i32, zero(), &mut c, n as i32);
        }
        c
    }
}

pub trait VType {}

#[derive(Debug,Clone,Copy,Hash,PartialEq,Eq)]
pub struct Row {}
impl VType for Row {}
#[derive(Debug,Clone,Copy,Hash,PartialEq,Eq)]
pub struct Col {}
impl VType for Col {}

#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct NVector<E, T> {
    data: Vec<E>,
    phantom: PhantomData<T>,
}

impl<E, T> NVector<E, T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn elts(&self) -> &[E] {
        self.data.as_slice()
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
}

impl <E: Blastype> NVector<E, Row> {
    pub fn dot(&self, y: &NVector<E, Col>) -> E {
        Blastype::xdot(&self.data, &y.data)
    }
}

impl <T> From<NVector<f64, T>> for NVector<Complex64, T> {
    fn from(v: NVector<f64, T>) -> Self {
        NVector { data: v.data.iter().map(Complex64::from).collect(), phantom: PhantomData }
    }
}

impl <T: Display> Display for NVector<T, Row> {
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

impl <'a, 'b, E, F, G, T> Add<&'b NVector<F, T>> for &'a NVector<E, T>
    where &'a E: Add<&'b F, Output = G>
{
    type Output = NVector<G, T>;

    fn add(self, rhs: &'b NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::add lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.iter().zip(rhs.data.iter()).map(|(x, y)| x + y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'a, E, F, G, T> Add<NVector<F, T>> for &'a NVector<E, T>
    where &'a E: Add<F, Output = G>
{
    type Output = NVector<G, T>;

    fn add(self, rhs: NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::add lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.iter().zip(rhs.data.into_iter()).map(|(x, y)| x + y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'b, E, F, G, T> Add<&'b NVector<F, T>> for NVector<E, T>
    where E: Add<&'b F, Output = G>
{
    type Output = NVector<G, T>;

    fn add(self, rhs: &'b NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::add lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.into_iter().zip(rhs.data.iter()).map(|(x, y)| x + y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <E, T> Add<NVector<E, T>> for NVector<E, T>
    where E: Add<E, Output = E>
{
    type Output = NVector<E, T>;

    fn add(self, rhs: NVector<E, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::add lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data: Vec<E> = self.data.into_iter().zip(rhs.data.into_iter()).map(|(x,y)| x+y).collect();
        
        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'a, 'b, E, F, G, T> Sub<&'b NVector<F, T>> for &'a NVector<E, T>
    where &'a E: Sub<&'b F, Output = G>
{
    type Output = NVector<G, T>;

    fn sub(self, rhs: &'b NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::sub lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.iter().zip(rhs.data.iter()).map(|(x, y)| x - y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'a, E, F, G, T> Sub<NVector<F, T>> for &'a NVector<E, T>
    where &'a E: Sub<F, Output = G>
{
    type Output = NVector<G, T>;

    fn sub(self, rhs: NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::sub lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.iter().zip(rhs.data.into_iter()).map(|(x, y)| x - y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'b, E, F, G, T> Sub<&'b NVector<F, T>> for NVector<E, T>
    where E: Sub<&'b F, Output = G>
{
    type Output = NVector<G, T>;

    fn sub(self, rhs: &'b NVector<F, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::sub lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data = self.data.into_iter().zip(rhs.data.iter()).map(|(x, y)| x - y).collect();

        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <E, T> Sub<NVector<E, T>> for NVector<E, T>
    where E: Sub<E, Output = E>
{
    type Output = NVector<E, T>;

    fn sub(self, rhs: NVector<E, T>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("NVector::sub lenght mismatch {} vs {}", self.data.len(), rhs.data.len());
        }

        let sum_data: Vec<E> = self.data.into_iter().zip(rhs.data.into_iter()).map(|(x,y)| x - y).collect();
        
        NVector { data: sum_data, phantom: PhantomData }
    }
}

impl <'a, E, F: Copy, G, T> Mul<F> for &'a NVector<E, T>
where &'a E: Mul<F, Output = G>
{
    type Output = NVector<G, T>;

    fn mul(self, rhs: F) -> Self::Output {
        let mut pdt_data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            pdt_data.push(&self.data[i] * rhs);
        }

        NVector { data: pdt_data, phantom: PhantomData }
    }
}

impl <E, F: Copy, G, T> Mul<F> for NVector<E, T>
where E: Mul<F, Output = G>
{
    type Output = NVector<G, T>;

    fn mul(self, rhs: F) -> Self::Output {
        let pdt_data = self.data.into_iter().map(|x| x * rhs).collect();
        NVector { data: pdt_data, phantom: PhantomData }
    }
}

impl <'a, E, F, T> Mul<&'a NVector<E, T>> for f64
  where E: Mul<f64, Output = F> + Copy
{
    type Output = NVector<F, T>;

    fn mul(self, rhs: &'a NVector<E, T>) -> Self::Output {
        NVector { data: rhs.data.iter().map(|x| *x * self).collect(), phantom: PhantomData }
    }
}

impl <E, F, T> Mul<NVector<E, T>> for f64
  where E: Mul<f64, Output = F> + Copy
{
    type Output = NVector<F, T>;

    fn mul(self, rhs: NVector<E, T>) -> Self::Output {
        NVector { data: rhs.data.iter().map(|x| *x * self).collect(), phantom: PhantomData }
    }
}

impl <'a, E, F, T> Mul<&'a NVector<E, T>> for Complex64
where E: Mul<Complex64, Output = F> + Copy
{
    type Output = NVector<F, T>;

    fn mul(self, rhs: &'a NVector<E, T>) -> Self::Output {
        NVector { data: rhs.data.iter().map(|x| *x * self).collect(), phantom: PhantomData }
    }
}

impl <E, F, T> Mul<NVector<E, T>> for Complex64
where E: Mul<Complex64, Output = F> + Copy
{
    type Output = NVector<F, T>;

    fn mul(self, rhs: NVector<E, T>) -> Self::Output {
        NVector { data: rhs.data.iter().map(|x| *x * self).collect(), phantom: PhantomData }
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

#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct MatrixSquare<E> {
    n: usize,
    data: Vec<E>,
}

impl<E> MatrixSquare<E> {
    pub fn n(&self) -> usize { self.n }

    fn data_index(&self, row_col: (usize, usize)) -> usize {
        let (row, col) = row_col;
        if row >= self.n || col >= self.n {
            panic!(
                "Index ({}, {}) for {}x{} MatrixSquare",
                row, col, self.n, self.n
            );
        }
        row + col * self.n
    }
}

impl <E: Copy> MatrixSquare<E> {
    pub fn rows(&self) -> Vec<NVector<E, Row>> {
        let mut rows = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut row = Vec::with_capacity(self.n);
            for j in 0..self.n {
                row.push(self[(i, j)]);
            }
            rows.push(NVector::row_from_vec(row));
        }
        rows
    }

    pub fn cols(&self) -> Vec<NVector<E, Col>> {
        let mut cols = Vec::with_capacity(self.n);
        for j in 0..self.n {
            let mut col = Vec::with_capacity(self.n);
            for i in 0..self.n {
                col.push(self[(i, j)]);
            }
            cols.push(NVector::col_from_vec(col));
        }
        cols
    }

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

impl <E: Blastype> MatrixSquare<E> {
    pub fn mmulv(&self, v: &NVector<E, Col>) -> NVector<E, Col> {
        NVector::col_from_vec(Blastype::xgemv_simple(false, &self.data, &v.data))
    }

    pub fn mmulm(&self, b: &MatrixSquare<E>) -> MatrixSquare<E> {
        MatrixSquare { n: self.n, data: Blastype::xgemm_simple(self.n, &self.data, &b.data) }
    }
}

impl <'a, 'b> Mul<&'b NVector<f64, Col>> for &'a MatrixSquare<f64> {
    type Output = NVector<f64, Col>;
    
    fn mul(self, rhs: &'b NVector<f64, Col>) -> Self::Output {
        let n = rhs.data.len();
        if self.data.len() != n * n {
            panic!("f64 matrix-vector mismatch {} vs {}", self.data.len(), rhs.data.len());
        }
        let mut y = vec![0.0; n];
        unsafe {
            dgemv(b'N', n as i32, n as i32, 1.0, &self.data, n as i32, &rhs.data, 1, 0.0, &mut y, 1);
        }
        NVector { data: y, phantom: PhantomData }
    }
}

impl <'a, 'b> Mul<&'b MatrixSquare<f64>> for &'a NVector<f64, Row> {
    type Output = NVector<f64, Row>;
    
    fn mul(self, rhs: &'b MatrixSquare<f64>) -> Self::Output {
        let n = self.data.len();
        if rhs.data.len() != n * n {
            panic!("f64 matrix-vector mismatch {} vs {}", rhs.data.len(), self.data.len());
        }
        let mut y = vec![0.0; n];
        unsafe {
            dgemv(b'T', n as i32, n as i32, 1.0, &rhs.data, n as i32, &self.data, 1, 0.0, &mut y, 1);
        }
        NVector { data: y, phantom: PhantomData }
    }
}

impl <'a, 'b> Mul<&'b NVector<Complex64, Col>> for &'a MatrixSquare<Complex64> {
    type Output = NVector<Complex64, Col>;
    
    fn mul(self, rhs: &'b NVector<Complex64, Col>) -> Self::Output {
        let n = rhs.data.len();
        if self.data.len() != n * n {
            panic!("Complex64 matrix-vector mismatch {} vs {}", self.data.len(), rhs.data.len());
        }
        let mut y = vec![zero(); n];
        unsafe {
            zgemv(b'N', n as i32, n as i32, one(), &self.data, n as i32, &rhs.data, 1, zero(), &mut y, 1);
        }
        NVector { data: y, phantom: PhantomData }
    }
}

impl <'a, 'b> Mul<&'b MatrixSquare<Complex64>> for &'a NVector<Complex64, Row> {
    type Output = NVector<Complex64, Row>;
    
    fn mul(self, rhs: &'b MatrixSquare<Complex64>) -> Self::Output {
        let n = self.data.len();
        if rhs.data.len() != n * n {
            panic!("Complex64 matrix-vector mismatch {} vs {}", rhs.data.len(), self.data.len());
        }
        let mut y = vec![zero(); n];
        unsafe {
            zgemv(b'T', n as i32, n as i32, one(), &rhs.data, n as i32, &self.data, 1, zero(), &mut y, 1);
        }
        NVector { data: y, phantom: PhantomData }
    }
}

impl MatrixSquare<f64> {
    pub fn dsyev(&self) -> (Vec<f64>, MatrixSquare<f64>) {
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


        (eigvals, MatrixSquare{ n: self.n, data: m })
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
                data.push(self[(j, i)].dagger());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_basics() {
        let a = NVector::row_from_vec(vec![1.0, 2.0, 3.0]);
        let b = NVector::row_from_vec(vec![1.0, 3.0, 5.0]);
        
        assert_eq!(&a + &b, NVector::row_from_vec(vec![2.0, 5.0, 8.0]));
        assert_eq!(&a + b.clone(), NVector::row_from_vec(vec![2.0, 5.0, 8.0]));
        assert_eq!(a.clone() + &b, NVector::row_from_vec(vec![2.0, 5.0, 8.0]));
        assert_eq!(a.clone() + b.clone(), NVector::row_from_vec(vec![2.0, 5.0, 8.0]));

        assert_eq!(&a - &b,        NVector::row_from_vec(vec![0.0, -1.0, -2.0]));
        assert_eq!(&a - b.clone(), &a - &b);
        assert_eq!(a.clone() - &b, &a - &b);
        assert_eq!(a.clone() - b.clone(), &a - &b);
        
        let c = NVector::row_from_vec(vec![1.0, 4.0, 9.0]);

        assert_eq!(&c * 1.5, NVector::row_from_vec(vec![1.5, 6.0, 13.5]));
        assert_eq!(1.5 * &c, NVector::row_from_vec(vec![1.5, 6.0, 13.5]));
        assert_eq!(c.clone() * 1.5, NVector::row_from_vec(vec![1.5, 6.0, 13.5]));
        assert_eq!(1.5 * c.clone(), NVector::row_from_vec(vec![1.5, 6.0, 13.5]));

        assert_eq!(c.len(), 3);
        assert_eq!(c.elts(), vec![1.0, 4.0, 9.0].as_slice());
    }
    
    #[test]
    fn matrix_basics() {
        let row0 = NVector::row_from_vec(vec![1.0, 2.0, 3.0]);
        let row1 = NVector::row_from_vec(vec![4.0, 5.0, 6.0]);
        let row2 = NVector::row_from_vec(vec![7.0, 8.0, 9.0]);

        let rows_in = vec![row0, row1, row2];

        let m = MatrixSquare::from_rows(&rows_in);

        assert_eq!(m.rows(), rows_in);

        for rowno in 0..3 {
            for colno in 0..3 {
                assert_eq!(m[(rowno, colno)], rows_in[rowno][colno]);
            }
        }

        let cols_out: Vec<NVector<f64,Row>> = m.dagger().cols().iter().map(Conjugate::dagger).collect();
        assert_eq!(cols_out, rows_in);
    }

    #[test]
    fn matrix_mult() {
        let row0 = NVector::row_from_vec(vec![1.0, 2.0]);
        let row1 = NVector::row_from_vec(vec![3.0, 4.0]);
        let m = MatrixSquare::from_rows(&vec![row0, row1]);

        let v = NVector::col_from_vec(vec![5.0, 6.0]);

        let w = m.mmulv(&v);
        assert_eq!(w[0], 1.0 * 5.0 + 2.0 * 6.0);
        assert_eq!(w[1], 3.0 * 5.0 + 4.0 * 6.0);

        let wexp = NVector::col_from_vec(vec![1.0 * 5.0 + 2.0 * 6.0, 3.0 * 5.0 + 4.0 * 6.0]);
        assert_eq!(&m * &v, wexp);
        
        let wexp = NVector::row_from_vec(vec![5.0 * 1.0 + 6.0 * 3.0, 5.0 * 2.0 + 6.0 * 4.0]);
        assert_eq!(&(v.dagger()) * &m, wexp);

        let row0a = NVector::row_from_vec(vec![7.0, 8.0]);
        let row1a = NVector::row_from_vec(vec![9.0, 1.0]);
        let n = MatrixSquare::from_rows(&vec![row0a, row1a]);

        let p = m.mmulm(&n);

        assert_eq!(p[(0, 0)], 7.0 * 1.0 + 9.0 * 2.0);
        assert_eq!(p[(1, 0)], 3.0 * 7.0 + 4.0 * 9.0);
        assert_eq!(p[(0, 1)], 1.0 * 8.0 + 2.0 * 1.0);
        assert_eq!(p[(1, 1)], 3.0 * 8.0 + 4.0 * 1.0);
    }
}
