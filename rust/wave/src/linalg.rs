#![recursion_limit="16"]
use std::borrow::Borrow;
use std::fmt::{Display,Formatter};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub};

use lapack::dsyev;
use num_complex::*;
use num_traits::{Num, NumOps};
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
    fn empty() -> Self {
        NVector { data: Vec::new(), phantom: PhantomData }
    }

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

impl <T> From<NVector<f64, T>> for NVector<Complex64, T> {
    fn from(v: NVector<f64, T>) -> Self {
        Self::from(&v)
    }
}

impl <'a, T> From<&'a NVector<f64, T>> for NVector<Complex64, T> {
    fn from(v: &'a NVector<f64, T>) -> Self {
        NVector { data: v.data.iter().map(Complex64::from).collect(), phantom: PhantomData }
    }
}

impl <E: Display> Display for NVector<E, Row> {
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

impl <E: Display> Display for NVector<E, Col> {
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
        write!(f, "]'")
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

macro_rules! nvector_val_val_binop {
    (impl $imp:ident, $method:ident, $lhsty:ty, $rhsty: ty, $output:ty, $foldzero: expr, $foldfunc: expr) => {
        impl <E: Copy + Num> $imp<$rhsty> for $lhsty
        {
            type Output = $output;

            #[inline]
            fn $method(self, rhs: $rhsty) -> Self::Output {
                if self.data.len() != rhs.data.len() {
                    panic!("NVector $imp length mismatch {} vs {}", self.data.len(), rhs.data.len());
                }
                self.data.iter().zip(rhs.data.iter()).fold($foldzero, $foldfunc)
            }
        }
    };
}

macro_rules! nvector_all_binop {
    (impl $imp:ident, $method:ident, $lhsty:ty, $rhsty:ty, $output:ty, $foldzero: expr, $foldfunc: expr) => {
        nvector_val_val_binop!(impl $imp, $method, $lhsty, $rhsty, $output, $foldzero, $foldfunc);
        nvector_val_val_binop!(impl $imp, $method, &$lhsty, $rhsty, $output, $foldzero, $foldfunc);
        nvector_val_val_binop!(impl $imp, $method, &$lhsty, &$rhsty, $output, $foldzero, $foldfunc);
        nvector_val_val_binop!(impl $imp, $method, $lhsty, &$rhsty, $output, $foldzero, $foldfunc);
    };
}

// NVector + NVector
nvector_all_binop!(impl Add, add, NVector<E, Row>, NVector<E, Row>, NVector<E, Row>, NVector::empty(), |mut acc, (&x, &y)| { acc.data.push(x + y); acc });
nvector_all_binop!(impl Add, add, NVector<E, Col>, NVector<E, Col>, NVector<E, Col>, NVector::empty(), |mut acc, (&x, &y)| { acc.data.push(x + y); acc });

// NVector - NVector
nvector_all_binop!(impl Sub, sub, NVector<E, Row>, NVector<E, Row>, NVector<E, Row>, NVector::empty(), |mut acc, (&x, &y)| { acc.data.push(x - y); acc });
nvector_all_binop!(impl Sub, sub, NVector<E, Col>, NVector<E, Col>, NVector<E, Col>, NVector::empty(), |mut acc, (&x, &y)| { acc.data.push(x - y); acc });

// NVector * NVector is dot product
nvector_all_binop!(impl Mul, mul, NVector<E, Row>, NVector<E, Col>, E, zero(), |mut acc, (&x, &y)| { acc = acc + (x * y); acc });

macro_rules! nvector_scalar_pdt {
    (impl $vecty:ty, $scaty: ty, $outeltty:ty) => {
        impl <T> Mul<$vecty> for $scaty
        {
            type Output = NVector<$outeltty, T>;

            #[inline]
            fn mul(self, rhs: $vecty) -> Self::Output {
                let y: Vec<$outeltty> = rhs.data.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(self)).collect();
                NVector { data: y, phantom: PhantomData }
            }
        }

        impl <T> Mul<$scaty> for $vecty
        {
            type Output = NVector<$outeltty, T>;

            #[inline]
            fn mul(self, rhs: $scaty) -> Self::Output {
                let y: Vec<$outeltty> = self.data.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(rhs)).collect();
                NVector { data: y, phantom: PhantomData }
            }
        }
    };
}

macro_rules! nvector_scalar_all_pdt {
    (impl $vecty:ty, $scaty: ty, $outeltty: ty) => {
        nvector_scalar_pdt!(impl $vecty, $scaty, $outeltty);
        nvector_scalar_pdt!(impl &$vecty, $scaty, $outeltty);
    }
}

// NVector * scalar
nvector_scalar_all_pdt!(impl NVector<f64, T>, f64, f64);
nvector_scalar_all_pdt!(impl NVector<Complex64, T>, f64, Complex64);
nvector_scalar_all_pdt!(impl NVector<Complex64, T>, Complex64, Complex64);

#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct MatrixSquare<E> {
    n: usize,
    data: Vec<E>,
}

impl<E> MatrixSquare<E> {
    pub fn n(&self) -> usize { self.n }

    #[inline]
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

    #[inline]
    fn data_rowcol(&self, index: usize) -> (usize, usize) {
        if index < 0 || index >= (self.n * self.n) {
            panic!("Data offset {} for {}x{} MatrixSquare", index, self.n, self.n);
        }
        (index % self.n, index / self.n)
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

impl<E: Zero> MatrixSquare<E>
{
    pub fn zeros(n: usize) -> Self {
        let mut data = Vec::with_capacity(n*n);
        for _i in 0..(n*n) {
            data.push(zero());
        }
        MatrixSquare {
            n: n,
            data: data,
        }
    }
}

impl<E: Zero + One> MatrixSquare<E> {
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
        Self::from(&v)
    }
}

impl <'a> From<&'a MatrixSquare<f64>> for MatrixSquare<Complex64> {    
    fn from(v: &'a MatrixSquare<f64>) -> Self {
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

macro_rules! matrix_val_val_binop {
    (impl $imp:ident, $method:ident, $lhsty:ty, $rhsty: ty, $output:ty, $foldzero: expr, $foldfunc: expr) => {
        impl <E: Copy + Num> $imp<$rhsty> for $lhsty
        {
            type Output = $output;

            #[inline]
            fn $method(self, rhs: $rhsty) -> Self::Output {
                if self.data.len() != rhs.data.len() {
                    panic!("NVector $imp length mismatch {} vs {}", self.data.len(), rhs.data.len());
                }
                self.data.iter().zip(rhs.data.iter()).fold($foldzero, $foldfunc)
            }
        }
    };
}

macro_rules! matrix_binop_refs {
    (impl $imp:ident, $method:ident) => {
        impl <E: Copy + Num> $imp<&MatrixSquare<E>> for MatrixSquare<E> {
            type Output = MatrixSquare<E>;
            #[inline]
            fn $method(self, rhs: &MatrixSquare<E>) -> Self::Output {
                self.$method(rhs)
            }
        }
        
        impl <E: Copy + Num> $imp<MatrixSquare<E>> for &MatrixSquare<E> {
            type Output = MatrixSquare<E>;
            #[inline]
            fn $method(self, rhs: MatrixSquare<E>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl <E: Copy + Num> $imp<MatrixSquare<E>> for MatrixSquare<E> {
            type Output = MatrixSquare<E>;
            #[inline]
            fn $method(self, rhs: MatrixSquare<E>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

impl <E: Copy + Num> Add<&MatrixSquare<E>> for &MatrixSquare<E> {
    type Output = MatrixSquare<E>;
    #[inline]
    fn add(self, rhs: &MatrixSquare<E>) -> Self::Output {
        if self.n != rhs.n {
            panic!("MatrixSquare add size mismatch {} vs {}", self.n, rhs.n);
        }
        let out: Vec<E> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x + y).collect();
        MatrixSquare { n: self.n, data: out }
    }
}

matrix_binop_refs!(impl Add, add);

impl <E: Copy + Num> Sub<&MatrixSquare<E>> for &MatrixSquare<E> {
    type Output = MatrixSquare<E>;
    #[inline]
    fn sub(self, rhs: &MatrixSquare<E>) -> Self::Output {
        if self.n != rhs.n {
            panic!("MatrixSquare sub size mismatch {} vs {}", self.n, rhs.n);
        }
        let out: Vec<E> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x + y).collect();
        MatrixSquare { n: self.n, data: out }
    }
}

matrix_binop_refs!(impl Sub, sub);

impl <E: Copy + Num> Mul<&MatrixSquare<E>> for &MatrixSquare<E> {
    type Output = MatrixSquare<E>;
    #[inline]
    fn mul(self, rhs: &MatrixSquare<E>) -> Self::Output {
        if self.n != rhs.n {
            panic!("MatrixSquare sub size mismatch {} vs {}", self.n, rhs.n);
        }
        let mut out: Vec<E> = Vec::with_capacity(self.n * self.n);
        for idx in 0..(self.n * self.n) {
            let (i, j) = self.data_rowcol(idx);
            let out_ij = (0..self.n).fold(zero(), |mut acc, k| { acc = acc + self[(i, k)] * rhs[(k, j)]; acc });
            out.push(out_ij);
        }
        MatrixSquare { n: self.n, data: out }
    }
}

matrix_binop_refs!(impl Mul, mul);

macro_rules! matrix_vector_binop_refs {
    (impl $imp:ident, $method:ident, $lhsty:ty, $rhsty:ty, $outty:ty) => {
        impl <E: Copy + Num> $imp<&$rhsty> for $lhsty {
            type Output = $outty;
            #[inline]
            fn $method(self, rhs: &$rhsty) -> Self::Output {
                self.$method(rhs)
            }
        }
        
        impl <E: Copy + Num> $imp<$rhsty> for &$lhsty {
            type Output = $outty;
            #[inline]
            fn $method(self, rhs: $rhsty) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl <E: Copy + Num> $imp<$rhsty> for $lhsty {
            type Output = $outty;
            #[inline]
            fn $method(self, rhs: $rhsty) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

impl <E: Copy + Num> Mul<&NVector<E, Col>> for &MatrixSquare<E> {
    type Output = NVector<E, Col>;
    #[inline]
    fn mul(self, rhs: &NVector<E, Col>) -> Self::Output {
        if self.n != rhs.len() {
            panic!("MatrixSquare Vector mul size mismatch {} vs {}", self.n, rhs.len());
        }
        let mut out: Vec<E> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let out_i = (0..self.n).fold(zero(), |mut acc, j| { acc = acc + self[(i, j)] * rhs[j]; acc });
            out.push(out_i);
        }
        NVector { data: out, phantom: PhantomData }
    }
}

matrix_vector_binop_refs!(impl Mul, mul, MatrixSquare<E>, NVector<E, Col>, NVector<E, Col>);

impl <E: Copy + Num> Mul<&MatrixSquare<E>> for &NVector<E, Row> {
    type Output = NVector<E, Row>;
    #[inline]
    fn mul(self, rhs: &MatrixSquare<E>) -> Self::Output {
        if self.len() != rhs.n {
            panic!("MatrixSquare Vector mul size mismatch {} vs {}", self.len(), rhs.n);
        }
        let mut out: Vec<E> = Vec::with_capacity(rhs.n);
        for j in 0..rhs.n {
            let out_j = (0..rhs.n).fold(zero(), |mut acc, i| { acc = acc + self[i] * rhs[(i, j)]; acc });
            out.push(out_j);
        }
        NVector { data: out, phantom: PhantomData }
    }
}

matrix_vector_binop_refs!(impl Mul, mul, NVector<E, Row>, MatrixSquare<E>, NVector<E, Row>);

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

        let w = &m * &v;
        assert_eq!(w[0], 1.0 * 5.0 + 2.0 * 6.0);
        assert_eq!(w[1], 3.0 * 5.0 + 4.0 * 6.0);

        let wexp = NVector::col_from_vec(vec![1.0 * 5.0 + 2.0 * 6.0, 3.0 * 5.0 + 4.0 * 6.0]);
        assert_eq!(&m * &v, wexp);
        
        let wexp = NVector::row_from_vec(vec![5.0 * 1.0 + 6.0 * 3.0, 5.0 * 2.0 + 6.0 * 4.0]);
        assert_eq!(&(v.dagger()) * &m, wexp);

        let row0a = NVector::row_from_vec(vec![7.0, 8.0]);
        let row1a = NVector::row_from_vec(vec![9.0, 1.0]);
        let n = MatrixSquare::from_rows(&vec![row0a, row1a]);

        let p = &m * &n;

        assert_eq!(p[(0, 0)], 7.0 * 1.0 + 9.0 * 2.0);
        assert_eq!(p[(1, 0)], 3.0 * 7.0 + 4.0 * 9.0);
        assert_eq!(p[(0, 1)], 1.0 * 8.0 + 2.0 * 1.0);
        assert_eq!(p[(1, 1)], 3.0 * 8.0 + 4.0 * 1.0);
    }
}

