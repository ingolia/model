use std::fmt::{Display,Formatter};
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use lapack::dsyev;
use num_complex::*;
use num_traits::Num;
use num_traits::identities::*;

/// A mathematical entity with a conjugate transpose
pub trait Conjugate {
    /// The resulting type after conjugate transpose
    type Output;

    /// Returns the conjugate transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::*;
    /// let z = Complex::new(+1.0, +1.0);
    /// let zdag = z.dagger();
    /// assert_eq!(zdag, -1.0);
    /// ```
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

pub trait NVector<E> {
    fn len(&self) -> usize;
    fn elts(&self) -> &[E];
}

/// Numerical column vector for linear algebra. The vector is
/// parameterized over an element type `E`.
#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct CVector<E>(Vec<E>);

/// Numerical row vector for linear algebra. The vector is
/// parameterized over an element type `E`.
#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct RVector<E>(Vec<E>);

macro_rules! nvector_type {
    (impl $vtype:ident) => {

        impl <E> NVector<E> for $vtype<E> {
            fn len(&self) -> usize {
                self.0.len()
            }
            
            fn elts(&self) -> &[E] {
                self.0.as_slice()
            }
        }

        impl <E> From<Vec<E>> for $vtype<E> {
            fn from(v: Vec<E>) -> Self {
                $vtype(v)
            }
        }

        impl<E> Index<usize> for $vtype<E> {
            type Output = E;
            
            fn index(&self, idx: usize) -> &E {
                &self.0[idx]
            }
        }

        impl<E> IndexMut<usize> for $vtype<E> {
            fn index_mut(&mut self, idx: usize) -> &mut E {
                &mut self.0[idx]
            }
        }

        impl From<$vtype<f64>> for $vtype<Complex64> {
            fn from(v: $vtype<f64>) -> Self {
                Self::from(&v)
            }
        }

        impl <'a> From<&'a $vtype<f64>> for $vtype<Complex64> {
            fn from(v: &'a $vtype<f64>) -> Self {
                $vtype(v.0.iter().map(Complex64::from).collect())
            }
        }
    }
}

nvector_type!(impl CVector);
nvector_type!(impl RVector);
    
impl <E: Display> Display for RVector<E> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let mut iter = self.0.iter();

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

impl <E: Display> Display for CVector<E> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let mut iter = self.0.iter();

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

impl<E: Conjugate> Conjugate for RVector<E> {
    type Output = CVector<E::Output>;
    fn dagger(&self) -> Self::Output {
        CVector(self.0.iter().map(Conjugate::dagger).collect())
    }
}

impl<E: Conjugate> Conjugate for CVector<E> {
    type Output = RVector<E::Output>;
    fn dagger(&self) -> Self::Output {
        RVector(self.0.iter().map(Conjugate::dagger).collect())
    }
}

macro_rules! nvector_val_val_binop {
    (impl $imp:ident, $method:ident, $lhsty:ty, $rhsty: ty, $output:ty, $foldzero: expr, $foldfunc: expr) => {
        impl <E: Copy + Num> $imp<$rhsty> for $lhsty
        {
            type Output = $output;

            #[inline]
            fn $method(self, rhs: $rhsty) -> Self::Output {
                if self.0.len() != rhs.0.len() {
                    panic!("NVector $imp length mismatch {} vs {}", self.0.len(), rhs.0.len());
                }
                self.0.iter().zip(rhs.0.iter()).fold($foldzero, $foldfunc)
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
nvector_all_binop!(impl Add, add, RVector<E>, RVector<E>, RVector<E>, RVector(Vec::new()), |mut acc, (&x, &y)| { acc.0.push(x + y); acc });
nvector_all_binop!(impl Add, add, CVector<E>, CVector<E>, CVector<E>, CVector(Vec::new()), |mut acc, (&x, &y)| { acc.0.push(x + y); acc });

// NVector - NVector
nvector_all_binop!(impl Sub, sub, RVector<E>, RVector<E>, RVector<E>, RVector(Vec::new()), |mut acc, (&x, &y)| { acc.0.push(x - y); acc });
nvector_all_binop!(impl Sub, sub, CVector<E>, CVector<E>, CVector<E>, CVector(Vec::new()), |mut acc, (&x, &y)| { acc.0.push(x - y); acc });

// NVector * NVector is dot product
nvector_all_binop!(impl Mul, mul, RVector<E>, CVector<E>, E, zero(), |mut acc, (&x, &y)| { acc = acc + (x * y); acc });

macro_rules! nvector_scalar_pdt {
    (impl $vec: ident, $eltty: ty, $scaty: ty, $outeltty:ty) => {
        impl Mul<$vec<$eltty>> for $scaty
        {
            type Output = $vec<$outeltty>;

            #[inline]
            fn mul(self, rhs: $vec<$eltty>) -> Self::Output {
                let y: Vec<$outeltty> = rhs.0.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(self)).collect();
                $vec(y)
            }
        }

        impl Mul<&$vec<$eltty>> for $scaty
        {
            type Output = $vec<$outeltty>;

            #[inline]
            fn mul(self, rhs: &$vec<$eltty>) -> Self::Output {
                let y: Vec<$outeltty> = rhs.0.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(self)).collect();
                $vec(y)
            }
        }

        impl Mul<$scaty> for $vec<$eltty>
        {
            type Output = $vec<$outeltty>;

            #[inline]
            fn mul(self, rhs: $scaty) -> Self::Output {
                let y: Vec<$outeltty> = self.0.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(rhs)).collect();
                $vec(y)
            }
        }

        impl Mul<$scaty> for &$vec<$eltty>
        {
            type Output = $vec<$outeltty>;

            #[inline]
            fn mul(self, rhs: $scaty) -> Self::Output {
                let y: Vec<$outeltty> = self.0.iter().map(|&x| <$outeltty>::from(x) * <$outeltty>::from(rhs)).collect();
                $vec(y)
            }
        }

    };
}

// NVector * scalar
nvector_scalar_pdt!(impl CVector, f64, f64, f64);
nvector_scalar_pdt!(impl CVector, Complex64, f64, Complex64);
nvector_scalar_pdt!(impl CVector, Complex64, Complex64, Complex64);

nvector_scalar_pdt!(impl RVector, f64, f64, f64);
nvector_scalar_pdt!(impl RVector, Complex64, f64, Complex64);
nvector_scalar_pdt!(impl RVector, Complex64, Complex64, Complex64);

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
        if index >= (self.n * self.n) {
            panic!("Data offset {} for {}x{} MatrixSquare", index, self.n, self.n);
        }
        (index % self.n, index / self.n)
    }
}

impl <E: Copy> MatrixSquare<E> {
    pub fn rows(&self) -> Vec<RVector<E>> {
        let mut rows = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut row = Vec::with_capacity(self.n);
            for j in 0..self.n {
                row.push(self[(i, j)]);
            }
            rows.push(RVector::from(row));
        }
        rows
    }

    pub fn cols(&self) -> Vec<CVector<E>> {
        let mut cols = Vec::with_capacity(self.n);
        for j in 0..self.n {
            let mut col = Vec::with_capacity(self.n);
            for i in 0..self.n {
                col.push(self[(i, j)]);
            }
            cols.push(CVector::from(col));
        }
        cols
    }

    pub fn from_rows(rows: &[RVector<E>]) -> Self {
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

macro_rules! matrix_binop_refs {
    (impl $imp:ident, $method:ident) => {
        impl <E: Copy + Num> $imp<&MatrixSquare<E>> for MatrixSquare<E> {
            type Output = MatrixSquare<E>;
            #[inline]
            fn $method(self, rhs: &MatrixSquare<E>) -> Self::Output {
                (&self).$method(rhs)
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
                (&self).$method(&rhs)
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
                (&self).$method(rhs)
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
                (&self).$method(&rhs)
            }
        }
    };
}

impl <E: Copy + Num> Mul<&CVector<E>> for &MatrixSquare<E> {
    type Output = CVector<E>;
    #[inline]
    fn mul(self, rhs: &CVector<E>) -> Self::Output {
        if self.n != rhs.len() {
            panic!("MatrixSquare Vector mul size mismatch {} vs {}", self.n, rhs.len());
        }
        let mut out: Vec<E> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let out_i = (0..self.n).fold(zero(), |mut acc, j| { acc = acc + self[(i, j)] * rhs[j]; acc });
            out.push(out_i);
        }
        CVector(out)
    }
}

matrix_vector_binop_refs!(impl Mul, mul, MatrixSquare<E>, CVector<E>, CVector<E>);

impl <E: Copy + Num> Mul<&MatrixSquare<E>> for &RVector<E> {
    type Output = RVector<E>;
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
        RVector(out)
    }
}

matrix_vector_binop_refs!(impl Mul, mul, RVector<E>, MatrixSquare<E>, RVector<E>);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_vector {
        (test $name:ident, $vec:ident) => {
            #[test]
            fn $name() {
                let a = $vec::from(vec![1.0, 2.0, 3.0]);
                let b = $vec::from(vec![1.0, 3.0, 5.0]);
                
                assert_eq!(&a + &b, $vec::from(vec![2.0, 5.0, 8.0]));
                assert_eq!(&a + b.clone(), $vec::from(vec![2.0, 5.0, 8.0]));
                assert_eq!(a.clone() + &b, $vec::from(vec![2.0, 5.0, 8.0]));
                assert_eq!(a.clone() + b.clone(), $vec::from(vec![2.0, 5.0, 8.0]));
                
                assert_eq!(&a - &b,        $vec::from(vec![0.0, -1.0, -2.0]));
                assert_eq!(&a - b.clone(), &a - &b);
                assert_eq!(a.clone() - &b, &a - &b);
                assert_eq!(a.clone() - b.clone(), &a - &b);
                
                let c = $vec::from(vec![1.0, 4.0, 9.0]);
                
                assert_eq!(&c * 1.5, $vec::from(vec![1.5, 6.0, 13.5]));
                assert_eq!(1.5 * &c, $vec::from(vec![1.5, 6.0, 13.5]));
                assert_eq!(c.clone() * 1.5, $vec::from(vec![1.5, 6.0, 13.5]));
                assert_eq!(1.5 * c.clone(), $vec::from(vec![1.5, 6.0, 13.5]));
                
                assert_eq!(c.len(), 3);
                assert_eq!(c.elts(), vec![1.0, 4.0, 9.0].as_slice());
            }
        };
    }

    test_vector!(test test_rvector, RVector);
    test_vector!(test test_cvector, CVector);
    
    #[test]
    fn matrix_basics() {
        let row0 = RVector::from(vec![1.0, 2.0, 3.0]);
        let row1 = RVector::from(vec![4.0, 5.0, 6.0]);
        let row2 = RVector::from(vec![7.0, 8.0, 9.0]);

        let rows_in = vec![row0, row1, row2];

        let m = MatrixSquare::from_rows(&rows_in);

        assert_eq!(m.rows(), rows_in);

        for rowno in 0..3 {
            for colno in 0..3 {
                assert_eq!(m[(rowno, colno)], rows_in[rowno][colno]);
            }
        }

        let cols_out: Vec<RVector<f64>> = m.dagger().cols().iter().map(Conjugate::dagger).collect();
        assert_eq!(cols_out, rows_in);
    }

    #[test]
    fn matrix_mult() {
        let row0 = RVector::from(vec![1.0, 2.0]);
        let row1 = RVector::from(vec![3.0, 4.0]);
        let m = MatrixSquare::from_rows(&vec![row0, row1]);

        let v = CVector::from(vec![5.0, 6.0]);

        let w = &m * &v;
        assert_eq!(w[0], 1.0 * 5.0 + 2.0 * 6.0);
        assert_eq!(w[1], 3.0 * 5.0 + 4.0 * 6.0);

        let wexp = CVector::from(vec![1.0 * 5.0 + 2.0 * 6.0, 3.0 * 5.0 + 4.0 * 6.0]);
        assert_eq!(&m * &v, wexp);
        
        let wexp = RVector::from(vec![5.0 * 1.0 + 6.0 * 3.0, 5.0 * 2.0 + 6.0 * 4.0]);
        assert_eq!(&(v.dagger()) * &m, wexp);

        let row0a = RVector::from(vec![7.0, 8.0]);
        let row1a = RVector::from(vec![9.0, 1.0]);
        let n = MatrixSquare::from_rows(&vec![row0a, row1a]);

        let p = &m * &n;

        assert_eq!(p[(0, 0)], 7.0 * 1.0 + 9.0 * 2.0);
        assert_eq!(p[(1, 0)], 3.0 * 7.0 + 4.0 * 9.0);
        assert_eq!(p[(0, 1)], 1.0 * 8.0 + 2.0 * 1.0);
        assert_eq!(p[(1, 1)], 3.0 * 8.0 + 4.0 * 1.0);
    }
}

