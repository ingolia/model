extern crate accelerate_src;

use std::f64::consts;

use blas::*;
use lapack::*;
use num_complex::*;

mod linalg;

use linalg::*;

fn main() {
    let a = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, consts::FRAC_1_SQRT_2),
        Complex::new(0.0, -consts::FRAC_1_SQRT_2),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, consts::FRAC_1_SQRT_2),
        Complex::new(0.0, consts::FRAC_1_SQRT_2),
    ];

    let b = vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
    ];

    let mut c = vec![Complex::new(-1.0, -1.0); 3];

    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);

    unsafe {
        zgemv(
            b'N',
            3,
            3,
            Complex::new(1.0, 0.0),
            &a,
            3,
            &b,
            1,
            Complex::new(0.0, 0.0),
            &mut c,
            1,
        );
    }

    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);

    let m = MatrixSquare::from_rows(
        &vec![NVector::row_from_vec(vec![ 1.0,  1.0,  0.0,  0.0 ]),
              NVector::row_from_vec(vec![ 1.0, -1.0,  0.0,  0.0 ]),
              NVector::row_from_vec(vec![ 0.0,  0.0,  1.0, -1.0 ]),
              NVector::row_from_vec(vec![ 0.0,  0.0, -1.0,  1.0 ])]);

    println!("{:6.3}", m);
    
    let (eigvals, eigvecs) = m.dsyev();

    println!("{:.4}", NVector::row_from_vec(eigvals));

    for eigvec in eigvecs {
        println!("{:7.4}", eigvec);
    }
}
