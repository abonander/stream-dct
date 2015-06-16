//! A Rust library for allocation-limited computation of the Discrete Cosine Transform.
//!
//! 1D DCTs are allocation-free but 2D requires allocation.
//!
//! Features:
//!
//! * `simd`: use SIMD types to speed computation (2D DCT only)
//! * `cos-approx`: use a Taylor series approximation of cosine instead of the stdlib
//! implementation (which is usually much slower but also higher precision)

use std::f64::consts::{PI, SQRT_2};
use std::ops::Range;

/// An allocation-free one-dimensional Discrete Cosine Transform.
///
/// Each iteration produces the next DCT value in the sequence.
#[derive(Clone, Debug)]
pub struct DCT1D<'a> {
    data: &'a [f64],
    curr: Range<usize>,
}

impl<'a> DCT1D<'a> {
    /// Create a new DCT 1D adaptor from a 1D vector of data.
    pub fn new(data: &[f64]) -> DCT1D {
        let curr = 0 .. data.len();

        DCT1D {
            data: data,
            curr: curr,
        }
    }

    // Converted from the C implementation here:
    // http://unix4lyfe.org/dct/listing2.c
    // Source page:
    // http://unix4lyfe.org/dct/ (Accessed 8/10/2014)
    fn next_dct_val(&mut self) -> Option<f64> {
        self.curr.next().map(|u| {
            let mut z = 0.0;

            let data_len = self.data.len();

            for (x_idx, &x) in self.data.iter().enumerate() {
                z += x * cos(
                    PI * u as f64 * (2 * x_idx + 1) as f64 
                    / (2 * data_len) as f64
                );
            } 

            if u == 0 {
                z *= 1.0 / SQRT_2;
            }

            z / 2.0
        })
    }
}

impl<'a> Iterator for DCT1D<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.next_dct_val()
    }
}

/// An implementation of cosine that switches to a Taylor-series approximation when throughput is
/// preferred over precision.
#[inline(always)]
pub fn cos(x: f64) -> f64 {
    // This branch should be optimized out.
    if cfg!(feature = "cos-approx") {
        // Normalize to [0, pi] or else the Taylor series spits out very wrong results.
        let x = (x.abs() + PI) % (2.0 * PI) - PI;

        // Approximate the cosine of `val` using a 4-term Taylor series.
        // Can be expanded for higher precision.
        let x2 = x.powi(2);
        let x4 = x.powi(4);
        let x6 = x.powi(6);
        let x8 = x.powi(8);

        1.0 - (x2 / 2.0) + (x4 / 24.0) - (x6 / 720.0) + (x8 / 40320.0)
    } else {
        x.cos()
    }
}

/// Perform a 2D DCT on a 1D-packed vector with a given rowstride.
///
/// E.g. a vector of length 9 with a rowstride of 3 will be processed as a 3x3 matrix.
///
/// Returns a vector of the same size packed in the same way.
pub fn dct_2d(packed_2d: &[f64], rowstride: usize) -> Vec<f64> {
    assert_eq!(packed_2d.len() % rowstride, 0);

    let mut row_dct: Vec<f64> = packed_2d
        .chunks(rowstride)
        .flat_map(DCT1D::new)
        .collect();

    swap_rows_columns(&mut row_dct, rowstride);

    let mut column_dct: Vec<f64> = packed_2d
        .chunks(rowstride)
        .flat_map(DCT1D::new)
        .collect();

    swap_rows_columns(&mut column_dct, rowstride);

    column_dct
}

fn swap_rows_columns(data: &mut [f64], rowstride: usize) {
    let height = data.len() / rowstride;

    for y in 0 .. height {
        for x in 0 .. rowstride {
            data.swap(y * rowstride + x, x * rowstride + y);
        }
    }    
}

#[cfg_attr(all(test, feature = "cos-approx"), test)]
#[cfg_attr(not(all(test, feature = "cos-approx")), allow(dead_code))]
fn test_cos_approx() {
    const ERROR: f64 = 0.05;

    fn test_cos_approx(x: f64) {
        let approx = cos(x);
        let cos = x.cos();

        assert!(
            approx.abs_sub(x.cos()) <= ERROR, 
            "Approximation cos({x}) = {approx} was outside a tolerance of {error}; control value: {cos}",
            x = x, approx = approx, error = ERROR, cos = cos,
        );
    }

    let test_values = [PI, PI / 2.0, PI / 4.0, 1.0, -1.0, 2.0 * PI, 3.0 * PI, 4.0 / 3.0 * PI];

    for &x in &test_values {
        test_cos_approx(x);
        test_cos_approx(-x);
    }
}

/*
#[cfg(feature = "simd")]
mod dct_simd {
    use simdty::f64x2;

    use std::f64::consts::{PI, SQRT_2};
    
    macro_rules! valx2 ( ($val:expr) => ( ::simdty::f64x2($val, $val) ) );

    const PI: f64x2 = valx2!(PI);
    const ONE_DIV_SQRT_2: f64x2 = valx2!(1 / SQRT_2);
    const SQRT_2: f64x2 = valx2!(SQRT_2);

    pub dct_rows(vals: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(vals.len());

        for pair in vals.iter().chunks(2) {
            if pair.len() == 2 {
                let vals = pair[0].iter().cloned().zip(pair[1].iter().cloned())
                    .map(f64x2)
                    .collect();

                dct_1dx2(vals);


        
        }
    }

    fn dct_1dx2(vec: Vec<f64x2>) -> Vec<f64x2> {
        let mut out = Vec::with_capacity(vec.len());

        for u in 0 .. vec.len() {
            let mut z = valx2!(0.0);

            for x in 0 .. vec.len() {
                z += vec[x] * cos_approx(
                    PI * valx2!(
                        u as f64 * (2 * x + 1) as f64 
                            / (2 * vec.len()) as f64
                    )
                );
            }

            if u == 0 {
                z *= ONE_DIV_SQRT_2;
            }

            out.insert(u, z / valx2!(2.0));
        }

        out 
    }

    fn cos_approx(x2: f64x2) -> f64x2 {
        #[inline(always)]
        fn powi(val: f64x2, pow: i32) -> f64x2 {
            unsafe { llvmint::powi_v2f64(val, pow) }
        }

        let x2 = powi(val, 2);
        let x4 = powi(val, 4);
        let x6 = powi(val, 6);
        let x8 = powi(val, 8);

        valx2!(1.0) - (x2 / valx2!(2.0)) + (x4 / valx2!(24.0)) 
            - (x6 / valx2!(720.0)) + (x8 / valx2!(40320.0))
    }
}
*/

