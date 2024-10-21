//! Provides numerical differentiation routines for functions of multiple variables.
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::common::{Vector, Matrix};
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------


const EPS: f64 = 1e-8;
const ABS_EPS_SMALL: f64 = 1e-10;    
const REL_EPS_SMALL: f64 = 1e-6;
const REL_EPS_LARGE: f64 = 1e-3;


pub struct GradOptions {
    pub periods: Option<Vec<f64>>,
    pub bounds: Option<Vec<(f64, f64)>>,
}

pub fn grad<F: Fn(&Vector) -> f64>(f: F, x_in: &Vector) -> Vector {
    //{{{ trace
    debug!("x_in = {}", x_in);
    //}}}
    let mut x = x_in.clone();
    let f_x = f(&x).abs();
    let mut grad = Vector::zeros(x.len());
    let mut f_plus = 0.0f64;
    let mut f_minus = 0.0f64;
    let jmax = 15;

    //{{{ trace
    debug!("f_x = {}", f_x);
    //}}}

    for i in 0..x.len() {
        let mut found = false;
        let mut h = EPS * (1.0 + x[i].abs());

        //{{{ trace
        trace!("............................................................. i = {}", i);
        trace!("h = {}", h);
        //}}}
        let mut j = 0;
        while !found && j < jmax {
            //{{{ trace
            trace!("......................... j = {}", j);
            //}}}
            let xi = x[i];
            x[i] += h;
            f_plus = f(&x);
            x[i] -= 2.0 * h;
            f_minus = f(&x);
            x[i] = xi;
            let delta_f = (f_plus - f_minus).abs();
            let rel_delta_f =  if delta_f < ABS_EPS_SMALL {
                delta_f
            } 
            else {
                delta_f / f_x
            };

            //{{{ trace
            trace!("f_plus = {} f_minus = {}", f_plus, f_minus);
            trace!("delta_f = {} rel_delta_f = {}", delta_f, rel_delta_f);
            //}}}

            if rel_delta_f < REL_EPS_SMALL {
                //{{{ trace
                trace!("rel_delta_f is small, enlarging h = {}", h);
                //}}}
                h *= 2.0;
            } else if rel_delta_f > REL_EPS_LARGE {
                //{{{ trace
                trace!("rel_delta_f is large, shortening h = {}", h);
                //}}}
                h *= 0.5;
            } else {
                //{{{ trace
                trace!("found an acceptable h = {}", h);
                //}}}
                found = true;
            }
            j += 1;
        }
        grad[i] = (f_plus - f_minus) / (2.0 * h);
        //{{{ trace
        trace!("grad[i] = {}", grad[i]);
        //}}}
    }
    //{{{ trace
    info!("grad = {}", grad);
    //}}}
    grad
}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {

    use core::f64;

    use approx::assert_relative_eq;
    use super::*;

    const MAX_REL: f64 = 1e-4;


    //{{{ macro: quadratic_test
    macro_rules! quadratic_test {
        ($test_name: ident, $point: expr) => {
            #[test]
            fn $test_name() {

                let f = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = (x - 1.0).powi(2) * (y - 2.0).powi(2);    
                    out
                };

                let grad_f_true = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = Vector::from_column_slice(&[
                        ((x - 1.0) * 2.0)*(y - 2.0).powi(2),
                        (x - 1.0).powi(2) * ((y - 2.0) * 2.0),
                    ]);
                    out
                };

                let x = Vector::from_column_slice(&$point);
                let grad_fx1 = grad(f, &x);
                let grad_fx2 = grad_f_true(&x);
                for i in 0..grad_fx1.len() {
                    assert_relative_eq!(grad_fx1[i], grad_fx2[i], max_relative = MAX_REL);
                }
            }
        };
    }
    //}}}
    //{{{ collection: quadratic_tests
    quadratic_test!(test_quadratic_1, [0.0,0.0]);
    quadratic_test!(test_quadratic_2, [1.0e5,1.0e5]);
    quadratic_test!(test_quadratic_3, [-1.0e5,-1.0e5]);
    quadratic_test!(test_quadratic_4, [1.0, 2.0]);
    quadratic_test!(test_quadratic_5, [1.0e10, 1.0e10]);
    quadratic_test!(test_quadratic_7, [-1.0e10, -1.0e10]);
    //}}}
    //{{{ macro: sin_test 
    macro_rules! sin_test {
        ($test_name: ident, $point: expr) => {
            #[test]
            fn $test_name() {

                let f = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = x.sin() * y.sin();
                    out
                };

                let grad_f_true = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = Vector::from_column_slice(&[
                        x.cos() * y.sin(),
                        x.sin() * y.cos()
                    ]);
                    out
                };

                let x = Vector::from_column_slice(&$point);
                let grad_fx1 = grad(f, &x);
                let grad_fx2 = grad_f_true(&x);
                for i in 0..grad_fx1.len() {
                    assert_relative_eq!(grad_fx1[i], grad_fx2[i], max_relative = MAX_REL, epsilon = 1e-10);
                }
            }
        };
    }
    //}}}
    //{{{ collection: sin_tests
    sin_test!(test_sin_1, [1.0, 1.0]);
    sin_test!(test_sin_2, [0.0, 0.0]);
    sin_test!(test_sin_3, [10000.0 * f64::consts::PI, 10000.0 * f64::consts::PI]);
    sin_test!(test_sin_4, [-10000.0 * f64::consts::PI, -10000.0 * f64::consts::PI]);
    //}}}
    //{{{ macro: log_test 
    macro_rules! log_test {
        ($test_name: ident, $point: expr) => {
            #[test]
            fn $test_name() {

                let f = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = x.ln() * y.ln();
                    out
                };

                let grad_f_true = |x_vec: &Vector| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = Vector::from_column_slice(&[
                        (1.0 / x) * y.ln(),
                        x.ln() * (1.0 / y)
                    ]);
                    out
                };

                let x = Vector::from_column_slice(&$point);
                let grad_fx1 = grad(f, &x);
                let grad_fx2 = grad_f_true(&x);
                for i in 0..grad_fx1.len() {
                    assert_relative_eq!(grad_fx1[i], grad_fx2[i], max_relative = MAX_REL, epsilon = 1e-10);
                }
            }
        };
    }
    //}}}
    //{{{ collection: log_tests

    log_test!(test_log_1, [1.0, 1.0]);
    log_test!(test_log_2, [1.0e10, 1.0e10]);
    //}}}
}
//}}}