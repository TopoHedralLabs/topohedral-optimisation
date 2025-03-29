//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg as la;
use topohedral_tracing::*;
//}}}
//{{{ exports
pub type SVector<const N: usize> = la::SCVector<f64, N>;
pub type SMatrix<const N: usize, const M: usize> = la::SMatrix<f64, N, M>;
pub use la::EvaluateSMatrix;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ constants
const EPS: f64 = 1e-8;
const ABS_EPS_SMALL: f64 = 1e-10;    
const REL_EPS_SMALL: f64 = 1e-6;
const REL_EPS_LARGE: f64 = 1e-3;
//}}}
//{{{ trait: RealFn 
pub trait RealFn<const N: usize>: Clone
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    //{{{ fun: eval
    fn eval(&mut self, x: &SVector<N>) -> f64;
    //}}}
    //{{{ fun: grad
    fn grad(&mut self, x_in: &SVector<N>) -> SVector<N> {

    //{{{ trace
    debug!("x_in = {}", x_in);
    //}}}
    let mut x = x_in.clone();
    let f_x = self.eval(x_in).abs();
    let mut grad = SVector::<N>::zeros();
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
            f_plus = self.eval(&x);
            x[i] -= 2.0 * h;
            f_minus = self.eval(&x);
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
    //}}}
}
//}}}
//{{{ struct: ZeroFn
#[derive(Debug, Clone)]
pub struct ZeroFn<const N: usize> {}   
//}}}
//{{{ impl: RealFn for ZeroFn
impl<const N: usize> RealFn<N> for ZeroFn<N> 
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn eval(&mut self, x: &SVector<N>) -> f64 {
        0.0
    }
    fn grad(&mut self, x_in: &SVector<N>) -> SVector<N> {
        SVector::<N>::zeros()
    }
}
//}}}
//{{{ struct: FnMutWrap
#[derive(Debug, Clone)]
pub struct FnMutWrap<const N: usize, F: FnMut(&SVector<N>) -> f64 + Clone> {
    f: F
}
//}}}
//{{{ impl: FnMutWrap
impl<const N: usize, F: FnMut(&SVector<N>) -> f64 + Clone> FnMutWrap<N, F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}
//}}}
//{{{ impl: RealFn for FnMutWrap
impl<const N: usize, F: FnMut(&SVector<N>) -> f64 + Clone> RealFn<N> for FnMutWrap<N, F> 
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn eval(&mut self, x: &SVector<N>) -> f64 {
        (self.f)(x)
    }
}
//}}}

trait Checkable {
    fn check(&self) -> bool;
}

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

                let mut f = FnMutWrap::new(|x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = (x - 1.0).powi(2) * (y - 2.0).powi(2);    
                    out
                });

                let grad_f_true = |x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = SVector::<2>::from_slice(&[
                        ((x - 1.0) * 2.0)*(y - 2.0).powi(2),
                        (x - 1.0).powi(2) * ((y - 2.0) * 2.0),
                    ]);
                    out
                };

                let x = SVector::<2>::from_slice(&$point);
                let grad_fx1 = f.grad(&x);
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

                let mut f = FnMutWrap::new(|x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = x.sin() * y.sin();
                    out
                });

                let grad_f_true = |x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = SVector::<2>::from_slice(&[
                        x.cos() * y.sin(),
                        x.sin() * y.cos()
                    ]);
                    out
                };

                let x = SVector::<2>::from_slice(&$point);
                let grad_fx1 = f.grad(&x);
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

                let mut f = FnMutWrap::new(|x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1]; 
                    let out = x.ln() * y.ln();
                    out
                });

                let grad_f_true = |x_vec: &SVector<2>| {
                    let x = x_vec[0];
                    let y = x_vec[1];
                    let out = SVector::<2>::from_slice(&[
                        (1.0 / x) * y.ln(),
                        x.ln() * (1.0 / y)
                    ]);
                    out
                };

                let x = SVector::<2>::from_slice(&$point);
                let grad_fx1 = f.grad(&x);
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