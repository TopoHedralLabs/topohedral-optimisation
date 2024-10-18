//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use nalgebra as na;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

pub type Vector = na::DVector<f64>;
pub type Matrix = na::DMatrix<f64>;

const EPS: f64 = 1e-5;
const EPS_SMALL: f64 = 1e-6;
const EPS_LARGE: f64 = 1e-2;

pub fn grad<F: Fn(&Vector) -> f64>(f: F, x_in: &Vector) -> Vector {
    //{{{ trace
    debug!("x_in = {}", x_in);
    //}}}
    let mut x = x_in.clone();
    let f_x = f(&x).abs();
    let mut grad = Vector::zeros(x.len());
    let mut f_plus = 0.0f64;
    let mut f_minus = 0.0f64;
    let jmax = 100;

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
            trace!("......................... j = {}", j);
            let xi = x[i];
            x[i] += h;
            f_plus = f(&x);
            x[i] -= 2.0 * h;
            f_minus = f(&x);
            x[i] = xi;
            let delta_f = (f_plus - f_minus).abs();
            let rel_delta_f = delta_f / f_x;
            //{{{ trace
            trace!("f_plus = {} f_minus = {}", f_plus, f_minus);
            trace!("delta_f = {} rel_delta_f = {}", delta_f, rel_delta_f);
            //}}}

            if rel_delta_f < EPS_SMALL {
                //{{{ trace
                trace!("rel_delta_f is small, enlarging h = {}", h);
                //}}}
                h *= 2.0;
            } else if rel_delta_f > EPS_LARGE {
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
    grad
}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use super::*;

    const MAX_REL: f64 = 1e-8;


    #[test]
    fn test_sin() {
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

        let x = Vector::from_column_slice(&[1.0f64,1.0f64]);


        let test_points: Vec<[f64; 2]> = vec![
            [0.0, 0.0] 
            // [1.0e5, 0.0],
            // [-1.0e5, 0.0],
            // [-1.0e5, 0.0],
        ];

        for point in test_points
        {
            let x = Vector::from_column_slice(&point);
            let grad_fx1 = grad(f, &x);
            let grad_fx2 = grad_f_true(&x);
            for i in 0..grad_fx1.len() {
                assert_relative_eq!(grad_fx1[i], grad_fx2[i], epsilon = MAX_REL);
            }
        }
    }
}
//}}}
