
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
//}}}
//--------------------------------------------------------------------------------------------------

pub type Vector = na::DVector<f64>;
pub type Matrix = na::DMatrix<f64>;

const EPS: f64 = 1e-5;
const tol_rel: f64 = 1e-8;


pub fn grad<F: Fn(&Vector) -> f64>(f: F, x: Vector) -> Vector
{
    let f_x = f(&x);

    let mut grad = Vector::zeros(x.len());
    let mut f_minus = 0.0f64;
    let mut f_plus = 0.0f64;

    for i in 0..x.len()
    {
        let mut h = EPS * (1.0 + x[i].abs());
        let delta_f = (f_plus - f_minus).abs();




        // while  
        // x[i] += dx;
        // let f_plus = f(&x);
        // x[i] -= 2.0 * dx;
        // let f_minus = f(&x); 



        // grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * dx);
    }
    grad
}









//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
  
}
//}}}