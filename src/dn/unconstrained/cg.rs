//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------


//{{{ crate imports 
use super::common::*;
use crate::common::{Vector, Matrix};
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------




//{{{ fun: min_uncon_cg
pub fn minimize<F: Fn(&Vector) -> f64>(f: F, x0: &Vector, opts: &MinimizeOptions) -> MinimizeReturns 
{

    let xk = x0.clone();
    let xk1 = xk.clone();

    // let 

    panic!()    
}
//}}}




//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_case_1()
    {

        let f = |x: &Vector| {
            x[0].powi(2) + x[1].powi(2)
        };

        let x0 = Vector::from_column_slice(&[1.0, 1.0]);

        let opts = MinimizeOptions {
            method: Method::CG,
            tol: 1e-6,
            max_iter: 1000,
            grad_f: None,  
            hess_f: None,
        };

        // let res = minimize(f, &x0, &opts);
    }
  
}
//}}}