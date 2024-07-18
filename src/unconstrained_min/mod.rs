//! Unconstrained minimization module
//!
//! This module contains all of the unconstrained minimization algorithms.
//--------------------------------------------------------------------------------------------------


//{{{ crate imports 
use crate::common::RealFn;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//{{{ exports
// exports from common
mod common;
pub use common::{Error,Method, Minimizer, Returns, Opts};
// export conjugate gradient    
pub mod conjugate_gradient;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ fun: create 
/// Create a minimizer from a method
/// 
/// This function is the entry point for the unconstrained minimization module.
/// 
/// # Arguments
/// - method: The method to use for the minimization. This enum contains a nested set of options 
///           structs which together specify all aspacts of the minimization.
/// # Returns 
/// - A boxed trait object that implements the `Minimizer` trait.
pub fn create<const N: usize, F: RealFn<N> + 'static>(method: Method<N>)
-> Box<dyn Minimizer<N, F> >
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    match method {
        Method::CG(opts) => Box::new(conjugate_gradient::ConjugateGradient::new(opts))
    }
}
//}}}

