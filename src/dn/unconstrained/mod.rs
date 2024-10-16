//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod common;
mod cg;
mod bgfs;
mod newton;

use common::*;
use crate::common::Vector; 


//{{{ fun: minimize
pub fn minimize<F: Fn(&Vector) -> f64>(f: F, x0: &Vector, opts: &MinimizeOptions) -> MinimizeReturns {

    match opts.method 
    {
        Method::CG => cg::minimize(f, x0, opts),
        Method::BFGS => bgfs::minimize(f, x0, opts),
        Method::NEWTON => newton::minimize(f, x0, opts),
        _ => panic!("Unrecognized options")
    }
}
//..............................................................................
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {}
//}}}
