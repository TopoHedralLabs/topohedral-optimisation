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


//{{{ fun: minimize
pub fn min_uncon<F: Fn(&[f64]) -> f64>(f: F, opts: &MinUnconOptions) -> MinUnconReturns {

    match opts.method 
    {
        UnconMethod::CG => cg::minimize(f, opts),
        UnconMethod::BFGS => bgfs::minimize(f, opts),
        UnconMethod::NEWTON => newton::minimize(f, opts),
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
