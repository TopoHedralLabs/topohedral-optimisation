//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------


//{{{ crate imports 
use super::common::*;
use crate::common::Vector;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------




//{{{ fun: min_uncon_cg
pub fn minimize<F: Fn(&Vector) -> f64>(f: F, x0: &Vector, opts: &MinimizeOptions) -> MinimizeReturns 
{
    panic!()    
}
//}}}




//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
  
}
//}}}