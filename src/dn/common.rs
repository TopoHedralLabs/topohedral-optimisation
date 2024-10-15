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


//{{{ enum:   UnconMethod
pub enum UnconMethod
{
   CG, 
   BFGS, 
   NEWTON
}
//}}}
//{{{ struct: MinUnconOptions
pub struct MinUnconOptions {
    pub method: UnconMethod,
    pub tol: f64,
    pub max_iter: usize,
}
//}}}
//{{{ struct: MinUnconReturns
pub struct MinUnconReturns {}
//}}}








//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
  
}
//}}}