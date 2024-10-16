//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::common::{Vector, Matrix};
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------


//{{{ enum:   UnconMethod
pub enum Method
{
   CG, 
   BFGS, 
   NEWTON
}
//}}}
//{{{ struct: IterHistory
pub struct IterHistory {
    pub x: Vec<Vector>,
    pub f: Vec<f64>,
    pub grad_f: Vec<Vector>,
}
//}}}
//{{{ struct: MinimizeOptions 
pub struct MinimizeOptions {
    pub method: Method,
    pub tol: f64,
    pub max_iter: usize,
    pub grad_f: Option<Box<dyn Fn(&Vector) -> Vector>>,
    pub hess_f: Option<Box<dyn Fn(&Vector) -> Matrix>>
}
//}}}
//{{{ struct: MinimizeReturns 
pub struct MinimizeReturns {
    pub xmin: Vector,
    pub fmin: f64,
    pub iter_history: Option<IterHistory>,
}
//}}}








//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
  
}
//}}}