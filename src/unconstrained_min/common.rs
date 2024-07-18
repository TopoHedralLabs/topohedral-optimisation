//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::common::{SVector, SMatrix, RealFn};
use crate::line_search as ls;
use crate::unconstrained_min::conjugate_gradient as cg;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Error, Debug)] 
pub enum Error {

    #[error("Linear search failed with error {0}")]
    LineSearch(#[from] ls::Error),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
}

pub enum Method<const N: usize> 
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    CG(cg::Opts<N>)
}
//{{{ struct: UnconstrainedOpts
#[derive(Debug, Clone)]
pub struct Opts<const N: usize> 
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub abstol: f64,
    pub max_iter: usize,
    pub ls_method: ls::LineSearchMethod,
}
//}}}

#[derive(Debug, Clone)]
pub struct Returns<const N: usize>
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub xmin: SVector<N>,
    pub fmin: f64,
    pub iter: usize,
    pub funcalls: usize,
    pub gradcalls: usize,
    pub num_restarts: usize,
}

pub trait Minimizer<const N: usize, F: RealFn<N>>
    where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn minimize(&mut self, f: F, x0: SVector<N>) -> Result<Returns<N>, Error>;
}