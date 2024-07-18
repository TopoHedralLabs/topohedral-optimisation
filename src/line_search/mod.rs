//! The line search module implements the set of line search algorithms for this crate.
//!
//! 
//! # Introduction
//! Line search algorithms approximately solve the problem:
//! 
//! $$
//! \min_{\alpha \in \mathbb{R}} f(\mathbf{x} + \alpha \mathbf{d})
//! $$
//! 
//! wwhere $\mathbf{x}$ is the current location, $\mathbf{d}$ is the search direction, and $f$ is 
//! the multivariate objective function being minimized.
//! The purpose of line search methods is to find a step that is "good enough" using fewer function 
//! and derivative evaluations than a full 1D minimization. The set of supported algorithms are:
//! 
//! - Fixed Step: Fixed step size, unconditionally accept step
//! - Backtracking: Simple backtracking using bisection
//! - Nocedal: Nocedal's line search method
//! - Thuente: Thuente's line search method
//! 
//! # Line Search Algorithms
//! 
//! ## Fixed Step
//! 
//! ## Backtracking
//! 
//! ## Nocedal
//! 
//! ## Thuente
//--------------------------------------------------------------------------------------------------

mod backtracking;
mod fixed_step;
mod nocedal;
mod thuente;

//{{{ crate imports
use crate::common::{EvaluateSMatrix, SMatrix, SVector};
use crate::common::{FnMutWrap, RealFn};

pub use crate::line_search::backtracking::{BacktrackingLineSearch, BacktrackingOpts};
pub use crate::line_search::fixed_step::{FixedStepLineSearch, FixedStepOpts};
pub use crate::line_search::nocedal::{NocedalLineSearch, NocedalOpts};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Error
#[derive(PartialEq, Error, Debug)]
pub enum Error {
    #[error("Not decreasing")]
    NotDecreasing,
    #[error("Fails Armijo condition")]
    Armijo,
    #[error("Fails curvature condition")]
    Curvature,
    #[error("Max iterations reached")]
    MaxIterations,
    #[error("Step size too small")]
    StepSizeSmall,
}
//}}}
//{{{ struct: LineSearchOpts
/// Options for configuring a line search algorithm.
///
/// This struct contains the parameters needed to configure a line search algorithm,
/// such as the initial function value (`phi0`), the initial derivative value (`dphi0`),
/// and the Armijo and curvature conditions (`c1` and `c2`). The `method` field
/// specifies the line search method to use, which can be one of `FixedStep`, `Quadratic`,
/// or `Inexact`.
#[derive(Debug, Clone)]
pub struct LineSearchOpts {
    pub c1: f64,
    pub c2: f64,
}
//}}}
//{{{ struct: LineSearchReturns
/// The results of a line search algorithm.
///
/// This struct contains the following fields:
/// - `alpha`: The step size found by the line search.
/// - `falpha`: The function value at the step size `alpha`.
/// - `funcalls`: The number of function evaluations performed.
/// - `gradcalls`: The number of gradient evaluations performed.
#[derive(Debug, Clone, Default)]
pub struct LineSearchReturns {
    pub alpha: f64,
    pub falpha: f64,
    pub funcalls: usize,
    pub gradcalls: usize,
}
//}}}
//{{{ struct: LineSearchFn
pub(crate) struct LineSearchFn<const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    f: F,
    pub x: SVector<N>,
    pub dir: SVector<N>,
}
//}}}
//{{{ enum: LineSearchMethod
#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    FixedStep(FixedStepOpts),
    Backtracking(BacktrackingOpts),
    Nocedal(NocedalOpts),
}
//}}}
//{{{ impl: LineSearchFn
impl<const N: usize, F: RealFn<N>> LineSearchFn<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn new(f: F, x: SVector<N>, dir: SVector<N>) -> Self {
        Self {
            f: f,
            x: x,
            dir: dir,
        }
    }

    fn eval(&mut self, alpha: f64) -> f64 {
        let x_alpha = (&self.x + alpha * &self.dir).evals();
        self.f.eval(&x_alpha)
    }

    fn eval_diff(&mut self, alpha: f64) -> f64 {
        let x_alpha = (&self.x + alpha * &self.dir).evals();
        let grad_f = self.f.grad(&x_alpha);
        grad_f.dot(&self.dir)
    }
}
//}}}
//{{{ fun: satisfies_armijo
fn satisfies_armijo(c1: f64, alpha: f64, phi0: f64, dphi0: f64, phi1: f64) -> bool {
    //{{{ trace
    trace!(target: "ls", "armijo: left = {:1.4e} right = {:1.4e}", phi1, phi0 + c1 * alpha * dphi0);
    trace!(target: "ls", "Satisfies Armijo {}", phi1 <= phi0 + c1 * alpha * dphi0);
    //}}}
    phi1 <= phi0 + c1 * alpha * dphi0
}
//}}}
//{{{ fun:  satisfies_curvature
fn satisfies_curvature(c2: f64, dphi0: f64, dphi1: f64) -> bool {
    //{{{ trace
    trace!(target: "ls", "curvature: left = {:1.4e} right = {:1.4e}", dphi1, c2 * dphi0);
    trace!(target: "ls", "Satisfies curvature {}", dphi1 >= c2 * dphi0);
    //}}}
    dphi1 >= c2 * dphi0
}
//}}}
//{{{ fun: satisfies_wolfe
fn satisfies_wolfe(
    c1: f64,
    c2: f64,
    phi0: f64,
    dphi0: f64,
    phi1: f64,
    dphi1: f64,
    alpha: f64,
) -> Result<(), Error> {
    //{{{ trace
    trace!(target: "ls", "phi0 = {:1.4e} dphi0 = {:1.4e} phi1 = {:1.4e} dphi1 = {:1.4e} alpha = {:1.4e}", phi0, dphi0, phi1, dphi1, alpha);
    //}}}
    if !satisfies_armijo(c1, alpha, phi0, dphi0, phi1){
        return Err(Error::Armijo);
    }
    if !satisfies_curvature(c2, dphi0, dphi1){
        return Err(Error::Curvature);
    }
    Ok(())
}
//}}}
//{{{ trait: LineSearch
pub trait LineSearch<const N: usize>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, Error>;
    fn set_location_and_direction(&mut self, x: SVector<N>, dir: SVector<N>);
    fn set_initial_step_size(&mut self, alpha: f64);
    fn get_initial_step_size(&self) -> f64;
}
//}}}
//{{{ fun: create
/// Factory function to create a line search algorithm given some options.
pub fn create<'a, const N: usize, F: RealFn<N> + 'a>(
    f: F,
    x: SVector<N>,
    dir: SVector<N>,
    opts: LineSearchMethod,
) -> Box<dyn LineSearch<N> + 'a>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    let f_line_search = LineSearchFn::new(f, x, dir);

    match opts {
        LineSearchMethod::FixedStep(opts) => Box::new(FixedStepLineSearch {
            f: f_line_search,
            opts,
        }),
        LineSearchMethod::Backtracking(opts) => Box::new(BacktrackingLineSearch {
            f: f_line_search,
            opts,
        }),
        LineSearchMethod::Nocedal(opts) => Box::new(NocedalLineSearch {
            f: f_line_search,
            opts,
        }),
    }
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let f = FnMutWrap::new(|x: &SVector<2>| x[0] * x[0] + x[1] * x[1]);
        let x = SVector::from_slice(&[1.0, 1.0]);
        let dir = SVector::from_slice(&[1.0, 1.0]);
        let opts = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts {
                c1: 0.1,
                c2: 0.9,
            },
            step_size: 0.1,
        });
        let mut line_search = create(f, x, dir, opts);
    }
    //}}}
}
