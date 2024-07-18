//! This module contains the conjugate gradient method.
//!
//! The conjugate gradient method is a first-order optimization algorithm that picks the search
//! dirction based only on the gradient and information from the previous iteration.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{RealFn, SVector},
    line_search as ls,
    line_search::{LineSearch, LineSearchReturns},
};

use super::common as com;
use crate::line_search::create;

//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::EvaluateSMatrix;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: ConvergedReason
pub enum ConvergedReason {
    Rtol,
    Atol,
}
//}}}
//{{{ enum: DirectionMethod
/// Specifies the method used to update the search direction in the conjugate gradient algorithm.
/// - `FletcherReeves`: Uses the Fletcher-Reeves formula to update the search direction.
/// - `PolakRibiere`: Uses the Polak-Ribiere formula to update the search direction.
#[derive(Debug, Clone)]
pub enum DirectionMethod {
    FletcherReeves,
    PolakRibiere,
}
//}}}
//{{{ impl: Default for DirectionMethod
impl Default for DirectionMethod {
    fn default() -> Self {
        DirectionMethod::FletcherReeves
    }
}
//}}}
//{{{ struct: Opts
/// Holds the options for the conjugate gradient optimization method.
/// The `Opts` struct contains the unconstrained optimization options and the
/// direction update method to use for the conjugate gradient algorithm.
#[derive(Debug, Clone)]
pub struct Opts<const N: usize>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub unconstrained_opts: com::Opts<N>,
    pub direction_method: DirectionMethod,
}
//}}}
//{{{ struct: ConjugateGradient
/// A struct that represents the conjugate gradient optimization method.
/// The `ConjugateGradient` struct holds the optimization options and is used to
/// perform unconstrained optimization using the conjugate gradient algorithm.
pub struct ConjugateGradient<const N: usize>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    opts: Opts<N>,
    line_searcher: Option<Box<dyn LineSearch<N>>>,
}
//}}}
//{{{ impl: ConjugateGradient
impl<const N: usize> ConjugateGradient<N>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub fn new(opts: Opts<N>) -> Self {
        Self {
            opts: opts.clone(),
            line_searcher: None,
        }
    }

    fn line_search_with_restart(
        &mut self,
        xk: &SVector<N>,
        fk: f64,
        grad_fk: &SVector<N>,
        dir_k1: &mut SVector<N>,
        dir_k: &mut SVector<N>,
    ) -> Result<(LineSearchReturns, bool), ls::Error> {
        //{{{ trace
        error!("--- Entering line_search_with_restart ---");
        //}}}

        let phi0 = fk;
        let mut dphi0 = grad_fk.dot(dir_k);
        self.line_searcher
            .as_mut()
            .unwrap()
            .set_location_and_direction(*xk, *dir_k);
        let ls_ret = self
            .line_searcher
            .as_mut()
            .unwrap()
            .line_search(phi0, dphi0);

        if let Err(e) = &ls_ret {
            //{{{ trace
            error!(target: "cg", "Failed to perform line search: {}", e);
            //}}}
            *dir_k1 = -*grad_fk;
            *dir_k = -*grad_fk;
            dphi0 = grad_fk.dot(&dir_k);

            self.line_searcher
                .as_mut()
                .unwrap()
                .set_location_and_direction(*xk, *dir_k);
            let ls_ret_try_2 = self
                .line_searcher
                .as_mut()
                .unwrap()
                .line_search(phi0, dphi0);

            if let Err(e2) = ls_ret_try_2 {
                //{{{ trace
                error!(target: "cg", "Failed to perform line search: {}", e2);
                error!("--- Leaving line_search_with_restart ---");
                //}}}
                return Err(e2);
            }

            //{{{ trace
            error!("--- Leaving line_search_with_restart ---");
            //}}}
            return Ok((ls_ret_try_2.unwrap(), true));
        }
        //{{{ trace
        error!("--- Leaving line_search_with_restart ---");
        //}}}
        Ok((ls_ret.unwrap(), false))
    }

    /// Updates the search direction for the conjugate gradient method based on the
    /// current and previous gradients, and the current search direction.
    /// The update formula used depends on the `DirectionMethod` specified in the
    /// `Opts` struct.
    fn update_direction(
        &self,
        grad_fk1: &SVector<N>,
        grad_fk: &SVector<N>,
        norm_grad_fk1: f64,
        norm_grad_fk: f64,
        dir_k: &SVector<N>,
    ) -> SVector<N> {
        //{{{ trace
        error!(target: "cg", "--- Entering update_direction ---");
        trace!(target: "cg", "grad_fk1 = {}, grad_fk = {}, norm_grad_fk1 = {:1.4e} norm_grad_fk = {:1.4e}", 
                 grad_fk1, grad_fk, norm_grad_fk1, norm_grad_fk);
        //}}}
        // direction updates
        let beta = match self.opts.direction_method {
            DirectionMethod::FletcherReeves => {
                //{{{ trace
                trace!("Applying fletcher-reeves update");
                //}}}
                let beta_tmp = grad_fk.dot(&grad_fk1) / norm_grad_fk1.powi(2);
                beta_tmp
            }
            DirectionMethod::PolakRibiere => {
                //{{{ trace
                trace!("Applying polak-ribiere update");
                //}}}
                let yk: SVector<N> = (grad_fk - grad_fk1).evals();
                let mut beta_tmp = grad_fk.dot(&yk) / norm_grad_fk1.powi(2);
                beta_tmp = beta_tmp.max(0.0);
                beta_tmp
            }
        };

        let new_dir_k = ((beta * dir_k) - grad_fk).evals();
        //{{{ trace
        trace!(target: "cg", "beta = {:1.4e}", beta);
        error!(target: "cg", "--- Leaving update_direction ---");
        //}}}
        new_dir_k
    }
}
//}}}
//{{{ impl: Minimizer for ConjugateGradient
impl<const N: usize, F: RealFn<N> + 'static> com::Minimizer<N, F> for ConjugateGradient<N>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn minimize(&mut self, mut f: F, x0: SVector<N>) -> Result<com::Returns<N>, super::Error> {
        // initialize position, function value, gradient, and direction
        let mut xk = x0;
        let mut fk = f.eval(&xk);
        let mut grad_fk = f.grad(&xk);
        let mut grad_fk1 = grad_fk;
        let mut norm_grad_fk = grad_fk.norm();
        let mut dir_k = -grad_fk;
        let mut dir_k1 = -grad_fk;
        // initialize line searcher
        let ls_method = self.opts.unconstrained_opts.ls_method.clone();
        self.line_searcher = Some(create(f.clone(), x0, dir_k, ls_method));
        // initialize iteration, function and grad counters
        let mut outer_it = 0;
        let mut outer_funcalls = 1;
        let mut outer_gradcalls = 1;
        let mut num_restarts = 0;
        // initialize stopping criteria
        let max_it = self.opts.unconstrained_opts.max_iter;
        let abstol = self.opts.unconstrained_opts.abstol;
        // main loop

        while norm_grad_fk > abstol {
            //{{{ trace
            trace!(target: "cg", "\n\n=========================================== cg it = {}", outer_it);
            trace!(target: "cg", "fk = {:1.4e} norm_grad_fk = {:1.4e}", fk, norm_grad_fk);
            trace!(target: "cg", "xk = \n{}", xk);
            trace!(target: "cg", "grad_fk = \n{}", grad_fk);
            trace!(target: "cg", "dir_k = \n{}", dir_k);
            trace!(target: "cg", "norm_grad_fk = {:1.4e}", norm_grad_fk);
            //}}}

            let ls_ret = self.line_search_with_restart(&xk, fk, &grad_fk, &mut dir_k1, &mut dir_k);
            if let Err(e) = ls_ret {
                //{{{ trace
                error!(target: "cg", "Failed to perform line search: {}", e);
                error!("--- Leaving minimize ---");
                //}}}
                return Err(super::Error::LineSearch(e));
            }

            let (
                LineSearchReturns {
                    alpha,
                    falpha,
                    funcalls,
                    gradcalls,
                },
                restarted,
            ) = ls_ret.unwrap();

            if restarted {
                num_restarts += 1;
            }
            //{{{ trace
            trace!(target: "gc", "Results of line search {:1.4e} {:1.4e} {} {} {}", alpha, falpha, funcalls, gradcalls, restarted);
            //}}}
            // function call updates
            outer_funcalls += funcalls;
            outer_gradcalls += gradcalls;
            // location and function value updates
            xk = (&xk + alpha * &dir_k).evals();
            fk = falpha;
            // gradient updates
            grad_fk1 = grad_fk;
            grad_fk = f.grad(&xk);
            let norm_grad_fk1 = norm_grad_fk;
            norm_grad_fk = grad_fk.norm();
            // direction update
            dir_k1 = dir_k;
            dir_k = self.update_direction(&grad_fk1, &grad_fk, norm_grad_fk1, norm_grad_fk, &dir_k);
            // iter update
            outer_it += 1;

            if outer_it == max_it {
                //{{{ trace
                trace!(target: "cg", "Failed to converge, hit max iterations");
                //}}}
                return Err(com::Error::MaxIterations(max_it));
            }
        }

        //{{{ trace
        trace!(target: "cg", "Successfully converged to fk = {} and xk = \n\n {}", fk, xk);
        //}}}
        Ok(com::Returns {
            xmin: xk,
            fmin: fk,
            iter: outer_it,
            funcalls: outer_funcalls,
            gradcalls: outer_gradcalls,
            num_restarts: num_restarts,
        })
    }
}
//}}}
