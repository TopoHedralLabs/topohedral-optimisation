//! This module implements the More-Thuente line search algorithm.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::{LineSearch, LineSearchFn, LineSearchOpts, LineSearchReturns};
use crate::common::{GreaterThan, RealFn, SMatrix, SVector};
use serde_json::map::Iter;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ collection: Constants for the algorithm
const P5: f64 = 0.5f64;
const P66: f64 = 0.66;
const XTRAPL: f64 = 1.1;
const XTRAPU: f64 = 4.0;
//}}}
//{{{ struct: ThuenteOpts
#[derive(Debug, Clone)]
pub struct ThuenteOpts {
    pub ls_opts: LineSearchOpts,
    pub initial_step_size: f64,
    pub min_step_size: f64,
    pub max_step_size: f64,
    pub max_iter: usize,
    alpha_tol: f64,
}
//}}}
//{{{ struct: ThuenteLineSearch
pub struct ThuenteLineSearch<const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
    (): GreaterThan<N, 1>,
{
    pub(crate) f: LineSearchFn<N, F>,
    pub opts: ThuenteOpts,

    stage: usize,
    phi_init: f64,
    dphi_init: f64,
    dphi_test: f64,

    alpha_a: f64,
    phi_a: f64,
    dphi_a: f64,

    alpha_b: f64,
    phi_b: f64,
    dphi_b: f64,

    phi: f64,
    dphi: f64,

    brackt: bool,
    width: f64,
    width1: f64,
    stmin: f64,
    stmax: f64,
}
//}}}
//{{{ impl: LineSearch for ThuenteLineSearch
impl<const N: usize, F: RealFn<N>> LineSearch<N> for ThuenteLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
    (): GreaterThan<N, 1>,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, super::Error> {
        //{{{ trace: enter
        error!(target: "ls", "Entering line search with phi0 = {phi0}, dphi0 = {dphi0}");
        //}}}
        self.initialise(0.0, phi0, dphi0);
        let (mut alpha, mut phi_alpha, mut dphi_alpha) = (0.0, phi0, dphi0);
        let (mut funcalls, mut gradcalls) = (0, 0);

        for i in 0..self.opts.max_iter {
            //{{{ trace
            info!(target: "ls", "-------------- On iteration {i}");
            info!(target: "ls", "alpha = {alpha}, phi_alpha = {phi_alpha}, dphi_alpha = {dphi_alpha}");
            //}}}
            match self.iterate(IterateData {
                alpha,
                phi_alpha,
                dphi_alpha,
                funcalls: 0,
                gradcalls: 0,
            }) {
                IterateReturn::Converged(data) => {
                    //{{{ trace
                    info!(target: "ls", 
                        "Converged with alpha = {}, phi_alpha = {}, dphi_alpha = {}",
                        data.alpha, data.phi_alpha, data.dphi_alpha
                    );
                    //}}}
                    funcalls += data.funcalls;
                    gradcalls += data.gradcalls;
                    return Ok(LineSearchReturns {
                        alpha: data.alpha,
                        falpha: data.phi_alpha,
                        funcalls,
                        gradcalls,
                    });
                }
                IterateReturn::Unconverged(data) => {
                    //{{{ trace
                    info!(target: "ls", "Unconverged with alpha = {alpha}, phi_alpha = {phi_alpha}, dphi_alpha = {dphi_alpha}");
                    //}}}
                    alpha = data.alpha;
                    phi_alpha = data.phi_alpha;
                    dphi_alpha = data.dphi_alpha;
                    funcalls += data.funcalls;
                    gradcalls += data.gradcalls;
                }
                IterateReturn::Error(_data) => {}
            }
        }

        //{{{ trace: leave
        debug!(target: "ls", "Exiting line search)");
        //}}}
        Ok(LineSearchReturns {
            alpha: 0.0,
            falpha: 0.0,
            funcalls: 0,
            gradcalls: 0,
        })
    }

    fn set_location_and_direction(&mut self, x: SVector<N>, dir: SVector<N>) {
        self.f.x = x;
        self.f.dir = dir;
    }

    fn set_initial_step_size(&mut self, alpha: f64) {
        self.opts.initial_step_size = alpha;
    }

    fn get_initial_step_size(&self) -> f64 {
        self.opts.initial_step_size
    }
}
//}}}
//{{{ struct: IterateData
struct IterateData {
    alpha: f64,
    phi_alpha: f64,
    dphi_alpha: f64,
    funcalls: usize,
    gradcalls: usize,
}
//}}}
//{{{ struct: IterateReturn
enum IterateReturn {
    Converged(IterateData),
    Unconverged(IterateData),
    Error(IterateData),
}
//}}}
//{{{ impl: ThuenteLineSearch
impl<const N: usize, F: RealFn<N>> ThuenteLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
    (): GreaterThan<N, 1>,
{
    pub fn new(f: F, x: SVector<N>, dir: SVector<N>, opts: ThuenteOpts) -> Self {
        Self {
            f: LineSearchFn { f, x, dir },
            opts,
            stage: 0,
            phi_init: 0.0,
            dphi_init: 0.0,
            dphi_test: 0.0,
            alpha_a: 0.0,
            phi_a: 0.0,
            dphi_a: 0.0,
            alpha_b: 0.0,
            phi_b: 0.0,
            dphi_b: 0.0,
            phi: 0.0,
            dphi: 0.0,
            brackt: false,
            width: 0.0,
            width1: 0.0,
            stmin: 0.0,
            stmax: 0.0,
        }
    }

    fn initialise(&mut self, alpha: f64, phi_alpha: f64, dphi_alpha: f64) {
        self.brackt = false;
        self.stage = 1;
        self.phi_init = phi_alpha;
        self.dphi_init = dphi_alpha;
        self.dphi_test = self.opts.ls_opts.c2 * self.dphi_init;
        self.width = self.opts.max_step_size - self.opts.min_step_size;
        self.width1 = self.width * P5;
        self.alpha_a = 0.0;
        self.phi_a = phi_alpha;
        self.dphi_a = dphi_alpha;
        self.alpha_b = 0.0;
        self.phi_b = phi_alpha;
        self.dphi_b = dphi_alpha;
        self.stmin = 0.0;
        self.stmax = (1.0 + XTRAPU) * alpha
    }

    fn iterate(&mut self, data: IterateData) -> IterateReturn {
        let IterateData {
            mut alpha,
            mut phi_alpha,
            mut dphi_alpha,
            mut funcalls,
            mut gradcalls,
        } = data;

        let phi_test = self.phi_init + alpha * self.dphi_test;
        if phi_alpha < phi_test && dphi_alpha.abs() <= self.opts.ls_opts.c1 * self.dphi_init.abs() {
            //{{{ trace
            warn!(target: "ls", "Successfully converged");
            //}}}
            return IterateReturn::Converged(IterateData {
                alpha,
                phi_alpha,
                dphi_alpha,
                funcalls,
                gradcalls,
            });
        }

        if self.stage == 1 && phi_alpha <= phi_test && dphi_alpha >= 0.0 {
            self.stage = 2;
        }

        if self.stage == 1 && phi_alpha < self.phi_a && phi_alpha > phi_test 
        //{{{ case: stage 1
        {
            // A modified function is used to predict the step during the
            // first stage if a lower function value has been obtained but
            // the decrease is not sufficient.
            let phi_m = phi_alpha - alpha * self.dphi_test;
            let phi_am = self.alpha_a - self.alpha_a * self.dphi_test;
            let phi_bm = self.phi_b - self.alpha_b * self.dphi_test;

            let dphi_m = dphi_alpha - self.dphi_test;
            let dphi_am = self.dphi_a - self.dphi_test;
            let dphi_bm = self.dphi_b - self.dphi_test;

            let dcstep_res = dcstep(DcstepArgs {
                alpha_a: self.alpha_a,
                phi_a: phi_am,
                dphi_a: dphi_am,
                alpha_b: self.alpha_b,
                phi_b: phi_bm,
                dphi_b: dphi_bm,
                alpha: alpha,
                phi_alpha: phi_m,
                dphi_alpha: dphi_m,
                brackt: self.brackt,
                stpmin: self.opts.min_step_size,
                stpmax: self.opts.max_step_size,
            });

            self.alpha_a = dcstep_res.alpha_a;
            self.alpha_b = dcstep_res.alpha_b;
            self.phi_a = dcstep_res.phi_a + self.alpha_a * self.dphi_test;
            self.phi_b = dcstep_res.phi_b + self.alpha_b * self.dphi_test;
            self.dphi_a = dcstep_res.dphi_a + self.dphi_test;
            self.dphi_b = dcstep_res.dphi_b + self.dphi_test;
            alpha = dcstep_res.alpha;
            self.brackt = dcstep_res.brackt;
        } 
        //}}}
        else 
        //{{{ case: stage 2
        {
            // Call dcstep to update stx, sty, and to compute the new step.
            // dcstep can have several operations which can produce NaN
            // e.g. inf/inf. Filter these out.
            let dcstep_res = dcstep(DcstepArgs {
                alpha_a: self.alpha_a,
                phi_a: self.phi_a,
                dphi_a: self.dphi_a,
                alpha_b: self.alpha_b,
                phi_b: self.phi_b,
                dphi_b: self.dphi_b,
                alpha: alpha,
                phi_alpha,
                dphi_alpha,
                brackt: self.brackt,
                stpmin: self.opts.min_step_size,
                stpmax: self.opts.max_step_size,
            });
            alpha = dcstep_res.alpha;
            self.alpha_a = dcstep_res.alpha_a;
            self.alpha_b = dcstep_res.alpha_b;
            self.phi_a = dcstep_res.phi_a;
            self.phi_b = dcstep_res.phi_b;
            self.dphi_a = dcstep_res.dphi_a;
            self.dphi_b = dcstep_res.dphi_b;
            self.brackt = dcstep_res.brackt;

        }
        //}}}
        
        if self.brackt {
            // decide if a bisection is needed
            if (self.alpha_b - self.alpha_a).abs() >= P66 * self.width1 {
                alpha = self.alpha_a + P5 * (self.alpha_b - self.alpha_a);
            }
            // save previous width and assign new width
            self.width1 = self.width;
            self.width = (self.alpha_b - self.alpha_a);
        }

        // set min and max allowable step sizes
        if self.brackt {
            self.stmin = f64::min(self.alpha_a, self.alpha_b);
            self.stmax = f64::max(self.alpha_a, self.alpha_b);
        } else {
            self.stmin = alpha + XTRAPL * (alpha - self.alpha_a);
            self.stmax = alpha + XTRAPU * (alpha - self.alpha_a);
        }

        // force a step to be within the bounds
        alpha = alpha.clamp(self.stmin, self.stmax);

        // If further progress is not possible, return the best point obtained during the search
        if (self.brackt && (alpha <= self.stmin || alpha >= self.stmax))
            || (self.brackt && (self.stmax - self.stmin <= self.opts.alpha_tol * self.width))
        {
            alpha = self.alpha_a;
        }

        phi_alpha = self.f.eval(alpha);
        funcalls += 1;
        dphi_alpha = self.f.eval_diff(alpha);
        gradcalls += 1;

        IterateReturn::Unconverged(IterateData {
            alpha,
            phi_alpha,
            dphi_alpha,
            funcalls, 
            gradcalls
        }) 
    }
}
//}}}
//{{{ struct: DcstepArgs
struct DcstepArgs {
    // values at lower bound of bracket in order step, function, derivative
    alpha_a: f64,
    phi_a: f64,
    dphi_a: f64,
    // values at upper bound of bracket in order step, function, derivative
    alpha_b: f64,
    phi_b: f64,
    dphi_b: f64,
    // current values in order step, function, derivative
    alpha: f64,
    phi_alpha: f64,
    dphi_alpha: f64,
    // bracketing flag, true if a minimizer has been bracketed, false otherwise
    brackt: bool,
    // max and min allowable step sizes
    stpmin: f64,
    stpmax: f64,
}
//}}}
//{{{ struct: DcstepReturns
struct DcstepReturns {
    // values at lower bound of bracket in order step, function, derivative
    alpha_a: f64,
    phi_a: f64,
    dphi_a: f64,
    // values at upper bound of bracket in order step, function, derivative
    alpha_b: f64,
    phi_b: f64,
    dphi_b: f64,
    // current step after dcstep
    alpha: f64,
    brackt: bool,
}
//}}}
//{{{ fun: dcstep
fn dcstep(inputs: DcstepArgs) -> DcstepReturns {
    let DcstepArgs {
        mut alpha_a,
        mut phi_a,
        mut dphi_a,
        mut alpha_b,
        mut phi_b,
        mut dphi_b,
        mut alpha,
        phi_alpha,
        dphi_alpha,
        mut brackt,
        stpmin,
        stpmax,
    } = inputs;
    let sgnd = dphi_alpha.signum() * dphi_a.signum();
    let mut alpha_candidate;

    if phi_alpha > phi_a
    //{{{ case: 1
    {
        // First case: A higher function value. The minimum is bracketed.
        // If the cubic step is closer to stx than the quadratic step, the
        // cubic step is taken, otherwise the average of the cubic and
        // quadratic steps is taken.
        let theta = 3.0 * (phi_a - phi_alpha) / (alpha - alpha_a) + dphi_a + dphi_alpha;
        let s = theta.abs().max(dphi_a.abs().max(dphi_alpha.abs()));
        let mut gamma = s * ((theta / s).powi(2) - (dphi_a / s) * (dphi_alpha / s)).sqrt();
        if alpha < alpha_a {
            gamma = -gamma;
        }
        let p = (gamma - dphi_a) + theta;
        let q = (gamma - dphi_a) + gamma + dphi_alpha;
        let r = p / q;
        // quadratic step
        let alpha_c = alpha_a + r * (alpha - alpha_a);
        // cubic step
        let alpha_q = alpha_a
            + ((dphi_a / ((phi_a - phi_alpha) / (alpha - alpha_a) + dphi_alpha)) / 2.0)
                * (alpha - alpha_a);
        // candidate for the next step
        alpha_candidate = if (alpha_c - alpha_a).abs() < (alpha_q - alpha_a).abs() {
            alpha_c
        } else {
            alpha_q
        };
        brackt = true;
    }
    //}}}
    else if sgnd < 0.0
    //{{{ case: 2
    {
        // Second case: A lower function value and derivatives of opposite
        // sign. The minimum is bracketed. If the cubic step is farther from
        // stp than the secant step, the cubic step is taken, otherwise the
        // secant step is taken.
        let theta = 3.0 * (phi_a - phi_alpha) / (alpha - alpha_a) + dphi_a + dphi_alpha;
        let s = theta.abs().max(dphi_a.abs().max(dphi_alpha.abs()));
        let mut gamma = s * ((theta / s).powi(2) - (dphi_a / s) * (dphi_alpha / s)).sqrt();
        if alpha > alpha_a {
            gamma = -gamma;
        }
        let p = (gamma - dphi_a) + theta;
        let q = (gamma - dphi_a) + gamma + dphi_alpha;
        let r = p / q;
        let alpha_c = alpha_a + r * (alpha - alpha_a);
        let alpha_q = alpha + (dphi_alpha / (dphi_alpha - dphi_a)) * (alpha_a - alpha);
        alpha_candidate = if (alpha_c - alpha).abs() > (alpha_q - alpha).abs() {
            alpha_c
        } else {
            alpha_q
        };
        brackt = true;
    }
    //}}}
    else if dphi_alpha.abs() < dphi_a.abs()
    //{{{ case: 3
    {
        // Third case: A lower function value, derivatives of the same sign,
        // and the magnitude of the derivative decreases.

        // The cubic step is computed only if the cubic tends to infinity
        // in the direction of the step or if the minimum of the cubic
        // is beyond stp. Otherwise the cubic step is defined to be the
        // secant step.
        let theta = 3.0 * (phi_a - phi_alpha) / (alpha - alpha_a) + dphi_a + dphi_alpha;
        let s = theta.abs().max(dphi_a.abs().max(dphi_alpha.abs()));

        // The case gamma = 0 only arises if the cubic does not tend
        // to infinity in the direction of the step.
        let mut gamma =
            s * (f64::max(0.0, (theta / s).powi(2) - (dphi_a / s) * (dphi_alpha / s))).sqrt();
        if alpha > alpha_a {
            gamma = -gamma;
        }
        let p = (gamma - dphi_alpha) + theta;
        let q = (gamma + (dphi_alpha - dphi_a)) + gamma;
        let r = p / q;
        let alpha_c = if r < 0.0 && gamma != 0.0 {
            alpha_a + r * (alpha_a - alpha)
        } else if alpha > alpha_a {
            stpmax
        } else {
            stpmin
        };

        let alpha_q = alpha_a + (dphi_a / (dphi_a - dphi_alpha)) * (alpha_a - alpha);

        if brackt {
            // A minimizer has been bracketed. If the cubic step is
            // closer to stp than the secant step, the cubic step is
            // taken, otherwise the secant step is taken.
            alpha_candidate = if (alpha_c - alpha).abs() < (alpha_q - alpha).abs() {
                alpha_c
            } else {
                alpha_q
            };

            alpha_candidate = if alpha > alpha_a {
                f64::min(alpha + 0.66 * (alpha_b - alpha), alpha_candidate)
            } else {
                f64::max(alpha + 0.66 * (alpha_a - alpha), alpha_candidate)
            };
        } else {
            // A minimizer has not been bracketed. If the cubic step is
            // farther from stp than the secant step, the cubic step is
            // taken, otherwise the secant step is taken.
            alpha_candidate = if (alpha_c - alpha).abs() > (alpha_q - alpha).abs() {
                alpha_c
            } else {
                alpha_q
            };
            alpha_candidate = alpha_candidate.clamp(stpmin, stpmax);
        }
    }
    //}}}
    else
    //{{{ case 4:
    {
        // Fourth case: A lower function value, derivatives of the same sign,
        // and the magnitude of the derivative does not decrease. If the
        // minimum is not bracketed, the step is either stpmin or stpmax,
        // otherwise the cubic step is taken.

        if brackt {
            let theta = 3.0 * (phi_alpha - phi_b) / (alpha_b - alpha) + dphi_b + dphi_alpha;
            let s = theta.abs().max(dphi_a.abs().max(dphi_alpha.abs()));
            let mut gamma = s * ((theta / s).powi(2) - (dphi_b / s) * (dphi_alpha / s)).sqrt();
            if alpha > alpha_b {
                gamma = -gamma;
            }
            let p = (gamma - dphi_alpha) + theta;
            let q = ((gamma - dphi_alpha) + gamma) + dphi_b;
            let r = p / q;
            // take the cubic step
            alpha_candidate = alpha + r * (alpha_b - alpha);
        } else if alpha > alpha_a {
            alpha_candidate = stpmax;
        } else {
            alpha_candidate = stpmin;
        }
    }
    //}}}

    // update the interval wihch contains the minimizer
    if phi_alpha > phi_a {
        alpha_b = alpha;
        phi_b = phi_alpha;
        dphi_b = dphi_alpha;
    } else {
        if sgnd < 0.0 {
            alpha_b = alpha_a;
            phi_b = phi_a;
            dphi_b = dphi_a;
        }
        phi_a = phi_alpha;
        dphi_a = dphi_alpha;
        alpha_a = alpha;
    }
    // compute new step
    alpha = alpha_candidate;

    DcstepReturns {
        alpha_a,
        phi_a,
        dphi_a,
        alpha_b,
        phi_b,
        dphi_b,
        alpha,
        brackt,
    }
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::FnMutWrap;
    use crate::line_search::{create, LineSearchMethod};
    use approx::assert_relative_eq;

    use topohedral_linalg::VectorOps;


    //{{{ collection: line search tests
    #[test]
    fn test_fcn1() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { x[0].powi(2) + x[1].powi(2) });

        let x = SVector::<2>::from_col_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_col_slice(&[-1.0, -1.0]);

        let method = LineSearchMethod::Thuente(ThuenteOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            initial_step_size: 1.0,
            min_step_size: 1e-8,
            max_step_size: 100.0,
            max_iter: 10,
            alpha_tol: 1e-6,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        assert!(res.is_ok());
        let res = res.unwrap();
        assert_relative_eq!(res.alpha, 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.falpha, 0.0, epsilon = 1e-6);
        assert_eq!(res.funcalls, 5);
        assert_eq!(res.gradcalls, 5);
    }

    #[test]
    fn test_fcn2() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 2.0;
            let alpha = x[0];
            -alpha / (alpha.powi(2) + beta)
        });

        let method = LineSearchMethod::Thuente(ThuenteOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            initial_step_size: 1.0,
            min_step_size: 1e-8,
            max_step_size: 100.0,
            max_iter: 10,
            alpha_tol: 1e-6,
        });

        let x = SVector::<2>::from_col_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_col_slice(&[1.0, 0.0]);
        let mut line_searcher = create(f.clone(), x, dir, method);

        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        assert!(res.is_ok());
        let res = res.unwrap();
        assert_relative_eq!(res.alpha, 10.0, epsilon = 1e-6);
        assert_relative_eq!(res.falpha, -0.09803921568627451, epsilon = 1e-6);
        assert_eq!(res.funcalls, 1);
        assert_eq!(res.gradcalls, 1);
    }
    //}}}
}
//}}}