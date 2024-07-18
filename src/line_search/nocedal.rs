//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{RealFn, SMatrix, SVector};
use crate::line_search::{
    satisfies_armijo, satisfies_wolfe, Error, LineSearch, LineSearchFn, LineSearchOpts,
    LineSearchReturns,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::MatMul;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

const SMALL: f64 = 1e-32;

//{{{ fun: quadmin
/// Forms quadratic interpolation to find the minimum of a function.
///
/// Forms the function:
///
/// phi_q(x) = beta (x - a)^2 + gamma (x - a) + delta
///
/// And finds the minimum of this function ananlytically.
fn quadmin(a: f64, phi_a: f64, dphi_a: f64, b: f64, phi_b: f64) -> Option<f64> {
    //{{{ trace
    error!(target: "ls", "--- Entering quadmin ---");
    trace!(target: "ls", "Entering with phi_a = {:1.4e}, phi_b = {:1.4e}, dphi_a = {:1.4e}, b = {:1.4e}", phi_a, phi_b, dphi_a, b);
    //}}}
    let delta = phi_a;
    let gamma = dphi_a;
    let db = b - a;

    //{{{ trace
    trace!(target: "ls", "db = {:1.4e}", db);
    //}}}

    if db * db < SMALL {
        //{{{ trace
        trace!(target: "ls", "db * db too small");
        error!(target: "ls", "--- Leaving quadmin ---");
        //}}}
        return None;
    }

    let beta = (phi_b - delta - gamma * db) / (db * db);
    //{{{ trace
    trace!(target: "ls", "beta = {:1.4e}", beta);
    //}}}

    if (2.0 * beta).abs() < SMALL {
        //{{{ trace
        trace!(target: "ls", "2 * beta too small");
        error!(target: "ls", "--- Leaving quadmin ---");
        //}}}
        return None;
    }

    let alpha_min = a - gamma / (2.0 * beta);
    //{{{ trace
    error!(target: "ls", "Returning alpha_min = {:1.4e}", alpha_min);
    error!(target: "ls", "--- Leaving quadmin ---");
    //}}}
    return Some(alpha_min);
}
//}}}
//{{{ fun: cubicmin
/// Forms the cubic interpolation to find the minimum of a function.
///
/// Forms the function:
///
/// phi_cu(x) = beta(x - a)^3 + gamma (x - a)^2 + delta (x - a) + epsilon
///
/// And finds the minimum of this function ananlytically.
fn cubicmin(
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    c: f64,
    phi_c: f64,
) -> Option<f64> {
    //{{{ trace
    error!(target: "ls", "--- Entering cubicmin ---");
    trace!(target: "ls", "Enterin with a = {:1.4e}, b = {:1.4e}, c = {:1.4e}", a, b, c);
    trace!(target: "ls", "phi_a = {:1.4e}, phi_b = {:1.4e}, phi_c = {:1.4e}", phi_a, phi_b, phi_c);
    //}}}
    let db = b - a;
    let dc = c - a;
    let denom = (db * dc).powi(2) * (db - dc);
    //{{{ trace
    trace!(target: "ls", "db = {:1.4e}, dc = {:1.4e}, denom = {:1.4e}", db, dc, denom);
    //}}}

    if denom.abs() < SMALL {
        //{{{ trace
        trace!(target: "ls", "Denominator is too small");
        error!(target: "ls", "--- Leaving cubicmin ---");
        //}}}
        return None;
    }

    let mut diff_mat: SMatrix<2, 2> = SMatrix::<2, 2>::zeros();
    diff_mat[(0, 0)] = dc.powi(2);
    diff_mat[(0, 1)] = -db.powi(2);
    diff_mat[(1, 0)] = -dc.powi(3);
    diff_mat[(1, 1)] = db.powi(3);

    let mut diff_vec: SVector<2> = SVector::<2>::zeros();
    diff_vec[0] = phi_b - phi_a - dphi_a * db;
    diff_vec[1] = phi_c - phi_a - dphi_a * dc;

    let coeffs = diff_mat.matmul(&diff_vec);
    let beta = coeffs[0] / denom;
    let gamma = coeffs[1] / denom;
    let radical = (gamma * gamma - 3.0 * beta * dphi_a).sqrt();

    if (3.0 * beta).abs() < SMALL {
        //{{{ trace
        error!(target: "ls", "3 * beta too small");
        error!(target: "ls", "--- Leaving cubicmin ---");
        //}}}
        return None;
    }

    let alpha_min = a + (-gamma + radical) / (3.0 * beta);
    //{{{ trace
    error!(target: "ls", "Returning alpha_min = {:1.4e}", alpha_min);
    error!(target: "ls", "--- Leaving cubicmin ---");
    //}}}
    return Some(alpha_min);
}
//}}}
//{{{ fun: zoom
fn zoom<const N: usize, F: RealFn<N>>(
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut phi_hi: f64,
    mut dphi_lo: f64,
    phi_fcn: &mut LineSearchFn<N, F>,
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> Option<(f64, f64, f64, usize, usize)>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    //{{{ trace
    error!(target: "ls", "--- Entering zoom ---");
    trace!(target: "ls", "Entering with a_lo = {:1.4e} a_hi = {:1.4e} phi_lo = {:1.4e} phi_hi = {:1.4e}", a_lo, a_hi, phi_lo, phi_hi);
    trace!(target: "ls", "phi0 = {:1.4e} dphi0 = {:1.4e} c1 = {:1.4e} c2 = {:1.4e}", phi0, dphi0, c1, c2);
    //}}}
    let mut iter = 0;
    let delta1 = 0.2;
    let delta2 = 0.1;
    let mut phi_rec = phi0;
    let mut a_rec = 0.0;

    let mut funcalls = 0;
    let mut gradcalls = 0;

    let mut cchk = 0.0;
    let mut qchk = 0.0;

    loop {
        //{{{ trace
        debug!(target: "ls", "\n\n...............zoom iter = {}", iter);
        trace!(target: "ls", "cchk = {:1.4e}  qchk = {:1.4e}", cchk, qchk);
        //}}}

        let dalpha = a_hi - a_lo;

        let (a, b) = if dalpha < 0.0 {
            (a_hi, a_lo)
        } else {
            (a_lo, a_hi)
        };

        //{{{ trace
        trace!(target: "ls", "dalpha = {:1.4e} a_lo = {:1.4e} a_hi = {:1.4e}", dalpha, a_lo, a_hi);
        //}}}

        let mut opt_a_j: Option<f64> = None;

        // first try cubic interpolation
        if iter > 0 {
            //{{{ trace
            trace!(target: "ls", "trying cubic interpolation");
            //}}}
            cchk = delta1 * dalpha;
            opt_a_j = cubicmin(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, a_rec, phi_rec);
        }

        // if not good enough first try quadratic interpolation
        if iter == 0
            || opt_a_j.is_none()
            || opt_a_j.unwrap() > b - cchk
            || opt_a_j.unwrap() < a + cchk
        {
            //{{{ trace
            trace!(target: "ls", "trying quadratic interpolation");
            //}}}
            qchk = delta2 * dalpha;
            opt_a_j = quadmin(a_lo, phi_lo, dphi_lo, a_hi, phi_hi);

            // finally try bisection
            if opt_a_j.is_none() || opt_a_j.unwrap() > b - qchk || opt_a_j.unwrap() < a + qchk {
                //{{{ trace
                trace!(target: "ls", "Trying bisection");
                //}}}
                opt_a_j = Some(0.5 * (a + b));
            }
        }

        let a_j = opt_a_j.unwrap();
        //{{{ trace
        trace!(target: "ls", "New value of a_j = {:1.4e}", a_j);
        //}}}
        // try new value of alpha
        let phi_aj = phi_fcn.eval(a_j);
        funcalls += 1;

        let not_sat_armijo = !satisfies_armijo(c1, a_j, phi0, dphi0, phi_aj);
        let not_decreasing = phi_aj >= phi_lo;

        if not_sat_armijo || not_decreasing {
            //{{{ trace
            trace!(target: "ls", "Failed armijo condition with {} {}", not_sat_armijo, not_decreasing);
            //}}}
            phi_rec = phi_hi;
            a_rec = a_hi;
            a_hi = a_j;
            phi_hi = phi_aj;
        } else {
            //{{{ trace
            trace!(target: "ls", "Passed armijo condition");
            //}}}
            let dphi_aj = phi_fcn.eval_diff(a_j);
            gradcalls += 1;
            if dphi_aj.abs() <= -c2 * dphi0 {
                //{{{ trace
                trace!(target: "ls","Passed curvature condition");
                error!(target: "ls", "Returning a_j = {:1.4e} phi_aj = {:1.4e} dphi_aj = {:1.4e}", 
                      a_j, phi_aj, dphi_aj);
                error!(target: "ls", "--- Leaving zoom ---");
                //}}}
                return Some((a_j, phi_aj, dphi_aj, funcalls, gradcalls));
            }

            if dphi_aj * (a_hi - a_lo) >= 0.0 {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_lo;
                phi_hi = phi_lo;
            } else {
                phi_rec = phi_lo;
                a_rec = a_lo;
            }

            a_lo = a_j;
            phi_lo = phi_aj;
            dphi_lo = dphi_aj;
        }
        iter += 1;
        if iter == max_iter {
            //{{{ trace
            error!(target: "ls", "Reached max iterations in zoom");
            error!(target: "ls", "--- Leaving zoom ---");
            //}}}
            return None;
        }
    }
}
//}}}
//{{{ struct: NocedalOpts
#[derive(Debug, Clone)]
pub struct NocedalOpts {
    pub ls_opts: LineSearchOpts,
    pub initial_step_size: f64,
    pub max_iter: usize,
    pub zoom_max_iter: usize,
}
//}}}
//{{{ struct NocedalLineSearch
pub struct NocedalLineSearch<const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub(crate) f: LineSearchFn<N, F>,
    pub opts: NocedalOpts,
}
//}}}
//{{{ impl: NocedalLineSearch
impl<const N: usize, F: RealFn<N>> NocedalLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub fn new(f: F, x: SVector<N>, dir: SVector<N>, opts: NocedalOpts) -> Self {
        Self {
            f: LineSearchFn { f, x, dir },
            opts,
        }
    }
}
//}}}
//{{{ impl: LineSearch for NocedalLineSearch
impl<const N: usize, F: RealFn<N>> LineSearch<N> for NocedalLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, Error> {
        //{{{ trace
        error!(target: "ls", "--- Entering line_search ---");
        trace!(target: "ls", "Entering with phi0 = {:1.4e} dphi0 = {:1.4e}", phi0, dphi0);
        //}}}

        if dphi0 > 0.0 {
            return Err(Error::NotDecreasing);
        }

        let mut funcalls = 0;
        let mut gradcalls = 0;
        let c1 = self.opts.ls_opts.c1;
        let c2 = self.opts.ls_opts.c2;
        let mut alpha0 = 0.0;
        let mut phi_a0 = phi0;
        let mut dphi_a0 = dphi0;
        let mut dphi_a1 = 0.0;
        let mut alpha1 = self.opts.initial_step_size;
        let mut phi_a1 = self.f.eval(alpha1);
        funcalls += 1;

        let max_iter = self.opts.max_iter;

        for i in 0..max_iter {
            //{{{ trace
            trace!(target: "ls", "\n\n--------------------------------------- nocedal it = {}", i);
            trace!(target: "ls", "alpha0 = {:1.4e} alpha1 = {:1.4e}", alpha0, alpha1);
            trace!(target: "ls", "phi_a0 = {:1.4e} phi1 {:1.4e}", phi_a0, phi_a1);
            trace!(target: "ls", "dphi_aa0 = {:1.4e} dphi1 {:1.4e}", dphi_a0, dphi_a1);
            //}}}

            if alpha1 < f64::EPSILON * 100.0 {
                //{{{ trace
                error!(target: "ls", "Too small step size detected");
                error!("--- Leaving line_search ---");
                //}}}
                return Err(Error::StepSizeSmall);
            }

            let not_first_iteration = i > 0;
            let not_decreasing = phi_a1 >= phi_a0;

            // First check if current step is armijo-acceptable, if not then try zoom and check
            // again.
            if !satisfies_armijo(c1, alpha1, phi_a0, dphi_a0, phi_a1)
                || (not_decreasing && not_first_iteration)
            {
                //{{{ trace
                trace!(target: "ls", "Does not satisfy armijo");
                //}}}
                let zoom_result = zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    &mut self.f,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                    self.opts.zoom_max_iter,
                );

                let (alpha_tmp, phi_tmp, dphi_tmp, funcalls_tmp, gradcalls_tmp) = match zoom_result
                {
                    None => {
                        //{{{ trace
                        error!(target: "ls","Zoom failed");
                        error!(target: "ls","--- Leaving line_search ---");
                        //}}}
                        return Err(Error::NotDecreasing);
                    }
                    Some(result) => result,
                };

                funcalls += funcalls_tmp;
                gradcalls += gradcalls_tmp;

                //{{{ trace
                error!(target: "ls", "Leaving with {:1.4e} {:1.4e} {} {}", alpha_tmp, phi_tmp, funcalls, gradcalls);
                error!(target: "ls", "--- Leaving line_search ---");
                //}}}
                return Ok(LineSearchReturns {
                    alpha: alpha_tmp,
                    falpha: phi_tmp,
                    funcalls: funcalls,
                    gradcalls: gradcalls,
                });
            }

            // current step is armijo-acceptable, so check if curvature-accepttable
            dphi_a1 = self.f.eval_diff(alpha1);
            gradcalls += 1;

            if dphi_a1.abs() <= -c2 * dphi0 {
                //{{{ trace
                trace!(target: "ls", "Does not satisfy curvature");
                trace!(target: "ls","Returning alpha = {:1.4e} falpha = {:1.4e}", alpha1, phi_a1);
                error!(target: "ls", "--- Leaving line_search ---");
                //}}}
                return Ok(LineSearchReturns {
                    alpha: alpha1,
                    falpha: phi_a1,
                    funcalls: funcalls,
                    gradcalls: gradcalls,
                });
            }

            if dphi_a1 >= 0.0 {
                //{{{ trace
                trace!(target: "ls", "Curvature is positive {:1.4e}", dphi_a1);
                //}}}
                let zoom_result = zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    &mut self.f,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                    self.opts.zoom_max_iter,
                );
                let (alpha_tmp, phi_tmp, dphi_tmp, funcalls_tmp, gradcalls_tmp) = match zoom_result
                {
                    None => {
                        //{{{  trace
                        error!(target: "ls", "Zoom failed");
                        error!(target: "ls", "--- Leaving line_search ---");
                        //}}}
                        return Err(Error::NotDecreasing);
                    },
                    Some(result) => {
                        //{{{ trace
                        trace!(target: "ls","Zoom succeeded with result {:1.4e} {:1.4e} {:1.4e} {} {}", 
                                  result.0, result.1, result.2, result.3, result.4);
                        //}}}
                        result
                    },
                };

                funcalls += funcalls_tmp;
                gradcalls += gradcalls_tmp;

                //{{{ trace
                error!(target: "ls", "Returning alpha = {:1.4e} falpha = {:1.4e}", alpha_tmp, phi_tmp);
                error!(target: "ls", "--- Leaving line_search ---");
                //}}}
                return Ok(LineSearchReturns {
                    alpha: alpha_tmp,
                    falpha: phi_tmp,
                    funcalls: funcalls,
                    gradcalls: gradcalls,
                });
            }

            //{{{ trace
            trace!(target: "ls", "Doubling alpha for next iteration to {:1.4e}", 2.0 * alpha1);
            //}}}
            let alpha2 = 2.0 * alpha1;
            alpha0 = alpha1;
            alpha1 = alpha2;
            phi_a0 = phi_a1;
            dphi_a0 = dphi_a1;
            phi_a1 = self.f.eval(alpha1);
        }

        //{{{ trace
        error!("Reached max number of iterations"); 
        error!(target: "ls", "--- Leaving line_search ---");
        //}}}
        return Ok(LineSearchReturns {
            alpha: alpha1,
            falpha: phi_a1,
            funcalls: funcalls,
            gradcalls: gradcalls,
        });
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

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_search::{create, FnMutWrap, LineSearchMethod};
    use approx::assert_relative_eq;
    use topohedral_linalg::EvaluateSMatrix;

    //{{{ collection: quadmin tests
    #[test]
    fn test_quadmin_normal_case() {
        let result = quadmin(0.0, 1.0, -1.0, 2.0, 5.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 0.333333333333, epsilon = 1e-10);
    }

    #[test]
    fn test_quadmin_small_interval() {
        let result = quadmin(1.0, 2.0, -1.0, 1.0 + 1e-11, 2.0);
        // assert!(result.is_none());
    }

    #[test]
    fn test_quadmin_small_curvature() {
        let result = quadmin(0.0, 1.0, -1e-5, 1.0, 1.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quadmin_large_values() {
        let result = quadmin(1000.0, 5000.0, -100.0, 2000.0, 8000.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 1485.43689, epsilon = 1e-5);
    }
    //}}}
    //{{{ collection: cubmin tests
    #[test]
    fn test_cubmin_standard_case() {
        let result = cubicmin(0.0, 1.0, -1.0, 1.0, 0.5, 2.0, 2.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 0.8685170918213297, epsilon = 1e-10);
    }

    #[test]
    fn test_cubmin_collinear_points() {
        let result = cubicmin(0.0, 0.0, -1.0, 1.0, 1.0, 2.0, 2.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 0.18350341907227405, epsilon = 1e-10);
    }

    #[test]
    fn test_cubmin_large_values() {
        let result = cubicmin(1000.0, 5000.0, -100.0, 2000.0, 8000.0, 3000.0, 12000.0);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap(), 1406.50406, epsilon = 1e-5);
    }

    #[test]
    fn test_cubmin_small_interval() {
        let result = cubicmin(1.0, 1.0, -1.0, 1.0 + 1e-11, 1.0, 1.0 + 2e-11, 1.0);
        assert!(result.is_none());
    }
    //}}}
    //{{{ collection: zoom tests
    #[test]
    fn test_zoom_quad_left() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { (x[0] - 2.0).powi(2) });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let mut fline = LineSearchFn {
            f: f.clone(),
            x: x.clone(),
            dir: dir.clone(),
        };

        let phi0 = fline.eval(0.0);
        let dphi0 = fline.eval_diff(0.0);
        let phi1 = fline.eval(1.0);
        let res = zoom(
            0.0, 1.0, phi0, phi1, dphi0, &mut fline, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 0.5, epsilon = 1e-6);
        assert_relative_eq!(res.1, 2.25, epsilon = 1e-6);
        assert_relative_eq!(res.2, -3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zoom_quad_center() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { (x[0] - 2.0).powi(2) });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let mut fline = LineSearchFn {
            f: f.clone(),
            x: x.clone(),
            dir: dir.clone(),
        };

        let a0 = -10.0;
        let phi0 = fline.eval(a0);
        let dphi0 = fline.eval_diff(a0);
        let a_lo = -5.0;
        let a_hi = 15.0;
        let phi_lo = fline.eval(a_lo);
        let dphi_lo = fline.eval_diff(a_lo);
        let phi_hi = fline.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut fline, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.1, 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.2, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zoom_quad_right_none() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { (x[0] - 2.0).powi(2) });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let mut fline = LineSearchFn {
            f: f.clone(),
            x: x.clone(),
            dir: dir.clone(),
        };

        let a0 = 3.0;
        let phi0 = fline.eval(a0);
        let dphi0 = fline.eval_diff(a0);
        let a_lo = 5.0;
        let a_hi = 100.0;
        let phi_lo = fline.eval(a_lo);
        let dphi_lo = fline.eval_diff(a_lo);
        let phi_hi = fline.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut fline, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_none());
    }

    #[test]
    fn test_zoom_quad_right_some() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { (x[0] - 2.0).powi(2) });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let mut fline = LineSearchFn {
            f: f.clone(),
            x: x.clone(),
            dir: dir.clone(),
        };

        let a0 = 1.0;
        let phi0 = fline.eval(a0);
        let dphi0 = fline.eval_diff(a0);
        let a_lo = 1.5;
        let a_hi = 100.0;
        let phi_lo = fline.eval(a_lo);
        let dphi_lo = fline.eval_diff(a_lo);
        let phi_hi = fline.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut fline, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.1, 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.2, 0.0, epsilon = 1e-6);
    }
    //}}}
    //{{{ collection: line search tests
    #[test]
    fn test_fcn1() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { x[0].powi(2) + x[1].powi(2) });

        let x = SVector::<2>::from_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, -1.0]);

        let method = LineSearchMethod::Nocedal(NocedalOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            initial_step_size: 10.0,
            max_iter: 5,
            zoom_max_iter: 10,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        assert!(res.is_ok());
        let res = res.unwrap();
        assert_relative_eq!(res.alpha, 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.falpha, 0.0, epsilon = 1e-6);
        assert_eq!(res.funcalls, 3);
        assert_eq!(res.gradcalls, 1);
    }

    #[test]
    fn test_fcn2() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 2.0;
            let alpha = x[0];
            -alpha / (alpha.powi(2) + beta)
        });

        let method = LineSearchMethod::Nocedal(NocedalOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            initial_step_size: 10.0,
            max_iter: 5,
            zoom_max_iter: 10,
        });

        let mut x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let mut dir = SVector::<2>::from_slice(&[1.0, 0.0]);
        let mut line_searcher = create(f.clone(), x, dir, method);

        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        println!("{:?}", res);

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
