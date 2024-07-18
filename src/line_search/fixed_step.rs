//! Fixed step line search. Step size is fixed and accepted unconditionally.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{RealFn, SMatrix, SVector};
use crate::line_search::{
    satisfies_wolfe, Error, LineSearch, LineSearchFn, LineSearchOpts, LineSearchReturns,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: FixedStepOpts
#[derive(Debug, Clone)]
pub struct FixedStepOpts {
    pub ls_opts: LineSearchOpts,
    pub step_size: f64,
}
//}}}
//{{{ struct: FixedStepLineSearch
pub struct FixedStepLineSearch<const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub(crate) f: LineSearchFn<N, F>,
    pub opts: FixedStepOpts,
}
//}}}
impl<const N: usize, F: RealFn<N>> FixedStepLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub fn new(f: F, x: SVector<N>, dir: SVector<N>, opts: FixedStepOpts) -> Self {
        Self {
            f: LineSearchFn { f, x, dir },
            opts,
        }
    }
}
//{{{ impl: LineSearch for FixedStepLineSearch
impl<const N: usize, F: RealFn<N>> LineSearch<N> for FixedStepLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, Error> {

        if dphi0 > 0.0 {
            return Err(Error::NotDecreasing);
        }

        let alpha = self.opts.step_size;
        let falpha = self.f.eval(alpha);
        let funcalls = 1;
        let gradcalls = 1;
        Ok(LineSearchReturns {
            alpha,
            falpha,
            funcalls,
            gradcalls,
        })
    }

    fn set_location_and_direction(&mut self, x: SVector<N>, dir: SVector<N>) {
        self.f.x = x;
        self.f.dir = dir;
    }

    fn set_initial_step_size(&mut self, alpha: f64) {
        self.opts.step_size = alpha;
    }

    fn get_initial_step_size(&self) -> f64 {
        self.opts.step_size
    }
}
//}}}

//{{{ mod: tests
#[cfg(test)]
mod tests {
    use crate::common::FnMutWrap;
    use crate::line_search::{create, satisfies_armijo, satisfies_curvature, BacktrackingOpts, LineSearchMethod};
    use topohedral_linalg::EvaluateSMatrix;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_fcn1_ok() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { x[0].powi(2) + x[1].powi(2) });

        let x = SVector::<2>::from_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, -1.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 1.0,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);
        assert!(res.is_ok());
        let res = res.unwrap();

        assert_abs_diff_eq!(res.alpha, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(res.falpha, 0.0, epsilon = 1e-6);
        assert_eq!(res.funcalls, 1);
        assert_eq!(res.gradcalls, 1);
    }
    //..............................................................................................

    #[test]
    fn test_fcn1_armijo_err() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { x[0].powi(2) + x[1].powi(2) });

        let x = SVector::<2>::from_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, -1.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 10.0,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0).unwrap();

        assert!(!satisfies_armijo(1e-4, res.alpha, phi0, dphi0, res.falpha));
    }
    //..............................................................................................

    #[test]
    fn test_fcn1_non_decreasing_err() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 { x[0].powi(2) + x[1].powi(2) });

        let x = SVector::<2>::from_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 1.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 10.0,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);

        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), Error::NotDecreasing);
    }
    //..............................................................................................

    #[test]
    fn test_fcn2_ok() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 2.0;
            let alpha = x[0];
            -alpha / (alpha.powi(2) + beta)
        });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 1.0,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        println!("res: {}", res.unwrap().alpha);
    }
    //..............................................................................................

    #[test]
    fn test_fcn2_curvature_err() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 2.0;
            let alpha = x[0];
            -alpha / (alpha.powi(2) + beta)
        });

        let x = SVector::<2>::from_slice(&[10.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, 0.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 0.1,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        let xnew = (&x + res.unwrap().alpha * &dir).evals();
        let dphi_new = f.grad(&xnew).dot(&dir);
        assert!(!satisfies_curvature(0.9, dphi0, dphi_new));
    }

    #[test]
    fn test_fcn3_ok() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 0.004;
            let alpha = x[0];
            (alpha + beta).powi(5) - 2.0 * (alpha + beta).powi(4)
        });

        let x = SVector::<2>::from_slice(&[2.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, 0.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 0.1,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);
        assert!(res.is_ok());
    }

    #[test]
    fn test_fcn3_err_curvature() {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 0.004;
            let alpha = x[0];
            (alpha + beta).powi(5) - 2.0 * (alpha + beta).powi(4)
        });

        let x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let method = LineSearchMethod::FixedStep(FixedStepOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            step_size: 0.1,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        let xnew = (&x + res.unwrap().alpha * &dir).evals();
        let dphi_new = f.grad(&xnew).dot(&dir);
        assert!(!satisfies_curvature(0.9, dphi0, dphi_new));
    }
}
//}}}
