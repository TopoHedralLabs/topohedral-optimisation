//! Simple backtracking line search via bisection
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{RealFn, SVector};
use crate::line_search::{
    satisfies_wolfe, Error, LineSearch, LineSearchFn, LineSearchOpts, LineSearchReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;  
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BacktrackingOpts {
    pub ls_opts: LineSearchOpts,
    pub initial_step_size: f64,
    pub factor: f64,
    pub max_iter: usize,
}

pub struct BacktrackingLineSearch<const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub(crate) f: LineSearchFn<N, F>,
    pub opts: BacktrackingOpts,
}

impl<const N: usize, F: RealFn<N>> BacktrackingLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub fn new(f: F, x: SVector<N>, dir: SVector<N>, opts: BacktrackingOpts) -> Self {
        Self {
            f: LineSearchFn { f, x, dir },
            opts,
        }
    }
}

impl<const N: usize, F: RealFn<N>> LineSearch<N> for BacktrackingLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, Error> {


        if dphi0 > 0.0 {
            return Err(Error::NotDecreasing);
        }

        let c1 = self.opts.ls_opts.c1;
        let c2 = self.opts.ls_opts.c2;
        let mut alpha_k = self.opts.initial_step_size;
        let rho_down = self.opts.factor;
        let max_iter = self.opts.max_iter;

        let mut funcalls = 0;
        let mut gradcalls = 0;

        let mut i = 0;
        loop {
            //{{{ trace
            trace!(target: "ls", "================================== i = {}", i);
            trace!(target: "ls", "Looking left with alpha_k = {:1.4e}", alpha_k);
            //}}}
            let phi1 = self.f.eval(alpha_k);
            let dphi1 = self.f.eval_diff(alpha_k);
            if let Ok(()) = satisfies_wolfe(c1, c2, phi0, dphi0, phi1, dphi1, alpha_k) {
                //{{{ trace
                trace!(target: "ls", "Satisfied wolfe conditions with alpha_l = {:1.4e}", alpha_k);  
                //}}}
                return Ok(LineSearchReturns {
                    alpha: alpha_k,
                    falpha: phi1,
                    funcalls,
                    gradcalls,
                });
            }

            alpha_k = alpha_k * rho_down;
            funcalls += 1;
            gradcalls += 1;
            i += 1;

            if i == max_iter {
                return Err(Error::MaxIterations)
            }
        }
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

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use topohedral_linalg::EvaluateSMatrix;
    use crate::common::FnMutWrap;
    use crate::line_search::{create, LineSearchMethod};
    use super::*;




    #[test]
    fn test_fcn1()
    {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            x[0].powi(2) + x[1].powi(2)
        });

        let x = SVector::<2>::from_slice(&[1.0, 1.0]);
        let dir = SVector::<2>::from_slice(&[-1.0, -1.0]);

        let method = LineSearchMethod::Backtracking(BacktrackingOpts{
            ls_opts: LineSearchOpts {
                c1: 1e-4,
                c2: 0.9,
            },
            initial_step_size: 10.0,
            factor: 0.5,
            max_iter: 5,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);
        
        let alpha = res.unwrap().alpha;
        let x1: SVector<2> = (&x + alpha * &dir).evals();
        println!("res: x1 = {}  alpha = {}", x1, alpha);
    }

    #[test] 
    fn test_fcn2() 
    {

        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 2.0;
            let alpha = x[0];
            - alpha / (alpha.powi(2) + beta)
        });



        let method = LineSearchMethod::Backtracking(BacktrackingOpts{
            ls_opts: LineSearchOpts {
                c1: 1e-4,
                c2: 0.9,
            },
            initial_step_size: 10.0,
            factor: 0.5,
            max_iter: 5,
        });

        let mut x = SVector::<2>::from_slice(&[0.0, 0.0]);
        let mut dir = SVector::<2>::from_slice(&[1.0, 0.0]);
        let mut line_searcher = create(f.clone(), x, dir, method);

        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res1 = line_searcher.line_search(phi0, dphi0);
    }


    #[test]
    fn test_fn3() 
    {
        let mut f = FnMutWrap::new(|x: &SVector<2>| -> f64 {
            let beta = 0.004;
            let alpha = x[0];
            (alpha + beta).powi(5) - 2.0 * (alpha + beta).powi(4)
        });

        let x = SVector::<2>::from_slice(&[0.2, 0.0]);
        let dir = SVector::<2>::from_slice(&[1.0, 0.0]);

        let method = LineSearchMethod::Backtracking(BacktrackingOpts{
            ls_opts: LineSearchOpts {
                c1: 1e-4,
                c2: 0.9,
            },
            initial_step_size: 10.0,
            factor: 0.75,
            max_iter: 10,
        });

        let mut line_searcher = create(f.clone(), x, dir, method);
        let phi0 = f.eval(&x);
        let dphi0 = f.grad(&x).dot(&dir);
        let res = line_searcher.line_search(phi0, dphi0);

        println!("res: {:?}", res);
    }




}
//}}}