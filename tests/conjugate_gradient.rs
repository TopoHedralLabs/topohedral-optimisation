//! Tests for multi-dimensional optimization.
//!
//!
//--------------------------------------------------------------------------------------------------

//{{{ features
#![feature(generic_const_exprs)]
#![feature(impl_trait_in_assoc_type)]
//}}}
//{{{ crate imports 
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use topohedral_optimisation::{SVector, RealFn, EvaluateSMatrix};
use topohedral_optimisation::line_search as ls;
use topohedral_optimisation::unconstrained_min as umin;
use topohedral_optimisation::unconstrained_min::conjugate_gradient as cg;
use topohedral_tracing::*;
use ctor::ctor;
use rstest::rstest;
use approx::assert_relative_eq;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ fun: init_logger
#[ctor]
fn init_logger() {
    init().unwrap();
}
//}}}


//{{{ struct: Quadratic
#[derive(Clone)]
struct Quadratic
{
    xmin: SVector<5>
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn<5> for Quadratic {
    fn eval(&mut self, x: &SVector<5>) -> f64 {
        
        let tmp: SVector<5> = (x - &self.xmin).evals();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(2);
        }
        out
    }

    fn grad(&mut self, x_in: &SVector<5>) -> SVector<5> {
        
        let tmp: SVector<5> = (x_in - &self.xmin).evals();
        let mut out = SVector::<5>::zeros(); 
        for i in 0..5 {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}
//{{{ fun: test_quadratic
//{{{ case1: FletcherReeves + FixedStep
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::FixedStep(ls::FixedStepOpts{
                ls_opts: ls::LineSearchOpts{
                    c1: 1e-4,
                    c2: 0.4,
                },
                step_size: 0.5,
            }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        2, 
        2)]
//}}}
//{{{ case2: PolakRibiere + FixedStep
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::FixedStep(ls::FixedStepOpts{
                ls_opts: ls::LineSearchOpts{
                    c1: 1e-4,
                    c2: 0.4,
                },
                step_size: 0.5,
            }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        2, 
        2)]
//}}}
//{{{ case3: FletcherReeves + Backtracking 
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::Backtracking(ls::BacktrackingOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 2.0,
            factor: 0.5,
            max_iter: 100,
        }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        3, 
        3)]
//}}}
//{{{ case4: PolakRibiere + Backtracking
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::Backtracking(ls::BacktrackingOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 2.0,
            factor: 0.5,
            max_iter: 100,
        }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        3, 
        3)]
//}}}
//{{{ case5: FletcherReeves + Nocedal 
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 100.0,
            max_iter: 100,
            zoom_max_iter: 15,
        }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        8, 
        2)]
//}}}
//{{{ case6: PolakRibiere + Nocedal 
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 100.0,
            max_iter: 100,
            zoom_max_iter: 15,
        }),
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        0.0, 
        8, 
        2)]
//}}}
fn test_quadratic(  #[case] direction_method: cg::DirectionMethod, 
                    #[case] ls_method: ls::LineSearchMethod,
                    #[case] exp_xmin: SVector<5>, 
                    #[case] exp_fmin: f64, 
                    #[case] exp_funcalls: usize, 
                    #[case] exp_gradcalls: usize) {

    let f = Quadratic {xmin: exp_xmin};
    let x0 = SVector::from_slice(&[1000.0, -100.0, 0.0 , 567.0, -23.0]);
    let method  = umin::Method::CG(cg::Opts{
        unconstrained_opts: umin::Opts {
            abstol: 1e-12,
            max_iter: 1000,
            ls_method: ls_method,
        },
        direction_method: direction_method
    });

    let mut minimizer = umin::create(method);
    let res = minimizer.minimize(f, x0).unwrap();
    let xdiff: SVector<5> = (&res.xmin - &exp_xmin).evals();
    assert!(xdiff.norm() < 1e-6);
    assert_relative_eq!(res.fmin, exp_fmin, epsilon = 1e-6);
    assert_eq!(res.funcalls, exp_funcalls);
    assert_eq!(res.gradcalls, exp_gradcalls);
}
//}}}

//{{{ struct: Quartic 
#[derive(Clone, Debug)]
struct Quartic {
    xmin: SVector<5>
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn<5> for Quartic {

    fn eval(&mut self, x: &SVector<5>) -> f64 {
        let tmp: SVector<5> = (x - &self.xmin).evals();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn grad(&mut self, x_in: &SVector<5>) -> SVector<5> {
        let tmp: SVector<5> = (x_in - &self.xmin).evals();
        let mut out = SVector::<5>::zeros();
        for i in 0..5 {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
//}}}
//{{{ fun: test_quartic
//{{{ case1: FletcherReeves + FixedStep
// some curious things about this test, only converges within 5000 iterations
// with a step size of 1e-4, slightly larger (e.g. 1.5e-4) and it eventually 
// diverges.
// Additionally, this one diverges for abstol tighter that 1e-8
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::FixedStep(ls::FixedStepOpts{
                ls_opts: ls::LineSearchOpts{
                    c1: 1e-4,
                    c2: 0.4,
                },
                step_size: 1.0e-4, 
            }),
        1e-8,
        1e-3,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        Some((0.0, 4039, 4040, 4040)))]
//}}}
//{{{ case2: PolakRibiere + FixedStep
// there does not seem to be a step length for which this converges in 
// reasonable time
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::FixedStep(ls::FixedStepOpts{
                ls_opts: ls::LineSearchOpts{
                    c1: 1e-4,
                    c2: 0.4,
                },
                step_size: 1.0e-4,
            }),
        1e-4,
        1e-3,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        None )]
//}}}
//{{{ case3: FletcherReeves + Backtracking 
// initial step length of 1 is too small for this case as the direction vector
// becomes small toward the end of the optimisation, 10 appears to work however
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::Backtracking(ls::BacktrackingOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e1,
            factor: 0.5,
            max_iter: 100,
        }),
        1e-8,
        1e-4,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        Some((0.0, 13, 130, 130)))]
//}}}
//{{{ case4: PolakRibiere + Backtracking
// initial step length of 1 is too small for this case as the direction vector
// becomes small toward the end of the optimisation, 100 appears to work however
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::Backtracking(ls::BacktrackingOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e2,
            factor: 0.5,
            max_iter: 100,
        }),
        1e-8,
        1e-2,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        Some((0.0, 8, 77, 77)))]
//}}}
//{{{ case5: FletcherReeves + Nocedal 
// mostly works but seems to get stuck for tolerances tighter than 1e-5
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e0,
            max_iter: 100,
            zoom_max_iter: 50,
        }),
        1e-5,
        5e-2,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        Some((0.0, 9, 65, 15)))]
//}}}
//{{{ case6: PolakRibiere + Nocedal 
// This combination appears to be the most robust
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e1,
            max_iter: 100,
            zoom_max_iter: 15,
        }),
        1e-10,
        1e-3,
        SVector::<5>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 
        Some((0.0, 13 ,105, 27)))]
//}}}
fn test_quartic( #[case] direction_method: cg::DirectionMethod, 
                 #[case] ls_method: ls::LineSearchMethod,
                 #[case] abstol: f64,
                 #[case] exp_xdiff_tol: f64,
                 #[case] exp_xmin: SVector<5>, 
                 #[case] results: Option<(f64, usize, usize, usize)>) {

    let f = Quartic{xmin: exp_xmin};
    let x0 = SVector::from_slice(&[10.0, -10.0, 0.0 , 57.0, -23.0]);

    let method  = umin::Method::CG(cg::Opts{
        unconstrained_opts: umin::Opts {
            abstol: abstol,
            max_iter: 5000,
            ls_method: ls_method,
        },
        direction_method: direction_method
    });

    let mut minimizer = umin::create(method);
    let res = minimizer.minimize(f, x0);

    match results {
        Some((exp_fmin, exp_iters, exp_funcalls, exp_gradcalls)) => {

            let res = res.unwrap();
            let xdiff: SVector<5> = (&res.xmin - &exp_xmin).evals();
            assert!(xdiff.norm() < exp_xdiff_tol);
            assert_relative_eq!(res.fmin, exp_fmin, epsilon = 1e-4);
            assert_eq!(res.iter, exp_iters);
            assert_eq!(res.funcalls, exp_funcalls);
            assert_eq!(res.gradcalls, exp_gradcalls);
        },
        None => {
            assert!(res.is_err());
        }
    }
}
//}}}





#[derive(Clone, Debug)]
struct RosenBrock {
    a: f64,
    b: f64,
}

impl RealFn<2> for RosenBrock {

    fn eval(&mut self, xvec: &SVector<2>) -> f64 {
        let x = xvec[0];
        let y = xvec[1];
        let a = self.a;
        let b = self.b;
        let out = (a - x).powi(2) + b * (y - x.powi(2)).powi(2);
        out
    }

    fn grad(&mut self, xvec: &SVector<2>) -> SVector<2> {

        let x = xvec[0];
        let y = xvec[1];
        let a = self.a;
        let b = self.b;
        let out = SVector::<2>::from_slice(&[
            -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2)), 
            2.0 * b * (y - x.powi(2))
        ]);
        out
    }
} 


//{{{ case1: FletcherReeves + Nocedal 
// mostly works but seems to get stuck for tolerances tighter than 1e-5
#[rstest]
#[case(cg::DirectionMethod::FletcherReeves, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e0,
            max_iter: 10,
            zoom_max_iter: 30,
        }),
        1e-5,
        5e-2,
        SVector::<2>::from_slice(&[1.0, 1.0]), 
        Some((0.0, 4438, 55575, 4446, 1401)))]
//}}}
//{{{ case2: PolakRibiere + Nocedal 
// This combination appears to be the most robust
#[case(cg::DirectionMethod::PolakRibiere, 
        ls::LineSearchMethod::Nocedal(ls::NocedalOpts{
            ls_opts: ls::LineSearchOpts{
                c1: 1e-4,
                c2: 0.4,
            },
            initial_step_size: 1.0e0,
            max_iter: 10,
            zoom_max_iter: 30,
        }),
        1e-5,
        1e-3,
        SVector::<2>::from_slice(&[1.0, 1.0]), 
        Some((0.0, 331 ,2743, 337, 289)))]
//}}}
fn test_rosenbrock(#[case] direction_method: cg::DirectionMethod, 
                   #[case] ls_method: ls::LineSearchMethod,
                   #[case] abstol: f64,
                   #[case] exp_xdiff_tol: f64,
                   #[case] exp_xmin: SVector<2>, 
                   #[case] results: Option<(f64, usize, usize, usize, usize)>) {
                    
    let f = RosenBrock{a: 1.0, b: 100.0};
    let x0 = SVector::from_slice(&[0.0, 3.0]);

    let method  = umin::Method::CG(cg::Opts{
        unconstrained_opts: umin::Opts {
            abstol: abstol,
            max_iter: 5000,
            ls_method: ls_method,
        },
        direction_method: direction_method
    });

    let mut minimizer = umin::create(method);
    let res = minimizer.minimize(f, x0);

    println!("res: {:?}", res);

    match results {
        Some((exp_fmin, exp_iters, exp_funcalls, exp_gradcalls, exp_restarts)) => {

            let res = res.unwrap();
            let xdiff: SVector<2> = (&res.xmin - &exp_xmin).evals();
            assert!(xdiff.norm() < exp_xdiff_tol);
            assert_relative_eq!(res.fmin, exp_fmin, epsilon = 1e-4);
            assert_eq!(res.iter, exp_iters);
            assert_eq!(res.funcalls, exp_funcalls);
            assert_eq!(res.gradcalls, exp_gradcalls);
            assert_eq!(res.num_restarts, exp_restarts);
        },
        None => {
            assert!(res.is_err());
        }
    }
}