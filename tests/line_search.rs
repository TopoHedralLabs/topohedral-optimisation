#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]

mod thuente {

    use approx::assert_relative_eq;
    
    use topohedral_linalg::VectorOps;
    use topohedral_optimisation::{
        line_search::{self as ls, LineSearchMethod, LineSearchOpts, ThuenteOpts},
        RealFn, SVector,
    };

    #[derive(Clone)]
    struct Fcn1 {
        beta: f64,
    }

    impl RealFn<2> for Fcn1 {
        fn eval(&mut self, x: &SVector<2>) -> f64 {
            let alpha = x[0];
            -alpha / (alpha.powi(2) + self.beta)
        }
        fn grad(&mut self, x_in: &SVector<2>) -> SVector<2> {
            let alpha = x_in[0];
            let mut out = SVector::<2>::zeros();
            out[0] = (alpha.powi(2) - self.beta) / (alpha.powi(2) + self.beta).powi(2);
            out[1] = 0.0;
            out
        }
    }

    #[test]
    fn test_fcn1() {
        let method = LineSearchMethod::Thuente(ThuenteOpts {
            ls_opts: LineSearchOpts { c1: 1e-4, c2: 0.9 },
            initial_step_size: 1.0,
            min_step_size: 1e-8,
            max_step_size: 100.0,
            max_iter: 10,
            alpha_tol: 1e-6,
        });
        let mut f = Fcn1 { beta: 2.0 };
        let x = SVector::<2>::from_col_slice(&[0.0, 0.0]);
        let dir = SVector::<2>::from_col_slice(&[1.0, 0.0]);
        let mut line_searcher = ls::create(f.clone(), x, dir, method);
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
}
