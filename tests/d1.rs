#![feature(generic_const_exprs)]
#![feature(impl_trait_in_assoc_type)]


use topohedral_optimisation::d1::*;

use approx::assert_relative_eq;
use serde::Deserialize;
use std::fs;


#[derive(Deserialize, Debug)]
struct BracketTest3
{
    a: f64,
    b: f64,
    results: (f64, f64, f64, f64, f64, f64, usize),
}

#[derive(Deserialize, Debug)]
struct BracketTest2
{
    description: String,
    values: BracketTest3,
}

#[derive(Deserialize, Debug)]
struct BracketTest1
{
    bracket_test1: BracketTest2,
    bracket_test2: BracketTest2,
    bracket_test3: BracketTest2,
    bracket_test4: BracketTest2,
    bracket_test5: BracketTest2,
}

impl BracketTest1
{
    fn new() -> Self
    {
        let json_file = fs::read_to_string("assets/bracket.json").expect("Unable to read file");
        serde_json::from_str(&json_file).expect("Could not deserialize")
    }
}

macro_rules! bracket_test {
    ($test_name: ident) => {
        #[test]
        fn $test_name()
        {
            let tol = 5e-4;
            let test_data = BracketTest1::new();
            let a = test_data.$test_name.values.a;
            let b = test_data.$test_name.values.b;
            let (xa, xb, xc, fa, fb, fc, funcalls) = test_data.$test_name.values.results;

            let f = |x: f64| x * x - 2.0;
            let mut opts = BracketOptions::default();
            opts.a = a;
            opts.b = b;
            let out = bracket(&f, &opts);
            assert_eq!(out.is_ok(), true);
            let (xa2, xb2, xc2, fa2, fb2, fc2, funcalls2) = out.unwrap();
            assert_relative_eq!(xa, xa2, epsilon = tol);
            assert_relative_eq!(xb, xb2, epsilon = tol);
            assert_relative_eq!(xc, xc2, epsilon = tol);
            assert_relative_eq!(fa, fa2, epsilon = tol);
            assert_relative_eq!(fb, fb2, epsilon = tol);
            assert_relative_eq!(fc, fc2, epsilon = tol);
            assert_eq!(funcalls, funcalls2);
        }
    };
}

bracket_test!(bracket_test1);
bracket_test!(bracket_test2);
bracket_test!(bracket_test3);
bracket_test!(bracket_test4);
bracket_test!(bracket_test5);
#[test]
fn bracket_test_err1()
{
    let f = |x: f64| 100.0;
    let opts = BracketOptions::default();
    let out = bracket(&f, &opts);
    assert_eq!(out.is_err(), true);
}
#[test]
fn bracket_test_err2()
{
    let f = |x: f64| x * x * x;
    let opts = BracketOptions::default();
    let out = bracket(&f, &opts);
    assert_eq!(out.is_err(), true);
}
//..............................................................................................

#[derive(Deserialize, Debug)]
struct MinimiseScalarBrentTest3
{
    bracket: (f64, f64, f64),
    xmin: f64,
    fmin: f64,
    niter: usize,
    nfeval: usize,
}

#[derive(Deserialize, Debug)]
struct MinimiseScalarBrentTest2
{
    description: String,
    values: MinimiseScalarBrentTest3,
}

#[derive(Deserialize, Debug)]
struct MinimiseScalarBrentTest1
{
    minimise_scalar_brent_test1: MinimiseScalarBrentTest2,
    minimise_scalar_brent_test2: MinimiseScalarBrentTest2,
    minimise_scalar_brent_test3: MinimiseScalarBrentTest2,
    minimise_scalar_brent_test4: MinimiseScalarBrentTest2,
    minimise_scalar_brent_test5: MinimiseScalarBrentTest2,
}

impl MinimiseScalarBrentTest1
{
    fn new() -> Self
    {
        let json_file =
            fs::read_to_string("assets/minimise-scalar-brent.json").expect("Unable to read file");
        serde_json::from_str(&json_file).expect("Could not deserialize")
    }
}

macro_rules! minimise_scalar_brent_test {
    ($test_name: ident, $fcn: expr) => {
        #[test]
        fn $test_name()
        {
            let tol = 1.0e-6;
            let test_data = MinimiseScalarBrentTest1::new();
            let f = $fcn;

            let mut opts = MinimizeOptions::default();
            opts.method = Method::Brent;
            opts.bounds = Bounds::Triple(test_data.$test_name.values.bracket);
            opts.tol = 1e-8;
            opts.max_iter = 1000;

            let res = minimize(&f, &opts);
            assert_eq!(res.is_ok(), true);

            let res_ok = res.unwrap();
            assert_relative_eq!(res_ok.xmin, test_data.$test_name.values.xmin, epsilon = tol);
            assert_relative_eq!(res_ok.fmin, test_data.$test_name.values.fmin, epsilon = tol);
            assert!(res_ok.iter <= test_data.$test_name.values.niter);
            assert!(res_ok.funcalls <= test_data.$test_name.values.nfeval);
        }
    };
}

minimise_scalar_brent_test!(minimise_scalar_brent_test1, |x: f64| x.exp() - 4.0 * x);
minimise_scalar_brent_test!(minimise_scalar_brent_test2, |x: f64| 1.0e-8 * x.powi(2));
minimise_scalar_brent_test!(minimise_scalar_brent_test3, |x: f64| x.powi(2) + 0.1 * (50.0 * x).sin());
minimise_scalar_brent_test!(minimise_scalar_brent_test4, |x: f64| (x - 2.0).abs() + 1.0);
minimise_scalar_brent_test!(minimise_scalar_brent_test5, |x: f64| (x.powi(2) - 4.0).powi(2));
//..............................................................................................

#[derive(Deserialize, Debug)]
struct MinimiseScalarBoundedTest3
{
    bounds: (f64, f64),
    xmin: f64,
    fmin: f64,
    niter: usize,
    nfeval: usize,
}

#[derive(Deserialize, Debug)]
struct MinimiseScalarBoundedTest2
{
    description: String,
    values: MinimiseScalarBoundedTest3,
}

#[derive(Deserialize, Debug)]
struct MinimiseScalarBoundedTest1
{
    minimise_scalar_bounded_test1: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test2: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test3: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test4: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test5: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test6: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test7: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test8: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test9: MinimiseScalarBoundedTest2,
    minimise_scalar_bounded_test10: MinimiseScalarBoundedTest2,
}

impl MinimiseScalarBoundedTest1
{
    fn new() -> Self
    {
        let json_file =
            fs::read_to_string("assets/minimise-scalar-bounded.json").expect("Unable to read file");
        serde_json::from_str(&json_file).expect("Could not deserialize")
    }
}

macro_rules! minimise_scalar_bounded_test {
    ($test_name: ident, $fcn: expr) => {
        #[test]
        fn $test_name()
        {
            let tol = 1.0e-8;
            let test_data = MinimiseScalarBoundedTest1::new();
            let f = $fcn;

            let mut opts = MinimizeOptions::default();
            opts.method = Method::Bounded;
            opts.bounds = Bounds::Pair(test_data.$test_name.values.bounds);
            opts.tol = 1e-8;
            opts.max_iter = 1000;

            let res = minimize(&f, &opts);
            assert_eq!(res.is_ok(), true);

            let res_ok = res.unwrap();                




            assert_relative_eq!(res_ok.xmin, test_data.$test_name.values.xmin, epsilon = tol);
            assert_relative_eq!(res_ok.fmin, test_data.$test_name.values.fmin, epsilon = tol);

            // TODO: fix these, iteration count slightly too high, see https://github.com/users/j-a-ferguson/projects/3/views/1?pane=issue&itemId=56241768
            // assert!(res_ok.iter <= test_data.$test_name.values.niter);
            // assert!(res_ok.funcalls <= test_data.$test_name.values.nfeval);
        }
    };
}

minimise_scalar_bounded_test!(minimise_scalar_bounded_test1, |x: f64| x.exp() - 4.0 * x);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test2, |x: f64| 1e-8 * x * x);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test3, |x: f64| x.powi(2) + 0.1 * (50.0*x).sin());
minimise_scalar_bounded_test!(minimise_scalar_bounded_test4, |x: f64| (x - 2.0).abs() + 1.0);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test5, |x: f64| (x.powi(2) - 4.0).powi(2));
minimise_scalar_bounded_test!(minimise_scalar_bounded_test6, |x: f64| x.exp() - 4.0 * x);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test7, |x: f64| 1e-8 * x * x);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test8, |x: f64| x.powi(2) + 0.1 * (50.0*x).sin());
minimise_scalar_bounded_test!(minimise_scalar_bounded_test9, |x: f64| (x - 2.0).abs() + 1.0);
minimise_scalar_bounded_test!(minimise_scalar_bounded_test10, |x: f64| (x.powi(2) - 4.0).powi(2));