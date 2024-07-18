//! This module contains the implementation of the scalar optimisation algorithms.  
//!
//!  Scalar optimisation algorithms included are:
//! - Brent's method
//! - Bounded method
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
use core::{error, fmt};
use std::fmt::format;
//}}}
//{{{ dep imports 
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------


//{{{ const:  GOLDERN_RATIO
const GOLDERN_RATIO: f64 = 1.6180339887498948482;

//}}}
//{{{ enum:   Method
/// Enum representing the set of supported minimization methods.
#[derive(Debug, Clone)]
pub enum Method
{
    Brent,
    Bounded,
}

impl Default for Method
{
    fn default() -> Self
    {
        Method::Brent
    }
}

//}}}
//{{{ enum:   Bounds
/// Enum representing the set of supported bounds.
///
/// - ``None``: Means that the user wishes for the bracket to be computed completely from scratch.
/// - ``Pair``: Means that the user wishes for the bracket to be computed from an initial LB and UB
/// - ``Triple``: Means that the user has provided the bracket bounds.
#[derive(Copy, Clone, Debug)]
pub enum Bounds
{
    None,
    Pair((f64, f64)),
    Triple((f64, f64, f64)),
}

impl Default for Bounds
{
    fn default() -> Self
    {
        Bounds::None
    }
}

impl fmt::Display for Bounds
{
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result
    {
        match self
        {
            Bounds::None => write!(f, "None"),
            Bounds::Pair((a, b)) => write!(f, "Pair({}, {})", a, b),
            Bounds::Triple((a, b, c)) => write!(f, "Triple({}, {}, {})", a, b, c),
        }
    }
}
//}}}
//{{{ enum:   ScalarError
/// Enum representing the set of possible scalar minimization errors.
/// - ``MaxIterReached``: Means that the maximum number of iterations has been reached.
/// - ``UnknownError``: Means that the minimization algorithm encountered an unknown error.
#[derive(Error, Debug)]
pub enum Error
{
    #[error("Max iterations of {max_iter} reached with x = {x} fx = {fx} num_funcalls = {num_funcalls}")]
    MaxIterReached{max_iter: usize, x: f64, fx: f64, num_funcalls: usize},
    #[error("BracketNotFound:")]
    BracketNotFound((usize, bool, bool, bool)),
    #[error("BadOptions: {0}")]
    BadOptions(String),
    #[error("NaN encountered {0}")]
    NanEncountered(String),
    #[error("Inf encountered {0}")] 
    InfEncountered(String),
    #[error("UnknownError")]
    UnknownError,
}

//}}}
//{{{ struct: BracketOptions 
pub struct BracketOptions
{
    pub a: f64,
    pub b: f64,
    pub growth_limit: f64,
    pub maxiter: usize,
}

impl Default for BracketOptions
{
    fn default() -> Self
    {
        Self {
            a: 0.0,
            b: 1.0,
            growth_limit: 110.0,
            maxiter: 1000,
        }
    }
}
//}}}
//{{{ fun:    brack_conds
fn brack_conds<F: FnMut(f64) -> f64>(
    a: f64,
    b: f64,
    c: f64,
    mut f: F,
) -> (bool, bool, bool)
{
    let fa = f(a);
    let fb = f(b);
    let fc = f(c);
    let cond1 = (fb < fa && fb <= fc) || (fb <= fa && fb < fc);
    let cond2 = (a < b && b < c) || (c < b && b < a);
    let cond3 = a.is_finite()
        && b.is_finite()
        && c.is_finite()
        && fa.is_finite()
        && fb.is_finite()
        && fc.is_finite();
    (cond1, cond2, cond3)
}

//}}}
//{{{ fun:    bracket
/// Bracketing algorithm for scalar function
///
/// Given a function with a local minumum, this algorithm finds a bracketing interval
/// consisting of three distinct points $\{a, b, c\}$ where $a < b < c$ such that:
/// $$
/// f(a) > f(b) \quad \text{and} \quad f(c) > f(b)
/// $$
///
/// # Arguments
/// - `f`: The function to be bracketed.
/// - `opts`: The options for the bracketing algorithm.
///
/// # Returns
/// - `Ok((a, b, c, fa, fb, fc, funcalls))`: If the bracket was found.
/// - `Err(ScalarError::BracketNotFound)`: If the bracket was not found.
pub fn bracket<F: FnMut(f64) -> f64>(
    mut f: F,
    opts: &BracketOptions,
) -> Result<(f64, f64, f64, f64, f64, f64, usize), Error>
{
    let g = GOLDERN_RATIO;
    let very_small_num = 1e-21;
    let mut funcalls = 0;
    let mut a = opts.a;
    let mut b = opts.b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa < fb
    {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = b + g * (b - a);
    let mut fc = f(c);

    funcalls += 3;
    let mut iter = 0;

    while (fc < fb) && (iter < opts.maxiter)
    {
        let tmp1 = (b - a) * (fb - fc);
        let tmp2 = (b - c) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < very_small_num
        {
            2.0 * very_small_num
        }
        else
        {
            2.0 * val
        };

        let mut w = b - ((b - c) * tmp2 - (b - a) * tmp1) / denom;
        let wlim = b + opts.growth_limit * (c - b);
        let mut fw = 0.0;

        if (w - c) * (b - w) > 0.0
        {
            fw = f(w);
            funcalls += 1;

            if fw < fc
            {
                a = b;
                b = w;
                fa = fb;
                fb = fw;
                break;
            }
            else if fw > fb
            {
                c = w;
                fc = fw;
                break;
            }

            w = c + g * (c - b);
            fw = f(w);
            funcalls += 1;
        }
        else if (w - wlim) * (wlim - c) >= 0.0
        {
            w = wlim;
            fw = f(w);
            funcalls += 1;
        }
        else if (w - wlim) * (c - w) > 0.0
        {
            fw = f(w);
            funcalls += 1;
            if fw < fc
            {
                b = c;
                c = w;
                w = c + g * (c - b);
                fb = fc;
                fc = fw;
                fw = f(w);
                funcalls += 1;
            }
        }
        else
        {
            w = c + g * (c - b);
            fw = f(w);
            funcalls += 1;
        }

        a = b;
        b = c;
        c = w;
        fa = fb;
        fb = fc;
        fc = fw;
        iter += 1;
    }

    let (cond1, cond2, cond3) = brack_conds(a, b, c, f);

    let out = if cond1 && cond2 && cond3
    {
        Ok((a, b, c, fa, fb, fc, funcalls))
    }
    else
    {
        Err(Error::BracketNotFound {
            0: (iter, cond1, cond2, cond3),
        })
    };
    out
}

//}}}
//{{{ struct: MinimizeOptions
#[derive(Debug, Clone)]
pub struct MinimizeOptions
{
    pub method: Method,
    pub bounds: Bounds,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for MinimizeOptions
{
    fn default() -> Self
    {
        Self {
            method: Method::default(),
            bounds: Bounds::default(),
            tol: 1e-5,
            max_iter: 1000,
        }
    }
}


//}}}
//{{{ struct: MinimizeReturns
#[derive(Default, Debug)]
pub struct MinimizeReturns
{
    pub xmin: f64,
    pub fmin: f64,
    pub iter: usize,
    pub funcalls: usize,
}
//}}}
//{{{ struct: Brent
struct Brent<F: FnMut(f64) -> f64>
{
    // input members
    f: F,
    tol: f64,
    max_iter: usize,
    bounds: Bounds,
    // internal members
    _mintol: f64,
    // output values
    xmin: f64,
    fmin: f64,
    iter: usize,
    funcalls: usize,
}

impl<F: FnMut(f64) -> f64> Brent<F>
{
    fn new(
        f: F,
        opts: &MinimizeOptions,
    ) -> Self
    {
        Self {
            f,
            tol: opts.tol,
            max_iter: opts.max_iter,
            bounds: opts.bounds,
            _mintol: 1.0e-11,
            xmin: 0.0,
            fmin: 0.0,
            iter: 0,
            funcalls: 0,
        }
    }

    fn get_bracket_bounds(&mut self) -> Result<(f64, f64, f64, f64, f64, f64, usize), Error>
    {
        let out = match self.bounds
        {
            Bounds::None =>
            {
                let opts = BracketOptions::default();
                bracket(&mut self.f, &opts)
            }
            Bounds::Pair((a, b)) =>
            {
                let opts = BracketOptions {
                    a,
                    b,
                    ..Default::default()
                };

                bracket(&mut self.f, &opts)
            }
            Bounds::Triple((a, b, c)) =>
            {
                let (cond1, cond2, cond3) = brack_conds(a, b, c, &mut self.f);
                if cond1 && cond2 && cond3
                {
                    let fa = (self.f)(a);
                    let fb = (self.f)(b);
                    let fc = (self.f)(c);
                    Ok((a, b, c, fa, fb, fc, 3))
                }
                else
                {
                    Err(Error::BracketNotFound((0, cond1, cond2, cond2)))
                }
            }
        };
        out
    }

    fn optimize(&mut self) -> Result<(), Error>
    {
        // find the bracket in which the minimum occurs, or return an error
        let (xa, xb, xc, _fa, fb, _fc, mut funcalls) = self.get_bracket_bounds()?;
        // initialise local data
        let (mut x, mut w, mut v) = (xb, xb, xb);
        let (mut fw, mut fv, mut fx) = (fb, fb, fb);
        let mut a = xa.min(xc);
        let mut b = xa.max(xc);
        let mut deltax = 0.0f64;
        let mut iter = 0;
        let cg = 0.3819660;
        let mut u = 0.0;
        let mut rat = 0.0;
        // begin the main algorithm
        while iter < self.max_iter
        {
            // set tolerances for iteration
            let tol1 = self._mintol + (self.tol * x.abs());
            let tol2 = 2.0 * tol1;
            // find mid point
            let xmid = 0.5 * (a + b);
            // test for convergence
            if (x - xmid).abs() < (tol2 - 0.5 * (b - a))
            {
                break;
            }

            // main part of loop
            if deltax.abs() <= tol1
            {
                //............... do golden section step
                if x >= xmid
                {
                    deltax = a - x;
                }
                else
                {
                    deltax = b - x;
                }
                rat = cg * deltax;
            }
            else
            {
                //............... do parabolic step

                let tmp1 = (x - w) * (fx - fv);
                let mut tmp2 = (x - v) * (fx - fw);
                let mut p = (x - v) * tmp2 - (x - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                if tmp2 > 0.0
                {
                    p = -p;
                }
                tmp2 = tmp2.abs();
                let dx_temp = deltax;
                deltax = rat;
                // check parabolic fit
                if (p > tmp2 * (a - x))
                    && (p < tmp2 * (b - x))
                    && (p.abs() < (0.5 * tmp2 * dx_temp).abs())
                // if parabolic fit useful, do it
                {
                    rat = p * (1.0 / tmp2);
                    u = x + rat;
                    if ((u - a) < tol2) || ((b - u) < tol2)
                    {
                        if xmid - x >= 0.0
                        {
                            rat = tol1;
                        }
                        else
                        {
                            rat = -tol1;
                        }
                    }
                }
                else
                // if not, do golden section
                {
                    if x >= xmid
                    {
                        deltax = a - x;
                    }
                    else
                    {
                        deltax = b - x;
                    }
                    rat = cg * deltax;
                }
            }

            // update by at least tol1
            if rat.abs() < tol1
            {
                if rat >= 0.0
                {
                    u = x + tol1;
                }
                else
                {
                    u = x - tol1;
                }
            }
            else
            {
                u = x + rat;
            }

            // evaluate function at u
            let fu = (self.f)(u);
            funcalls += 1;

            if fu > fx
            {
                if u < x
                {
                    a = u;
                }
                else
                {
                    b = u;
                }

                if (fu <= fw) || (w == x)
                {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                }
                else if (fu <= fv) || (v == x) || (v == w)
                {
                    v = u;
                    fv = fu;
                }
            }
            else
            {
                if u >= x
                {
                    a = x;
                }
                else
                {
                    b = x;
                }

                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            }

            iter += 1;
        }

        self.xmin = x;
        self.fmin = fx;
        self.iter = iter;
        self.funcalls = funcalls;

        let out = if iter == self.max_iter
        {
            Err(Error::MaxIterReached{max_iter: self.max_iter, x: x, fx: (self.f)(x), num_funcalls: funcalls})
        }
        else
        {
            Ok(())
        };
        out
    }
}

//}}}
//{{{ struct: Bounded
struct Bounded<F: FnMut(f64) -> f64>
{
    // input members
    f: F,
    tol: f64,
    max_iter: usize,
    bounds: (f64, f64),
    // output values
    xmin: f64,
    fmin: f64,
    iter: usize,
    funcalls: usize,
}

impl<F: FnMut(f64) -> f64> Bounded<F>
{
    fn new(
        f: F,
        opts: &MinimizeOptions,
    ) -> Result<Self, Error>
    {
        let out = if let Bounds::Pair(bounds) = opts.bounds
        {
            Ok(Self {
                f,
                tol: opts.tol,
                max_iter: opts.max_iter,
                bounds: bounds,
                xmin: 0.0,
                fmin: 0.0,
                iter: 0,
                funcalls: 0,
            })
        }
        else
        {
            Err(Error::BadOptions(format!(
                "Invalid bounds {}",
                opts.bounds
            )))
        };
        out
    }

    fn optimize(
        &mut self        
    ) -> Result<(), Error>
    {
        let (mut x1, mut x2) = self.bounds;

        let sqrt_eps = f64::sqrt(f64::EPSILON);
        let golden_mean = 0.5 * (3.0 - f64::sqrt(5.0));
        let (mut a, mut b) = (x1, x2);
        let mut fulc = a + golden_mean * (b - a);
        let (mut nfc, mut xf) = (fulc, fulc);
        let (mut rat, mut e) = (0.0f64, 0.0f64);
        let mut x = xf;
        let mut fx = (self.f)(x);        
        let mut funcalls = 1;
        let mut fu = f64::INFINITY;

        let (mut ffulc, mut fnfc) = (fx, fx);
        let mut xm = 0.5 * (a + b);
        let mut tol1 = sqrt_eps * xf.abs() + 0.333333 * self.tol;
        let mut tol2 = 2.0 * tol1;

        let (mut r, mut q, mut p) = (0.0, 0.0, 0.0);
        let mut num_iter = 0;
        let mut reached_max_iter = false;

        while ((xf - xm).abs() > (tol2 - 0.5 * (b - a)))
        {
            let mut golden = true;
            // check for parabolic fit
            if e.abs() > tol1
            {
                golden = false;
                r = (xf - nfc) * (fx - ffulc);
                q = (xf - fulc) * (fx - fnfc);
                p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if q > 0.0
                {
                    p = -p;
                }
                q = q.abs();
                r = e;
                e = rat;

                // check for acceptability of the parabola
                if (p.abs() < (0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))
                {
                    rat = (p + 0.0) / q;
                    x = xf + rat;

                    if ((x - a) < tol2) || ((b - x) < tol2)
                    {
                        let si = sign(xm - xf) + (((xm == xf) as i32) as f64);
                        rat = tol1 * si;
                    }
                }
                else
                {
                    golden = true;
                }
            }

            if golden
            {
                if xf >= xm
                {
                    e = a - xf;
                }
                else
                {
                    e = b - xf;
                }
                rat = golden_mean * e;
            }

            let si = sign(rat) + (((rat == 0.0) as i32) as f64);
            x = xf + si * rat.abs().max(tol1);
            fu = (self.f)(x);
            funcalls += 1;
            num_iter += 1;

            if fu <= fx
            {
                if x >= xf
                {
                    a = xf;
                }
                else
                {
                    b = xf;
                }
                fulc = nfc;
                ffulc = fnfc;
                nfc = xf;
                fnfc = fx;
                xf = x;
                fx = fu;
            }
            else
            {
                if x < xf
                {
                    a = x;
                }
                else
                {
                    b = x;
                }

                if (fu <= fnfc) || (nfc == xf)
                {
                    fulc = nfc;
                    ffulc = fnfc;
                    nfc = x;
                    fnfc = fu;
                }
                else if (fu <= ffulc) || (fulc == xf) || (fulc == nfc)
                {
                    fulc = x;
                    ffulc = fu;
                }
            }

            if num_iter >= self.max_iter
            {
                reached_max_iter = true;
                break;
            }

            xm = 0.5 * (a + b);
            tol1 = sqrt_eps * xf.abs() + 0.333333 * self.tol;
            tol2 = 2.0 * tol1;

        }

        let out = if xf.is_nan() || fx.is_nan() || fu.is_nan()
        {
            Err(Error::NanEncountered(format!(
                "xf = {} fs = {} fu = {}",
                xf, fx, fu
            )))
        }
        else if xf.is_infinite() || fx.is_infinite() || fu.is_infinite()
        {
            Err(Error::InfEncountered(format!(
                "xf = {} fs = {} fu = {}",
                xf, fx, fu
            )))
        }
        else if reached_max_iter
        {
            Err(Error::MaxIterReached{max_iter: self.max_iter, x: xf, fx: fx, num_funcalls: funcalls}) 
        }
        else
        {
            self.xmin = xf;
            self.fmin = fx;
            self.iter = num_iter;
            self.funcalls = funcalls;
            Ok(())
        };
        out
    }
}
//}}}
//{{{ fun:    sign
fn sign(x: f64) -> f64
{
    if x > 0.0
    {
        1.0
    }
    else if x < 0.0
    {
        -1.0
    }
    else
    {
        0.0
    }
}

//}}}
//{{{ fun:    minimize
pub fn minimize<F: FnMut(f64) -> f64>(
    f: F,
    opts: &MinimizeOptions,
) -> Result<MinimizeReturns, Error>
{
    let mut out = MinimizeReturns::default();

    match opts.method
    {
        Method::Brent =>
        {
            let mut brent = Brent::new(f, opts);
            brent.optimize()?;
            out.xmin = brent.xmin;
            out.fmin = brent.fmin;
            out.iter = brent.iter;
            out.funcalls = brent.funcalls;
        }
        Method::Bounded =>
        {
            let mut bounded = Bounded::new(f, opts)?;
            bounded.optimize()?;
            out.xmin = bounded.xmin;
            out.fmin = bounded.fmin;
            out.iter = bounded.iter;
            out.funcalls = bounded.funcalls;

        }
    };
    Ok(out)
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
  
    use super::*;
    use approx::assert_relative_eq;
    use serde::Deserialize;
    use std::fs;
    //..............................................................................................

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
}
//}}}
