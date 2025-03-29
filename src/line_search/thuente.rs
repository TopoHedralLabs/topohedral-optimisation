//! This module implements the More-Thuente line search algorithm.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use super::{LineSearchOpts, LineSearch, LineSearchFn, LineSearchReturns};
use crate::common::{RealFn, SVector, SMatrix};
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ThuenteOpts {
    pub ls_opts: LineSearchOpts,
    pub initial_step_size: f64,
    pub min_step_size: f64,
    pub max_step_size: f64,
    pub max_iter: usize,
    phi_tol: f64,
    dphi_tol: f64,
    alpha_tol: f64,
    alpha_min: f64,
    alpha_max: f64,
}

pub struct ThuenteLineSearch <const N: usize, F: RealFn<N>>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    pub(crate) f: LineSearchFn<N, F>,
    pub opts: ThuenteOpts,

    stage:  usize,
    phi_init: f64, 
    dphi_init: f64,
    dphi_test: f64,

    alpha_a: f64,
    alpha_b: f64,
    phi_a: f64,

    phi_b: f64,
    dphi_a: f64,
    dphi_b: f64,

    phi: f64,
    dphi: f64,

    brackt: bool,
    width: f64,
    width1: f64,
    stmin: f64,
    stmax: f64,

}

impl<const N: usize, F: RealFn<N>>  ThuenteLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
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
            alpha_b: 0.0,
            phi_a: 0.0,
            phi_b: 0.0,
            dphi_a: 0.0,
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

        let p5 = 0.5;
        let p66 = 0.66;
        let xtrapl = 1.1;
        let xtrapu = 4.0;

        self.brackt = false;
        self.stage = 1;
        self.phi_init = phi_alpha;
        self.dphi_init = dphi_alpha;
        self.dphi_test = self.opts.phi_tol * self.dphi_init;
        self.width = self.opts.max_step_size - self.opts.min_step_size;
        self.width1 = self.width * p5; 

    }

    fn iterate(&mut self) {

    }
}

impl<const N: usize, F: RealFn<N>> LineSearch<N> for ThuenteLineSearch<N, F>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    fn line_search(&mut self, phi0: f64, dphi0: f64) -> Result<LineSearchReturns, super::Error> {

        self.initialise(0.0, phi0, dphi0);

        for i in 0..self.opts.max_iter {
            self.iterate();
        }

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


