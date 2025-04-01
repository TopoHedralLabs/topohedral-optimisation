//! # Topoohedral-Optimisation
//!
//! This crate provides optimisation algorithms for finding the minimum of a function.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]

mod common;
pub use common::{
    EvaluateSMatrix, FloatVectorOps, FnMutWrap, RealFn, SMatrix, SVector, VectorOps, ZeroFn,
};
pub mod d1;
pub mod line_search;
pub mod unconstrained_min;

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use ctor::ctor;
    use topohedral_tracing::*;

    #[ctor]
    fn init_logger() {
        init().unwrap();
    }

    #[test]
    fn test_logging() {
        info!("Logging is working!");
    }
}
//}}}
