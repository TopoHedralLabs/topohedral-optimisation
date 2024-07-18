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
#![feature(impl_trait_in_assoc_type)]

mod common;
pub use common::{SVector, SMatrix, EvaluateSMatrix, RealFn, FnMutWrap, ZeroFn};
pub mod line_search;
pub mod d1;
pub mod unconstrained_min;



//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use ctor::ctor;
    use topohedral_tracing::*;
    

    #[ctor]
    fn init_logger() {
        init().unwrap();
    }

    #[test]
    fn test_logging() 
    {
        info!("Logging is working!");
    }


}
//}}}