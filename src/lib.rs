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


mod common;
pub mod d1;
pub mod dn;


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