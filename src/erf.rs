//! Wrapper traits for error functions used in calculation of peak integrals
//!
//! We want to try and be generic over the input type in this library, however the `errorfunctions`
//! crate only implements methods on `f64`. The wrapper traits in this module delegate to the
//! methods in `errorfunctions`, automatically casting to and from `f64` on each side of the
//! function calls.
//!
//! In future it would be more idiomatic to re-implement these methods in full, but there is no
//! time to do so now.

use errorfunctions::ComplexErrorFunctions as ExternComplexErrorFunctions;
use errorfunctions::RealErrorFunctions as ExternRealErrorFunctions;
use nalgebra::Complex;
use num_traits::{FromPrimitive, ToPrimitive};

// Error function methods to be applied to real inputs
pub(crate) trait RealErrorFunctions {
    // The error function Erf for real argument
    fn erf(self) -> Self;
}

impl<R: FromPrimitive + ToPrimitive> RealErrorFunctions for R {
    fn erf(self) -> Self {
        let input = self.to_f64().unwrap();
        let result = ExternRealErrorFunctions::erf(input);
        R::from_f64(result).unwrap()
    }
}

// Error function methods to be applied to complex inputs
pub(crate) trait ComplexErrorFunctions {
    // The fadeeva function for complex input
    //
    // The Fadeeva function relates to the complex error function Erfc through
    // w(z) = - e^{-z^2} Erfc(- i z)
    fn fadeeva(self) -> Self;
}

impl<R: FromPrimitive + ToPrimitive> ComplexErrorFunctions for Complex<R> {
    fn fadeeva(self) -> Self {
        let input: Complex<f64> =
            Complex::new(self.re.to_f64().unwrap(), self.im.to_f64().unwrap());
        let result = ExternComplexErrorFunctions::w(input);
        Complex::new(
            R::from_f64(result.re).unwrap(),
            R::from_f64(result.im).unwrap(),
        )
    }
}
