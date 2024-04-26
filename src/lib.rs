//! `peak-util` provides primitives for constructing signals as a sum of probability density
//! functions.
//!
//!

// Implementation of functions related to the error function Erf for generic argument
mod erf;

pub(crate) use erf::{ComplexErrorFunctions, RealErrorFunctions};

// Specific probabiliy density functions
mod pdf;

pub(crate) use pdf::ProbabilityDensityFunction;
pub use pdf::{CauchyPDF, CenteredVoigtPDF, NormalPDF};

// Overlap between probability density functions
mod overlap;

pub use overlap::Overlap;
pub(crate) use overlap::{ProductOfPDF, ProductOverlap};

// Abstractions for distributions formed as a sum of individual PDFs
mod multi;

pub use multi::Spectrum;

// Single peaks formed from an underlying PDF
mod one;

pub use one::PeakInSpectrum;
