//! Probabiliy density funcionts are the underlying abstraction in `peak-util`

/// Cauchy probability distributions
mod cauchy;
/// Normal probability distributions
mod normal;
/// Voigt probability distributions
mod voigt;

pub use cauchy::CauchyPDF;
pub use normal::NormalPDF;
pub use voigt::CenteredVoigtPDF;

use nalgebra::RealField;
use num_traits::One;

/// All PDFs implement [`ProbabilityDensityFunction`]
pub trait ProbabilityDensityFunction: PartialEq {
    /// The underlying numerical type of the distribution
    type Float: RealField + One;
    fn central_value(&self) -> Self::Float;
    fn half_width_half_maximum(&self) -> Self::Float;
    /// The probability density function evaluated at the requested value
    fn probability_density_function(&self, at: Self::Float) -> Self::Float;
    /// The cumulative distribution function evaluated at the requested value.
    ///
    /// This represents the integral of the probability distribution function from negative
    /// infinity to `at`
    fn cumulative_distribution_function(&self, at: Self::Float) -> Self::Float;
    /// The q-function is just unity, minus the cumulative_distribution_function
    fn q_function(&self, at: Self::Float) -> Self::Float {
        <Self::Float as One>::one() - self.cumulative_distribution_function(at)
    }
    /// Integral of the product of this [`ProbabilityDensityFunction`] and the other
    /// [`ProbabilityDensityFunction`].
    ///
    /// This function is delegated to when calculating overlap integrals between PDFs of the same
    /// type.
    fn overlap_with(&self, other: &Self) -> Self::Float;
}
