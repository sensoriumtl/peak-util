//! A NormalPDF distribution
//!
//! The NormalPDF distribution is defined as a probability density function, whose integral over the
//! real line yields unity. The functional form of the distribution is given by
//! $$
//!     N\left(\mu; \sigma\right) = \frac{\exp\left[-\frac{1}{2} \left(\frac{x -
//!     \mu}{\sigma}\right)^2\right] }{\sigma \sqrt{2 \pi}}
//! $$
//! where $\mu$ is the mean of the distribution and $\sigma$ is the standard deviation.
//!

use super::ProbabilityDensityFunction;
use crate::RealErrorFunctions;
use nalgebra::RealField;
use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Clone, Debug, PartialEq)]
/// A normal probability density function
pub struct NormalPDF<R> {
    /// The central value of the distribution \mu
    mean: R,
    /// The standard deviation of the distribution \sigma
    std_dev: R,
}

impl<R> NormalPDF<R>
where
    R: Copy + RealField,
{
    // Generate a new `NormalPDF` from known mean and standard deviation
    pub fn new(mean: R, std_dev: R) -> Self {
        Self { mean, std_dev }
    }

    fn pi(&self) -> R {
        R::from_f64(std::f64::consts::PI).unwrap()
    }

    fn two(&self) -> R {
        R::one() + R::one()
    }
}

impl<R> ProbabilityDensityFunction for NormalPDF<R>
where
    R: Copy + RealField + ToPrimitive + FromPrimitive,
{
    type Float = R;
    fn central_value(&self) -> R {
        self.mean
    }

    fn half_width_half_maximum(&self) -> R {
        (self.two() * self.two().ln()).sqrt() * self.std_dev
    }

    fn probability_density_function(&self, at: R) -> R {
        (-((self.mean - at) / self.std_dev).powi(2) / self.two()).exp()
            / (self.two() * self.pi()).sqrt()
            / self.std_dev
    }

    fn cumulative_distribution_function(&self, at: R) -> R {
        let arg = (at - self.mean) / self.std_dev / self.two().sqrt();
        R::one() / self.two() * (R::one() + arg.erf())
    }

    fn overlap_with(&self, other: &Self) -> R {
        let sum_of_std_dev_squares = self.std_dev.powi(2) + other.std_dev.powi(2);
        let mean_difference = self.mean - other.mean;
        (-mean_difference.powi(2) / self.two() / sum_of_std_dev_squares).exp()
            / (self.two() * self.pi()).sqrt()
            / sum_of_std_dev_squares.sqrt()
    }
}

#[cfg(test)]
mod test {
    use nalgebra::RealField;
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use super::{NormalPDF, ProbabilityDensityFunction};

    impl<R> NormalPDF<R>
    where
        R: Copy + RealField,
    {
        // A standard Cauchy distribution is centered on zero and
        fn standard() -> Self {
            Self {
                mean: R::zero(),
                std_dev: R::one(),
            }
        }
    }

    #[test]
    fn standard_normal_at_0() {
        let dist = NormalPDF::standard();
        assert_eq!(
            1.0 / (2.0 * std::f64::consts::PI).sqrt(),
            dist.probability_density_function(0.0)
        );
    }

    #[test]
    fn standard_normal_integral_product_over_real_line() {
        let dist = NormalPDF::standard();
        assert_eq!(
            1.0 / 2.0 / std::f64::consts::PI.sqrt(),
            dist.overlap_with(&dist)
        );
    }

    #[test]
    fn normal_at_mean() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mean: f64 = rng.gen();
        let std_dev = rng.gen();
        let dist = NormalPDF::new(mean, std_dev);

        assert_eq!(
            1.0 / (2.0 * std::f64::consts::PI).sqrt() / std_dev,
            dist.probability_density_function(mean)
        );
    }

    #[test]
    fn normal_integral_product_over_real_line() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mean = rng.gen();
        let std_dev: f64 = rng.gen();
        let dist = NormalPDF::new(mean, std_dev);

        assert_eq!(
            1.0 / 2.0 / std::f64::consts::PI.sqrt() / std_dev,
            dist.overlap_with(&dist)
        );
    }

    #[test]
    fn different_normal_integral_product_over_real_line() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let first_mean = rng.gen();
        let first_std_dev: f64 = rng.gen();
        let first_dist = NormalPDF::new(first_mean, first_std_dev);

        let second_mean = rng.gen();
        let second_std_dev: f64 = rng.gen();
        let second_dist = NormalPDF::new(second_mean, second_std_dev);

        assert_eq!(
            (-(first_mean - second_mean).powi(2)
                / 2.0
                / (first_std_dev.powi(2) + second_std_dev.powi(2)))
            .exp()
                / (2.0 * std::f64::consts::PI).sqrt()
                / (first_std_dev.powi(2) + second_std_dev.powi(2)).sqrt(),
            first_dist.overlap_with(&second_dist)
        );
    }

    #[test]
    fn cumulative_distribution_function_at_large_positive_input_yields_unity() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mean = rng.gen();
        let std_dev: f64 = rng.gen();
        let dist = NormalPDF::new(mean, std_dev);
        assert_eq!(1.0, dist.cumulative_distribution_function(std::f64::MAX));
    }

    #[test]
    fn cumulative_distribution_function_at_large_negative_input_yields_zero() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mean = rng.gen();
        let std_dev: f64 = rng.gen();
        let dist = NormalPDF::new(mean, std_dev);
        assert_eq!(0.0, dist.cumulative_distribution_function(-std::f64::MAX));
    }
}
