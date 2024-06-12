//! Cauchy Probability Density functions
//!
//! A Cauchy probability density function is characterised by a central value and a scale factor.
//! The PDF is given by
//! 1/pi (scale / ((x - central)^2 + scale^2)
//! In this format the scale represents the half width at half maximum of the distribution.

use super::ProbabilityDensityFunction;
use nalgebra::RealField;

#[derive(Clone, Debug, PartialEq)]
pub struct CauchyPDF<R> {
    central: R,
    scale: R,
}

impl<R> CauchyPDF<R>
where
    R: Copy + RealField,
{
    /// Create a new [`CauchyPDF`] from its central value and half width at half maximum
    pub fn new(central: R, scale: R) -> Self {
        Self { central, scale }
    }

    fn pi(&self) -> R {
        R::from_f64(std::f64::consts::PI).unwrap()
    }

    fn two(&self) -> R {
        R::one() + R::one()
    }
}

impl<R> ProbabilityDensityFunction for CauchyPDF<R>
where
    R: Copy + RealField,
{
    type Float = R;

    fn central_value(&self) -> R {
        self.central
    }

    fn half_width_half_maximum(&self) -> R {
        self.scale
    }

    fn probability_density_function(&self, at: R) -> R {
        R::one() / self.pi() / self.scale / (R::one() + ((at - self.central) / self.scale).powi(2))
    }

    fn cumulative_distribution_function(&self, at: R) -> R {
        R::one() / self.two() + (at - self.central).atan2(self.scale) / self.pi()
    }

    fn overlap_with(&self, other: &Self) -> R {
        let sum_of_scale = self.scale + other.scale;
        let difference_of_central = self.central - other.central;
        sum_of_scale / self.pi() / (sum_of_scale.powi(2) + difference_of_central.powi(2))
    }
}

#[cfg(test)]
mod test {
    use nalgebra::RealField;
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use super::{CauchyPDF, ProbabilityDensityFunction};

    impl<R> CauchyPDF<R>
    where
        R: Copy + RealField,
    {
        // A standard Cauchy distribution is centered on zero and
        fn standard() -> Self {
            Self {
                central: R::zero(),
                scale: R::one(),
            }
        }
    }

    #[test]
    fn standard_cauchy_at_zero() {
        let dist = CauchyPDF::standard();
        assert_eq!(
            1.0 / std::f64::consts::PI,
            dist.probability_density_function(0.0)
        );
    }

    #[test]
    fn standard_cauchy_integral_product() {
        let dist: CauchyPDF<f64> = CauchyPDF::standard();
        assert_eq!(1.0 / 2.0 / std::f64::consts::PI, dist.overlap_with(&dist));
    }

    #[test]
    fn cauchy_at_zero() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central: f64 = rng.gen();
        let scale = rng.gen();
        let dist = CauchyPDF::new(central, scale);

        assert_eq!(
            1.0 / std::f64::consts::PI / scale,
            dist.probability_density_function(central)
        );
    }

    #[test]
    fn cauchy_integral_product() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central: f64 = rng.gen();
        let scale = rng.gen();
        let dist = CauchyPDF::new(central, scale);

        assert_eq!(
            1.0 / 2.0 / std::f64::consts::PI / scale,
            dist.overlap_with(&dist)
        );
    }

    #[test]
    fn different_cauchy_integral_product() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let first_central: f64 = rng.gen();
        let first_scale = rng.gen();
        let first_dist = CauchyPDF::new(first_central, first_scale);

        let second_central: f64 = rng.gen();
        let second_scale = rng.gen();
        let second_dist = CauchyPDF::new(second_central, second_scale);

        assert_eq!(
            (first_scale + second_scale)
                / ((first_scale + second_scale).powi(2) + (first_central - second_central).powi(2))
                / std::f64::consts::PI,
            first_dist.overlap_with(&second_dist)
        );
    }

    #[test]
    fn cumulative_distribution_function_at_large_positive_input_yields_unity() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central = rng.gen();
        let scale: f64 = rng.gen();
        let dist = CauchyPDF::new(central, scale);
        assert_eq!(1.0, dist.cumulative_distribution_function(std::f64::MAX));
    }

    #[test]
    fn cumulative_distribution_function_at_large_negative_input_yields_zero() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central = rng.gen();
        let scale: f64 = rng.gen();
        let dist = CauchyPDF::new(central, scale);
        assert_eq!(0.0, dist.cumulative_distribution_function(-std::f64::MAX));
    }
}
