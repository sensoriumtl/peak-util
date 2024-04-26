//! CenteredVoigt distributions
//!

use super::ProbabilityDensityFunction;
use crate::ComplexErrorFunctions;
use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{FromPrimitive, ToPrimitive};
use quad_rs::{
    AccumulateError, AdaptiveIntegrator, GenerateBuilder, Integrable, IntegrableFloat,
    IntegrationOutput, RescaleError,
};

#[derive(Clone, Debug, PartialEq)]
/// A centered Voigt probability function is the convolution of a Normal and Cauchy distribution,
/// in the specific case where both underlying distributions have the same mean. It is
/// characterised by a `central` value, a standard deviation corresponding to the normal
/// distribution and a `scale` which corresponds to the half-width at half maximum of the Cauchy
/// component
pub struct CenteredVoigtPDF<R> {
    /// Central value of the underlying Normal and Cauchy distributions
    central: R,
    /// Standard deviation of the underlying Normal distribution
    std_dev: R,
    /// Half width at half maximum of the underlying Cauchy distribution
    scale: R,
}

impl<R> CenteredVoigtPDF<R>
where
    R: Copy + RealField,
{
    pub fn new(central: R, std_dev: R, scale: R) -> Self {
        Self {
            central,
            std_dev,
            scale,
        }
    }

    fn pi(&self) -> R {
        R::from_f64(std::f64::consts::PI).unwrap()
    }

    fn two(&self) -> R {
        R::one() + R::one()
    }
}

impl<R> ProbabilityDensityFunction for CenteredVoigtPDF<R>
where
    R: Copy
        + RealField
        + ToPrimitive
        + FromPrimitive
        + AccumulateError<R>
        + IntegrableFloat
        + RescaleError,
    Complex<R>: ComplexField<RealField = R>
        + Copy
        + ToPrimitive
        + IntegrationOutput<Float = R, Scalar = Complex<R>, Real = R>,
{
    type Float = R;

    fn central_value(&self) -> R {
        self.central
    }

    // The half-width half-maximum of the distribution is numerically approximated using the
    // approach proposed by Kielkopf (doi:10.1364/JOSA.63.000987). This method is accurate to
    // within 0.2%.
    fn half_width_half_maximum(&self) -> R {
        let hwhm_cauchy = self.scale;
        let hwhm_normal =
            ComplexField::sqrt(self.two() * ComplexField::ln(self.two())) * self.std_dev;

        R::from_f64(0.5346).unwrap() * hwhm_cauchy
            + ComplexField::sqrt(
                R::from_f64(0.2166).unwrap() * ComplexField::powi(hwhm_cauchy, 2)
                    + ComplexField::powi(hwhm_normal, 2),
            )
    }

    fn probability_density_function(&self, at: R) -> R {
        let arg = Complex::new(at - self.central, self.scale)
            / Complex::new(self.std_dev * ComplexField::sqrt(self.two()), R::zero());
        arg.fadeeva().re / self.std_dev / ComplexField::sqrt(self.two() * self.pi())
    }

    fn cumulative_distribution_function(&self, at: R) -> R {
        let cdf = CenteredVoigtCDF { distribution: self };
        cdf.integrate_to(at)
    }

    fn overlap_with(&self, other: &Self) -> R {
        let product = VoigtPDFProduct {
            distribution: self,
            other,
        };
        product.integrate()
    }
}

/// A centered Voigt cumulative distribution function.
///
/// This is defined seperately as a wrapper to the probability density function because we need to
/// define the `Integrable` seperately for each structure.
struct CenteredVoigtCDF<'a, R> {
    /// The underlying probability density function
    distribution: &'a CenteredVoigtPDF<R>,
}

impl<'a, R> CenteredVoigtCDF<'a, R>
where
    R: Copy + RealField + ToPrimitive + AccumulateError<R> + IntegrableFloat + RescaleError,
    Complex<R>: ComplexField<RealField = R>
        + Copy
        + ToPrimitive
        + IntegrationOutput<Float = R, Scalar = Complex<R>, Real = R>,
{
    fn lower_cutoff(&self, num_hwhm: usize) -> R {
        let num_hwhm = R::from_usize(num_hwhm).unwrap();
        self.distribution.central - num_hwhm * self.distribution.half_width_half_maximum()
    }

    fn upper_cutoff(&self, num_hwhm: usize) -> R {
        let num_hwhm = R::from_usize(num_hwhm).unwrap();
        self.distribution.central + num_hwhm * self.distribution.half_width_half_maximum()
    }

    fn integrate_to(self, to: R) -> R {
        // How many hwhm on each side of the `central` the integral should extend.
        // This should be robust to all distributional properties, as the range will contract with
        // the distribution. We don't want to force callers to pass a limit, as that will break the
        // equality between calls on this distribution and the simpler ones.
        let num_hwhm = 10000;

        // The lower integration limit is `num_std_dev`
        let lower_cutoff = self.lower_cutoff(num_hwhm);
        // If `at` is less than the `lower_cutoff` an insignificant probability sits in the
        // integration range so return 0
        if to < lower_cutoff {
            return R::zero();
        }

        // If `at` is greater than the `upper_cutoff` an insignificant probability sits outside the
        // integration range so return 1
        if to > self.upper_cutoff(num_hwhm) {
            return R::one();
        }

        // If not we integrate
        let integrator = AdaptiveIntegrator::new(
            Complex::new(lower_cutoff, R::zero())..Complex::new(to, R::zero()),
            1000,
            R::from_f64(1e-6).unwrap(),
            vec![],
            R::from_f64(1e-6).unwrap(),
            R::from_f64(1e-6).unwrap(),
        );
        let runner = integrator.build_for(self).finalise().unwrap();

        let result = runner.run();

        result.unwrap().result().unwrap().real()
    }
}

impl<'a, R> Integrable for CenteredVoigtCDF<'a, R>
where
    R: Copy
        + RealField
        + ToPrimitive
        + FromPrimitive
        + AccumulateError<R>
        + IntegrableFloat
        + RescaleError,
    Complex<R>: ComplexField<RealField = R>
        + Copy
        + ToPrimitive
        + IntegrationOutput<Float = R, Scalar = Complex<R>, Real = R>,
{
    type Input = Complex<R>;
    type Output = Complex<R>;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        // The  cdf is the integral of the PDF
        Ok(Complex::new(
            self.distribution.probability_density_function(input.re),
            R::zero(),
        ))
    }
}

/// A produc of two Voigt distributions
struct VoigtPDFProduct<'a, R> {
    distribution: &'a CenteredVoigtPDF<R>,
    other: &'a CenteredVoigtPDF<R>,
}

impl<'a, R> VoigtPDFProduct<'a, R>
where
    R: Copy + RealField + ToPrimitive + AccumulateError<R> + IntegrableFloat + RescaleError,
    Complex<R>: ComplexField<RealField = R>
        + Copy
        + ToPrimitive
        + IntegrationOutput<Float = R, Scalar = Complex<R>, Real = R>,
{
    fn lower_central_value(&self) -> R {
        RealField::min(
            self.distribution.central_value(),
            self.other.central_value(),
        )
    }

    fn higher_central_value(&self) -> R {
        RealField::max(
            self.distribution.central_value(),
            self.other.central_value(),
        )
    }

    fn widest_hwhm(&self) -> R {
        RealField::max(
            self.distribution.half_width_half_maximum(),
            self.other.half_width_half_maximum(),
        )
    }

    fn lower_cutoff(&self, num_hwhm: usize) -> R {
        let num_hwhm = R::from_usize(num_hwhm).unwrap();
        self.lower_central_value() - num_hwhm * self.widest_hwhm()
    }

    fn upper_cutoff(&self, num_hwhm: usize) -> R {
        let num_hwhm = R::from_usize(num_hwhm).unwrap();
        self.higher_central_value() + num_hwhm * self.widest_hwhm()
    }

    fn integrate(self) -> R {
        let num_hwhm = 100;

        let integrator = AdaptiveIntegrator::new(
            Complex::new(self.lower_cutoff(num_hwhm), R::zero())
                ..Complex::new(self.upper_cutoff(num_hwhm), R::zero()),
            1000,
            R::from_f64(1e-6).unwrap(),
            vec![],
            R::from_f64(1e-6).unwrap(),
            R::from_f64(1e-6).unwrap(),
        );
        let runner = integrator.build_for(self).finalise().unwrap();

        let result = runner.run();

        result.unwrap().result().unwrap().real()
    }
}

impl<'a, R> Integrable for VoigtPDFProduct<'a, R>
where
    R: Copy
        + RealField
        + ToPrimitive
        + FromPrimitive
        + AccumulateError<R>
        + IntegrableFloat
        + RescaleError,
    Complex<R>: ComplexField<RealField = R>
        + Copy
        + ToPrimitive
        + IntegrationOutput<Float = R, Scalar = Complex<R>, Real = R>,
{
    type Input = Complex<R>;
    type Output = Complex<R>;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        Ok(Complex::new(
            self.distribution.probability_density_function(input.re)
                * self.other.probability_density_function(input.re),
            R::zero(),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::{CenteredVoigtPDF, ProbabilityDensityFunction};
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn voigt_cdf_at_central() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central: f64 = rng.gen();
        let scale = rng.gen();
        let std_dev = rng.gen();
        let dist = CenteredVoigtPDF::new(central, std_dev, scale);

        approx::assert_relative_eq!(
            0.5,
            dist.cumulative_distribution_function(central),
            max_relative = 1e-3
        );
    }

    #[test]
    fn voigt_cdf_at_large_input() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central: f64 = rng.gen();
        let scale = rng.gen();
        let std_dev = rng.gen();
        let dist = CenteredVoigtPDF::new(central, std_dev, scale);

        assert_eq!(1.0, dist.cumulative_distribution_function(std::f64::MAX),);
    }

    #[test]
    fn voigt_cdf_at_large_negative_input() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central: f64 = rng.gen();
        let scale = rng.gen();
        let std_dev = rng.gen();
        let dist = CenteredVoigtPDF::new(central, std_dev, scale);

        assert_eq!(0.0, dist.cumulative_distribution_function(-std::f64::MAX));
    }

    // This test compares to mathematica
    #[test]
    fn voigt_pdf_product_with_self() {
        let first_central: f64 = 0.25;
        let first_scale = 0.2;
        let first_std_dev = 0.2;
        let dist = CenteredVoigtPDF::new(first_central, first_std_dev, first_scale);

        let second_central: f64 = 1.0;
        let second_scale = 0.5;
        let second_std_dev = 0.1;
        let other = CenteredVoigtPDF::new(second_central, second_std_dev, second_scale);

        approx::assert_relative_eq!(0.221619, dist.overlap_with(&other), max_relative = 1e-4);
    }
}
