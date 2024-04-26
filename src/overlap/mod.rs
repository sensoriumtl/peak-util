//! Overlap integrals between primitive probability density functions

use crate::{CauchyPDF, CenteredVoigtPDF, NormalPDF, ProbabilityDensityFunction};

use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{FromPrimitive, ToPrimitive};
use quad_rs::{
    AccumulateError, AdaptiveIntegrator, GenerateBuilder, Integrable, IntegrableFloat,
    IntegrationOutput, RescaleError,
};

// Overlap betwee two probability distribution functions
pub trait Overlap<'a, R, D> {
    // Calculate the overlap integral from -infinity to +infinity for the product of self and other
    fn overlap(&'a self, other: &'a D) -> R;
}

pub(crate) struct ProductOfPDF<F, S> {
    pub(crate) first: F,
    pub(crate) second: S,
}

impl<'a, R, F, S> Overlap<'a, R, S> for F
where
    F: ProbabilityDensityFunction<Float = R> + 'a,
    S: ProbabilityDensityFunction<Float = R> + 'a,
    ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
{
    fn overlap(&'a self, other: &'a S) -> R {
        let product = ProductOfPDF {
            first: self,
            second: other,
        };
        product.calculate()
    }
}

pub(crate) trait ProductOverlap<R> {
    fn calculate(&self) -> R;
}

impl<'a, R, D> ProductOverlap<R> for ProductOfPDF<&'a D, &'a D>
where
    D: ProbabilityDensityFunction<Float = R>,
{
    fn calculate(&self) -> R {
        self.first.overlap_with(self.second)
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a CauchyPDF<R>, &'a NormalPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a CauchyPDF<R>, &'a CenteredVoigtPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a CenteredVoigtPDF<R>, &'a CauchyPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a CenteredVoigtPDF<R>, &'a NormalPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a NormalPDF<R>, &'a CauchyPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, R> ProductOverlap<R> for ProductOfPDF<&'a NormalPDF<R>, &'a CenteredVoigtPDF<R>>
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
    fn calculate(&self) -> R {
        self.numerical_overlap()
    }
}

impl<'a, F, S> ProductOfPDF<&'a F, &'a S>
where
    F: ProbabilityDensityFunction + Send + Sync,
    S: ProbabilityDensityFunction<Float = <F as ProbabilityDensityFunction>::Float> + Send + Sync,
    <F as ProbabilityDensityFunction>::Float: RealField,
{
    fn lower_central_value(&self) -> <F as ProbabilityDensityFunction>::Float {
        RealField::min(self.first.central_value(), self.second.central_value())
    }

    fn higher_central_value(&self) -> <F as ProbabilityDensityFunction>::Float {
        RealField::max(self.first.central_value(), self.second.central_value())
    }

    fn widest_hwhm(&self) -> <F as ProbabilityDensityFunction>::Float {
        RealField::max(
            self.first.half_width_half_maximum(),
            self.second.half_width_half_maximum(),
        )
    }

    fn lower_cutoff(&self, num_hwhm: usize) -> <F as ProbabilityDensityFunction>::Float {
        let num_hwhm = <F as ProbabilityDensityFunction>::Float::from_usize(num_hwhm).unwrap();
        self.lower_central_value() - num_hwhm * self.widest_hwhm()
    }

    fn upper_cutoff(&self, num_hwhm: usize) -> <F as ProbabilityDensityFunction>::Float {
        let num_hwhm = <F as ProbabilityDensityFunction>::Float::from_usize(num_hwhm).unwrap();
        self.higher_central_value() + num_hwhm * self.widest_hwhm()
    }

    fn numerical_overlap<R>(&self) -> R
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
        F: ProbabilityDensityFunction<Float = R> + Send + Sync,
        S: ProbabilityDensityFunction<Float = R> + Send + Sync,
    {
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

impl<'a, R, F, S> Integrable for &ProductOfPDF<&'a F, &'a S>
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
    F: ProbabilityDensityFunction<Float = R>,
    S: ProbabilityDensityFunction<Float = R>,
{
    type Input = Complex<R>;
    type Output = Complex<R>;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        Ok(Complex::new(
            self.first.probability_density_function(input.re)
                * self.second.probability_density_function(input.re),
            R::zero(),
        ))
    }
}
