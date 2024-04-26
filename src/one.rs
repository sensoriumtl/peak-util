use std::cmp::Ordering;

use nalgebra::RealField;

use crate::{Overlap, ProbabilityDensityFunction, ProductOfPDF, ProductOverlap};

#[derive(Clone, Debug)]
// A peak is a combination of a PDF and an oscillator strength.
pub struct PeakInSpectrum<R, D> {
    // The oscillator strength scales the distribution, so it's integral is `oscillator_strength`
    // over the range of independent variables
    pub oscillator_strength: R,
    // The integral of the distribution over the full range of independent variables is always
    // unity
    pub distribution: D,
}

impl<R, D: PartialEq> PartialEq for PeakInSpectrum<R, D> {
    fn eq(&self, other: &Self) -> bool {
        self.distribution == other.distribution
    }
}

impl<R, D: PartialEq> Eq for PeakInSpectrum<R, D> {}

impl<R: PartialOrd, D: PartialEq> PartialOrd for PeakInSpectrum<R, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.oscillator_strength, &other.oscillator_strength)
    }
}

impl<R: PartialOrd, D: PartialEq> Ord for PeakInSpectrum<R, D> {
    fn cmp(&self, other: &Self) -> Ordering {
        PartialOrd::partial_cmp(&self.oscillator_strength, &other.oscillator_strength).unwrap()
    }
}

// impl<R, D> PeakInSpectrum<R, D>
// where
//     R: Copy + RealField,
//     D: ProbabilityDensityFunction<Float = R>,
// {
//     fn central_value(&self) -> R {
//         self.distribution.central_value()
//     }
//
//     fn half_width_half_maximum(&self) -> R {
//         self.distribution.half_width_half_maximum()
//     }
//
//     // The integral of the distribution is 1, multiplied by the oscillator strength of the peak
//     fn integral(&self) -> R {
//         self.oscillator_strength
//     }
//
//     fn probability_density_function(&self, at: R) -> R {
//         self.distribution.probability_density_function(at)
//     }
//
//     fn cumulative_distribution_function(&self, at: R) -> R {
//         self.distribution.cumulative_distribution_function(at)
//     }
//
//     fn q_function(&self, at: R) -> R {
//         self.distribution.q_function(at)
//     }
// }

impl<'a, R, F, S> Overlap<'a, R, PeakInSpectrum<R, S>> for PeakInSpectrum<R, F>
where
    R: Copy + RealField,
    F: ProbabilityDensityFunction<Float = R> + 'a,
    S: ProbabilityDensityFunction<Float = R> + 'a,
    ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
{
    fn overlap(&'a self, other: &'a PeakInSpectrum<R, S>) -> R {
        let product = ProductOfPDF {
            first: &self.distribution,
            second: &other.distribution,
        };
        product.calculate() * self.oscillator_strength * other.oscillator_strength
    }
}
