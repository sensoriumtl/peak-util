use std::collections::BinaryHeap;

use nalgebra::RealField;
use ndarray::{Array1, Array2, Axis};

use crate::{Overlap, PeakInSpectrum, ProbabilityDensityFunction, ProductOfPDF, ProductOverlap};

// A `Spectrum` is a collection of individual peaks.
//
// The collection is ordered so as when iterating the peaks with the largest oscillator strength
// are visited first. The oscillator strength of the enclosed peaks is normalised to a sum of
// unity so the [`Spectrum`] can still be interpreted as a PDF.
#[derive(Debug, Clone)]
pub struct Spectrum<R: PartialOrd, D: PartialEq> {
    // The peaks comprising the spectrum.
    //
    // Oscillator strength in the peak heap is normalised so the sum acts as a probability density
    // function
    peaks: BinaryHeap<PeakInSpectrum<R, D>>,
    // The scale factor. True values from the spectrum are the probability density function in the
    // peaks field, multiplied by `scale_factor`
    scale_factor: R,
}

impl<R: Copy + PartialOrd + RealField, D: PartialEq> Spectrum<R, D> {
    pub fn new(input: Vec<PeakInSpectrum<R, D>>) -> Self {
        let mut peaks = BinaryHeap::new();
        let scale_factor = input
            .iter()
            .fold(R::zero(), |acc, each| acc + each.oscillator_strength);

        for mut each in input {
            each.oscillator_strength /= scale_factor;
            peaks.push(each);
        }

        Self {
            peaks,
            scale_factor,
        }
    }

    fn len(&self) -> usize {
        self.peaks.len()
    }
}

impl<R, F> Spectrum<R, F>
where
    R: Copy + PartialOrd + RealField,
    F: ProbabilityDensityFunction<Float = R> + PartialEq,
{
    pub fn probability_density_function(&self, at: R) -> R {
        self.peaks.iter().fold(R::zero(), |a, b| {
            a + b.distribution.probability_density_function(at)
        })
    }

    pub fn evaluate(&self, at: R) -> R {
        self.probability_density_function(at) * self.scale_factor
    }

    // The coverage factor describes how well `other` spans the spectrum of `self`.
    //
    // If other was a unit distribution, which returned unity for all input values where `self` is
    // non-zero the coverage factor would be unity. Alternatively if `other` were a zero
    // distribution at all inputs where self is zero the coverage factor would be zero.
    //
    // In the intermediate region we compute the overlap integral of all peaks in self and other,
    // weighted as ... and sum
    pub fn coverage<'a, S>(&'a self, other: &'a Spectrum<R, S>) -> R
    where
        S: ProbabilityDensityFunction<Float = R>,
        ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
    {
        self.scaled_coverage_matrix(other).sum()
    }

    // Summed over `other`, so the element count is equal to `self.len()`
    pub fn scaled_line_by_line_coverage<'a, S>(&'a self, other: &'a Spectrum<R, S>) -> Array1<R>
    where
        S: ProbabilityDensityFunction<Float = R>,
        ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
    {
        self.scaled_coverage_matrix(other).sum_axis(Axis(1))
    }

    // Summed over `other`, so the element count is equal to `self.len()`
    pub fn unscaled_line_by_line_coverage<'a, S>(&'a self, other: &'a Spectrum<R, S>) -> Array1<R>
    where
        S: ProbabilityDensityFunction<Float = R>,
        ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
    {
        self.unscaled_coverage_matrix(other).sum_axis(Axis(1))
    }

    // Ordered so `self` is on Axis(0) and `other` is on `Axis(1)`
    pub fn unscaled_coverage_matrix<'a, S>(&'a self, other: &'a Spectrum<R, S>) -> Array2<R>
    where
        S: ProbabilityDensityFunction<Float = R>,
        ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
    {
        let data = self
            .peaks
            .iter()
            .flat_map(|each_self| {
                other
                    .peaks
                    .iter()
                    .map(|each_other| each_self.distribution.overlap(&each_other.distribution))
            })
            .collect::<Vec<_>>();

        Array2::from_shape_vec((self.len(), other.len()), data)
            .expect("shape of internal data incompatible with expected size of coverage matrix")
    }

    // Ordered so `self` is on Axis(0) and `other` is on `Axis(1)`
    pub fn scaled_coverage_matrix<'a, S>(&'a self, other: &'a Spectrum<R, S>) -> Array2<R>
    where
        S: ProbabilityDensityFunction<Float = R>,
        ProductOfPDF<&'a F, &'a S>: ProductOverlap<R>,
    {
        let data = self
            .peaks
            .iter()
            .flat_map(|each_self| {
                other
                    .peaks
                    .iter()
                    .map(|each_other| each_self.overlap(&each_other))
            })
            .collect::<Vec<_>>();

        Array2::from_shape_vec((self.len(), other.len()), data)
            .expect("shape of internal data incompatible with expected size of coverage matrix")
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use crate::{pdf::CauchyPDF, ProbabilityDensityFunction};

    use super::{PeakInSpectrum, Spectrum};

    #[test]
    fn peak_heap_has_unit_oscillator_strength() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let num_peaks = 5;
        let peaks = (0..num_peaks)
            .map(|_| {
                let central = rng.gen();
                let scale: f64 = rng.gen();
                let oscillator_strength: f64 = rng.gen();
                let distribution = CauchyPDF::new(central, scale);
                PeakInSpectrum {
                    oscillator_strength,
                    distribution,
                }
            })
            .collect::<Vec<_>>();

        let multi = Spectrum::new(peaks);

        let total_oscillator_strength = multi
            .peaks
            .into_iter()
            .fold(0.0, |acc, each| acc + each.oscillator_strength);

        assert_eq!(1.0, total_oscillator_strength);
    }

    #[test]
    fn single_peak_heap_coverage_with_self_matches_bare_overlap() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let central = rng.gen();
        let scale: f64 = rng.gen();
        let oscillator_strength: f64 = rng.gen();
        let distribution = CauchyPDF::new(central, scale);
        let peaks = vec![PeakInSpectrum {
            oscillator_strength,
            distribution: distribution.clone(),
        }];
        let multi = Spectrum::new(peaks);

        let bare = distribution.overlap_with(&distribution);

        let coverage = multi.coverage(&multi);

        assert_eq!(bare, coverage);
    }

    #[test]
    fn unscaled_coverage_matrix_has_correct_contents() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let num_first = 3;
        let first_peaks = (0..num_first)
            .map(|_| {
                let central = rng.gen();
                let scale: f64 = rng.gen();
                let oscillator_strength: f64 = rng.gen();
                let distribution = CauchyPDF::new(central, scale);
                PeakInSpectrum {
                    oscillator_strength,
                    distribution,
                }
            })
            .collect::<Vec<_>>();
        let first = Spectrum::new(first_peaks.clone());

        let num_second = 5;
        let second_peaks = (0..num_second)
            .map(|_| {
                let central = rng.gen();
                let scale: f64 = rng.gen();
                let oscillator_strength: f64 = rng.gen();
                let distribution = CauchyPDF::new(central, scale);
                PeakInSpectrum {
                    oscillator_strength,
                    distribution,
                }
            })
            .collect::<Vec<_>>();
        let second = Spectrum::new(second_peaks.clone());

        let calculated_coverage_matrix = first.unscaled_coverage_matrix(&second);

        for ii in 0..num_first {
            for jj in 0..num_second {
                assert_eq!(
                    calculated_coverage_matrix[[ii, jj]],
                    first_peaks[ii]
                        .distribution
                        .overlap_with(&second_peaks[jj].distribution)
                );
            }
        }
    }

    #[test]
    fn scaled_coverage_matrix_has_correct_contents() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let num_first = 3;
        let first_peaks = (0..num_first)
            .map(|_| {
                let central = rng.gen();
                let scale: f64 = rng.gen();
                let oscillator_strength: f64 = rng.gen();
                let distribution = CauchyPDF::new(central, scale);
                PeakInSpectrum {
                    oscillator_strength,
                    distribution,
                }
            })
            .collect::<Vec<_>>();
        let first = Spectrum::new(first_peaks.clone());

        let num_second = 5;
        let second_peaks = (0..num_second)
            .map(|_| {
                let central = rng.gen();
                let scale: f64 = rng.gen();
                let oscillator_strength: f64 = rng.gen();
                let distribution = CauchyPDF::new(central, scale);
                PeakInSpectrum {
                    oscillator_strength,
                    distribution,
                }
            })
            .collect::<Vec<_>>();
        let second = Spectrum::new(second_peaks.clone());

        let first_scaling = first_peaks
            .iter()
            .fold(0.0, |a, b| a + b.oscillator_strength);
        let second_scaling = second_peaks
            .iter()
            .fold(0.0, |a, b| a + b.oscillator_strength);

        let calculated_coverage_matrix = first.scaled_coverage_matrix(&second);

        for ii in 0..num_first {
            for jj in 0..num_second {
                approx::assert_relative_eq!(
                    calculated_coverage_matrix[[ii, jj]] * first_scaling * second_scaling,
                    first_peaks[ii]
                        .distribution
                        .overlap_with(&second_peaks[jj].distribution)
                        * first_peaks[ii].oscillator_strength
                        * second_peaks[jj].oscillator_strength,
                    max_relative = 1e-10
                );
            }
        }
    }
}
