# peak-util

______________________________________________________________________

`peak-util` is a utility library for working with multipeak `Spectrum` objects.

A `Spectrum` can be written as a weighted sum of probability density functions, `peak-util` permits calculation of coverage matrices between spectrum objects. A coverage matrix provides the overlap between every peak in each spectrum, and can be used to assess how well an object with a tuneable spectrum is overlapped with another with a fixed spectrum. This is often useful when doing inverse design, in which we want to target the design of an optical element to a known set of resonances.

A `Spectrum` object is comprised of a set of `PeakInSpectrum`. A `PeakInSpectrum` has two components, an oscillator strength which is a scalar float representing the weight of the peak and a probability density function which describes the properties of the peak. Currently `peak-util` supports three fundamental probability density functions, which can be created as follows:

```rust
use peak_util::{CauchyPDF, NormalPDF, CenteredVoigtPDF};

// Normal distributions are characterised by a mean and standard deviation
let mean = 0f64;
let standard_deviation = 0.1;
let normal = NormalPDF::new(mean, standard_deviation);

// Cauchy distributions are characterised by a central value and half width at half maximum
let central = 0f64;
let hwhm = 0.1;
let cauchy = CauchyPDF::new(central, hwhm);

// Centered voigt distributions are characterised by a central value, a normal standard deviation and a Cauchy half width at half maximum
let centered_voigt = CenteredVoigtPDF(central, hwhm, standard_deviation);
```

A `PeakInSpectrum` can be created from a known distribution as

```rust
use peak_util::{NormalPDF, PeakInSpectrum};

// Normal distributions are characterised by a mean and standard deviation
let mean = 0f64;
let standard_deviation = 0.1;
let distribution = NormalPDF::new(mean, standard_deviation);

let oscillator_strength = 0.05;
let peak_in_spectrum = PeakInSpectrum { oscillator_strength, distribution };
```

## Spectrums

______________________________________________________________________

The fundamental use of `peak-util` is in creating `Spectrum` objects. A `Spectrum` is comprised of a vector of `PeakInSpectrum`. Currently every peak in a spectrum must have the same underlying distribution. A `Spectrum` can be created as follows

```rust
use peak_util::{NormalPDF, PeakInSpectrum, Spectrum};

// Normal distributions are characterised by a mean and standard deviation
let mean = 0f64;
let standard_deviation = 0.1;
let distribution = NormalPDF::new(mean, standard_deviation);

let oscillator_strength = 0.05;
let peak_in_spectrum = PeakInSpectrum { oscillator_strength, distribution };
let peaks = vec![peak_in_spectrum; 5];

let spectrum = Spectrum::new(peaks);
```

Given two `Spectrum` objects we can asses how well they overlap by generating a line-by-line coverage vector. A line-by-line coverage vector gives the overlap between each peak in the *first* spectrum, with every peak in the second. The degrees of freedom in the second `Spectrum` are traced out

```rust
use peak_util::{NormalPDF, PeakInSpectrum, Spectrum};

// Normal distributions are characterised by a mean and standard deviation
let mean = 0f64;
let standard_deviation = 0.1;
let distribution = NormalPDF::new(mean, standard_deviation);

let oscillator_strength = 0.05;
let peak_in_spectrum = PeakInSpectrum { oscillator_strength, distribution };
let peaks = vec![peak_in_spectrum; 5];

let this = Spectrum::new(peaks);
let other = this.clone;
let line_by_line_coverage = this.scaled_line_by_line_coverage(&other);
```
