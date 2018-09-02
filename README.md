# Pre/Posterous
Estimate the impact of an intervention, with simplicity and humility

**Pre/Posterous is under active development** and the current release can be considered a 'proof of concept'. 
The largest restriction is that it only imports data from the [Reporter app](http://www.reporter-app.com/), 
which is my current default recommendation for anyone who is trying to track something for Quantified Self.

## Installation

You can install with `pip install preposterous`

If you want to install from source, then clone this repository and run `pip install -e .` from the project root.

## Testing

Tests can be seen in the `tests/` directory and run with `pytest`

## Use cases
### Quantified Self data
The primary use case is for quantified self, where you have periodic measurements of the target metric (weight, categorical sleep quality, ect) and potential interventions (medications, diet shifts, ect). This library can organize these into 'natural experiments' that point the way towards a causal relationship

*Warning* Python is pretty great, but nothing can replace a well powered [Double Blind Randomized Controlled Trial](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3196997/) for establishing causality. That said, many (most?) situations do not lend themselves to RCTs, and yet we're still forced to make decisions. That's where tools like this, used with an appreciation for non-binary modes of belief, can be helpful.

## Example
```
import preposterous.preposterous as ppl
pdf = ppl.PrePostDF()
pdf.add_outcome(filename='data/sample_reporter_output.csv')
pdf.add_intervention(filename='data/sample_reporter_output.csv')

# Sanity check the data
pdf.basic_info()

# Basic statistical test of difference between periods pre and post intervention
pdf.fisher_test(intervention='Exercise')

# Bayesian comparison of relative impact of multiple interventions
# (note that the sample data only contains one)
# Output is written to an image file named 'relative_effectiveness_YYYYMMDD.png'
pdf.outcomes(
    positive_outcomes=['Totally fine'],
    negative_outcomes=['Noticeable', 'Distracting'],
    window=3
)
pdf.calculate_relative_effectiveness()
_ = pdf.plot_relative_effectiveness()

```
