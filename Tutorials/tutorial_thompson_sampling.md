# Thompson Sampling
by Mason Rumuly and Siddharth Ajit

## Problem Statement

Multi-arm bandit
Exploration vs exploitation
Regret vs. genie
For today, simple payout of 1 or nothing. Extends to more.

## Algorithm

Beta distribution: range, parameters, priors
What happens as samples are taken
Mention (reference paper) optimal regret is log(n)

## Example

3-arm bandit
Randomly generate the three payout probabilities
Step through samples in ipython notebook
Show choice, distributions, and payout after each sample (and sampled values on distribution before)

## Mention Extensions

Choice of priors (1, 1), (0.5, 0.5), (0, 0)
Alternate parameterizations (mean and sample size specifically)
Sample * payout_value for expected payout
Multiple possible outcomes, use Dirichlet distribution
All manner of contextual schemes
