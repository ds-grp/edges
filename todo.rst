EDGES TO-DO LIST
================

- Vary degree of foreground polynomial
- Set up the full posterior, including the Bayesian evidence
- Figure out what the right choice of priors is (required for evidence).
- Use a different parametrization for the trough
- Perform Parallel Tempering Sampling to get a sense of whether feature is global minimum
- Simplify parameters, NPOLY is redundant with list of polynomial terms in
  params_2_vary and param_priors. We should get rid of params_2_vary and only
  changes params_vary_priors to something like "params_vary"
-Understand why intermediate temperatures are poorly conditioned and whether
 this is a problem.
-Reduce temperature range for ensemble sampler.
-Understand thermodynamic integration (and normalization)?
-Compute evidence for various polynomial orders once we have evidence working
 and also have good prior choices.
