# AI Usage Declaration — GJ 1132 XUV Evolution

This workflow was developed and is maintained by the researcher
(Rory Barnes, University of Washington). AI assistance (Anthropic
Claude, running both on the host and inside the analysis container)
was used in the following capacities:

## Code and workflow assistance
- Drafting and refactoring the Python data-preparation, analysis,
  and plotting scripts in the step directories, under researcher
  direction and review.
- Generating initial unit/integrity/qualitative/quantitative test
  scaffolds via vaibify's test generator; the researcher reviewed
  the assertions and the quantitative benchmark values are derived
  from researcher-approved pipeline outputs.
- Debugging container-environment and pipeline-orchestration issues
  (PATH drift after container recreation, test-runner invocation).

## Automated pipeline operation
- On 2026-06-10/11, with explicit researcher authorization, an AI
  agent re-ran steps A09-A11 after script edits, re-ran test
  categories on A05/A08/A09-A11, and recorded step verifications on
  the researcher's behalf. The researcher remains responsible for
  re-reviewing those attestations.

## What AI did NOT do
- All scientific decisions — model selection (vplanet modules,
  vconverge convergence criteria), priors, observational
  constraints, and interpretation of results — are the
  researcher's.
- AI did not modify scientific calculations or regenerate
  quantitative benchmarks; benchmark acceptance is reserved for the
  researcher.

## Tooling
- Workflow orchestration, verification state, and reproducibility
  tracking: vaibify (https://github.com/RoryBarnes/vaibify).
- Forward models: vplanet, vconverge, maxlev (versions recorded in
  the workflow's binary declarations and environment snapshot).
