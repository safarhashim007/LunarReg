# LunarReg-GPU experiment integration

This document records the portable improvements observed in the external
`LunarReg-GPU` experiment archive. The archive contains data and generated
outputs only; it does not contain source code, Git history, tests, dependency
versions, or an executable entry point. Therefore no unreviewed GPU code is
merged into the production pipeline.

## Improvements retained

- CUDA execution was demonstrated on an NVIDIA RTX 4000 SFF Ada Generation.
- Blind registration used a complete 30-degree sweep across `[0, 360)` and
  recorded per-angle valid-match counts.
- Rotation match arrays and progress files were retained, allowing a partial
  sweep to resume without discarding completed angles.
- Output namespaces were experiment-specific, preserving failed and rejected
  attempts.
- Image-only fitting and a transform-freeze receipt were recorded before any
  possible geometry evaluation.
- Prototype experiments were labelled separately from production quality-gate
  results.
- GPU failures were preserved with actionable reasons, including missing model
  cache, invalid hypotheses, and CUDA out-of-memory.

These behaviors are implemented and guarded in the source pipeline through the
blind matcher, output-preservation checks, geometry seal, freeze receipt, and
post-freeze evaluator. This branch documents the GPU evidence without treating
the experiment outputs as source or as validated production results.

## Evidence boundary

The GPU archive's visible OHRC runs did not pass the production quality gate.
The strongest recorded blind run had an internal fit RMSE of 2.8008 pixels but
only 1.5625% strict spatial coverage, so it was correctly rejected. Prototype
gates passed in two experiments, while their production quality gates remained
false. A 20,000-sample run exhausted available GPU memory.

No GPU wall-clock benchmark, held-out post-freeze evaluation, readiness report,
or independent surveyed landmark validation was found. These remain required
before any GPU configuration can be promoted.

## Reproduction and promotion policy

Use a fresh output directory for every run. Keep blind fitting image-only;
open `geometry_controls_EVALUATION_ONLY.csv` only after the transform is
frozen; and report the complete post-freeze metric set. GPU acceleration may be
selected with the existing `--device cuda` option, but it does not relax the
quality gate or scientific validation requirements.

The original V9/frozen replay remains the production baseline until a new GPU
run passes the same gate, post-freeze evaluation, readiness checks, and
reproducibility review.
