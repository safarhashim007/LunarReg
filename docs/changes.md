# Change record

## 2026-09-07 — Initial repository snapshot

This is the first commit of the existing LunarReg working tree. There is no
previous committed baseline from which to reconstruct a per-edit history.

### Existing implementation captured

- Packaged the image-only registration pipeline and command-line entry points:
  cached RoMa matching, reverse cycle checks, multi-hypothesis geometry,
  locked similarity refinement and constrained affine refinement.
- Included experimental local refinement, controlled benchmarks, metadata
  evaluation, diagnostics, input validation and output-preservation safeguards.
- Preserved legacy matching, evaluation, cropping and pair-preparation scripts.
- Included package configuration, recorded dependency versions, regression tests
  and existing validation documentation.

The implementation and historical validation details are documented in
[validation.md](validation.md). Those historical results are distinct from the
checks performed while preparing this commit.

### Documentation and repository preparation

- Preserved the existing README and added a repository inventory and links.
- Added this change record to distinguish the initial snapshot from later changes.
- Retained the existing ignore rules for datasets, outputs, models, virtual
  environments, build products and caches.
- Included the existing empty `default.profraw` file at the user's request.
- Reused the already initialized `main` repository; no remote was configured.
- No algorithm changes or dependency installations were made for this commit.

### Validation for this commit

The existing unittest suite was run in the project's local virtual environment.
**Result: all 18 tests passed.** Full model inference and
historical benchmark runs were not repeated for this repository preparation.
