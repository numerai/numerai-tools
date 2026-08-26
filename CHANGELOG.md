# Changelog

Releases before 0.7.0 are recorded in the git history.

## 0.7.0

### Added

- `scoring.neutral_corr`: rank, gaussianize, and neutralize predictions against a
  neutralizer matrix, then correlate with the target. No 1.5 power is applied to
  the predictions or the targets, and there is no `target_pow15` flag. This is
  not `feature_neutral_corr`, which neutralizes and then calls `numerai_corr`,
  re-ranking and re-powering predictions that were already transformed.
- `scoring.neutral_meta_model_contribution`: `correlation_contribution` with the
  same neutralization step inserted after the rank/gaussianize step. Only the
  submissions are neutralized; the meta model is expected to already be neutral.
  No variance normalization is applied.
- `scoring.filter_sort_neutralizers`: align data to a neutralizer matrix whose
  universe is a superset of the scored universe, checking only the ratio of
  scored ids that get dropped.

### Changed

- `scoring.correlation_contribution` now shares its target-bucketing and dot
  product step with `neutral_meta_model_contribution` via the new
  `scoring.contribution_scores`. Scores are unchanged.
