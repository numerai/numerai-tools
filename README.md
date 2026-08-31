# numerai-tools
A collection of open-source tools to help interact with Numerai, model data, and automate submissions.

## Installation
```
pip install numerai-tools
```

## Structure

- The `scoring.py` module contains critical functions used to score submissions. We use this code in our scoring system system. Leverage this to optimize your models for the tournaments.

  The Signals payout scores are built from two of its functions:
  - `neutral_correlation` ranks, gaussianizes, and neutralizes predictions against a
    neutralizer matrix, then correlates them with the target. Unlike
    `numerai_corr` it applies no 1.5 power, and unlike `feature_neutral_corr` it
    does not re-rank and re-power the predictions after neutralizing them.
  - `neutral_contribution` is `correlation_contribution` with that
    same neutralization step inserted after the rank/gaussianize step. It
    neutralizes the submissions only; pass a meta model that is already neutral.

- The `submissions.py` module provides helper functions to ensure your submissions are valid and formatted correctly. Use this in your automated prediction pipelines to ensure uploads don't fail.

- The `signals.py` module provides code specific to Numerai Signals such as
  churn and turnover. `neutral_churn` measures churn after applying each era's
  neutralizers, `calculate_mean_neutral_churn` averages that metric across the
  provided recent submissions, and `neutral_churn_penalty` calculates the
  positive-payout retention multiplier described by the Signals v3 churn
  penalty.
