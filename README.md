# numerai-tools
A collection of open-source tools to help interact with Numerai, model data, and automate submissions.

## Installation
```
pip install numerai-tools
```

## Structure

- The `scoring.py` module contains critical functions used to score submissions. We use this code in our scoring system system. Leverage this to optimize your models for the tournaments.

- The `submissions.py` module provides helper functions to ensure your submissions are valid and formatted correctly. Use this in your automated prediction pipelines to ensure uploads don't fail.

- The `signals.py` module provides code specific to Numerai Signals such as churn and turnover. Use this to ensure your Signals submissions are properly formatted.

## Configurable neutralized weights

`generate_neutralized_weights`, `alpha`, and `meta_portfolio_contribution` accept
a `power` argument applied after ranking and gaussianization. The default is `1`.
Pass `power=1.5` when reproducing historical Alpha or MPC definitions:

```python
from numerai_tools.scoring import generate_neutralized_weights

weights = generate_neutralized_weights(
    predictions,
    neutralizers,
    sample_weights,
    power=1.5,
)
```
