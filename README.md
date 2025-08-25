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