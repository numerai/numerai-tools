import random
import string
from typing import Callable, Optional

import pandas as pd


def generate_unique_values(generator: Callable, length: int, num_rows: int) -> list:
    """Generates a list of unique values using the provided generator function."""
    values: set[str] = set()
    while len(values) < num_rows:
        new_value = generator(length)
        values.add(new_value)
    return list(values)


def generate_ticker_ascii_uppercase(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def generate_fake_universe(
    date_value: str = "20130308", ticker_col: str = "numerai_ticker"
) -> pd.DataFrame:
    num_rows = 100
    data = {
        "date": [date_value for _ in range(num_rows)],
        ticker_col: [
            ticker + " US"
            for ticker in generate_unique_values(
                generate_ticker_ascii_uppercase, 3, num_rows
            )
        ],
    }

    uni = pd.DataFrame(data)
    return uni


def generate_new_submission(
    universe: pd.DataFrame,
    date_value: str = "2013-03-08",
    ticker_col: str = "numerai_ticker",
    legacy_headers: bool = False,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    if legacy_headers and date_col is None:
        date_col = "friday_date"
    elif date_col is None:
        date_col = date_col
    else:
        date_col = "date"

    rows = []
    for ticker in universe[ticker_col].unique():
        if legacy_headers:
            rows.append(
                {
                    ticker_col: ticker,
                    "signal": random.random(),
                    "data_type": "live",
                    date_col: date_value,
                }
            )
        else:
            rows.append({ticker_col: ticker, "signal": random.random()})
    return pd.DataFrame(rows)
