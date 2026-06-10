import tomllib
from pathlib import Path


def test_pandas_dependency_supports_pandas_2_and_3():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject:
        config = tomllib.load(pyproject)

    pandas_requirement = config["tool"]["poetry"]["dependencies"]["pandas"]

    assert pandas_requirement == ">=2.2.2,<4.0.0"
