from setuptools import setup
from setuptools import find_packages

VERSION = "0.3.0"


def load(path):
    return open(path, "r").read()


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]


if __name__ == "__main__":
    setup(
        name="numerai_tools",
        version=VERSION,
        maintainer="Numerai",
        maintainer_email="support@numer.ai",
        description="A collection of open-source tools to help interact with Numerai, model data, and automate submissions.",
        long_description=load("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/numerai/numerai-tools",
        platforms="OS Independent",
        classifiers=classifiers,
        license="MIT License",
        package_data={
            "numerai_tools": ["LICENSE", "README.md", "py.typed"],
        },
        packages=find_packages(exclude=["tests"]),
        install_requires=[
            "pandas>=1.3.1",
            "numpy>=1.26.4,<2.0.0",
            "scipy>=1.11.4",
            "scikit-learn>=1.3.0",
        ],
    )
