from setuptools import setup, find_packages

setup(
    name="Gloss2text",
    version="1.0.0",
    author="",
    description="From Gloss to Meaning: Evaluating Pre-trained Language Models for Bidirectional Sign Language Translation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
)