from setuptools import setup, find_packages

setup(
    name="chargeshield-ai",
    version="1.0.0",
    packages=find_packages(include=["chargeshield", "chargeshield.*"]),
    python_requires=">=3.10",
)
