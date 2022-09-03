from setuptools import setup, find_packages

setup(
    name='bayesianmm',
    version="0.0.1",
    packages=find_packages(include=['bayesianmm', 'bayesianmm.*']),
)