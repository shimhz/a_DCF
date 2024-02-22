from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'a_dcf',
    packages = find_packages(include=["dcf"]),
    version = '0.0.1',  # Ideally should be same as your GitHub release tag varsion
    description = 'a-dcf preliminary release',
    author = 'Jee-weon Jung',
    author_email = 'jeeweonj@ieee.org',
    url = 'https://github.com/shimhz/a_dcf',
    keywords = ['adcf', 'a-dcf', 'a_dcf'],
    classifiers = [],
)
