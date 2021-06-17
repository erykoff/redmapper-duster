from setuptools import setup, find_packages


exec(open('duster/_version.py').read())

name = 'duster'

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    version=__version__, # noqa
    description="Dust Extinction with photometric Red galaxies",
    author="Eli Rykoff, Eduardo Rozo",
    author_email="erykoff@stanford.edu",
    url="https://github.com/erykoff/redmapper_duster",
)
