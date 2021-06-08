# Notes on setup: https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup

setup(
    name='SPARC_lib',
    version='0.12.1',
    description='Tools supporting the SPARC project at Hendrix',
    url='https://github.com/gjf2a/SPARC_lib',
    author='Gabriel Ferrer',
    author_email='ferrer@hendrix.edu',
    license='Public Domain',
    packages=['SPARC_lib']
)