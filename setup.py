from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bagpipes', 

    version='0.1.0',

    description='Galaxy spectral fitting',

    long_description=long_description,

    url='https://bagpipes.readthedocs.io',

    author='Adam Carnall',

    author_email='adamc@roe.ac.uk',

    packages=["bagpipes"],

    package_data={
    "tables/IGM" : ["Lyman_series_coefs_Inoue_2014_Table2.txt"]
    "tables/nebular/bc03_miles" : ["*.fits"]
    "tables/stellar/bc03_miles" : ["*.fits"]
    "tables/dust/bc03_miles" : ["*.txt"]
    "" : ["*.rst", "*.txt"]
    }

    install_requires=['numpy', "corner", "pymultinest", "matplotlib", "scipy", "astropy"],  

    project_urls={
        "GitHub": "https://github.com/ACCarnall/bagpipes",
        "ArXiv": "https://arxiv.org/abs/1712.04452",
    },
)