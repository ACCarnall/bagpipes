from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bagpipes',

    version='0.4.1',

    description='Galaxy spectral fitting',

    long_description=long_description,

    url='https://bagpipes.readthedocs.io',

    author='Adam Carnall',

    author_email='adamc@roe.ac.uk',

    packages=["bagpipes", "pipes_models", "pipes_filters"],

    include_package_data=True,

    install_requires=["numpy>=1.14.2", "corner", "pymultinest",
                      "matplotlib>=2.2.2", "scipy", "astropy",
                      "dynesty", "msgpack", "deepdish", "pandas"],

    project_urls={
        "readthedocs": "https://bagpipes.readthedocs.io",
        "GitHub": "https://github.com/ACCarnall/bagpipes",
        "ArXiv": "https://arxiv.org/abs/1712.04452"
    }
)
