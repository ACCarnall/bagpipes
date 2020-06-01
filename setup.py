from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bagpipes',

    version='0.8.3',

    description='Galaxy spectral fitting',

    long_description=long_description,

    url='https://bagpipes.readthedocs.io',

    author='Adam Carnall',

    author_email='adamc@roe.ac.uk',

    packages=["bagpipes", "bagpipes.fitting", "bagpipes.catalogue",
              "bagpipes.models", "bagpipes.filters", "bagpipes.input",
              "bagpipes.plotting", "bagpipes.models.making", "bagpipes.moons"],

    include_package_data=True,

    install_requires=["numpy>=1.14.2", "corner", "pymultinest", "astropy",
                      "matplotlib>=2.2.2", "scipy", "msgpack", "deepdish",
                      "pandas"],

    project_urls={
        "readthedocs": "https://bagpipes.readthedocs.io",
        "GitHub": "https://github.com/ACCarnall/bagpipes",
        "ArXiv": "https://arxiv.org/abs/1712.04452"
    }
)
