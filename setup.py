import setuptools
from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules.")
    

dataexts  = (".py",  ".pyx", ".pxd")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyBWMD",
    version="0.0.1",
    author="Edward Raff",
    author_email="raff.edward@gmail.com",
    description="An imeplementation of the Burrows Wheeler Markov Distance algorithm for converting byte sequences to feature vectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdwardRaff/pyBWMD",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'cython', 'scipy'],
    package_data={'pyBWMD': ['*.pxd', '*.pyx']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    zip_safe = False,
)
