from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="axiompy",
    version="3.0.0",
    author="RK RIAD & RK STUDIO 585",
    author_email="rkriad.official@gmail.com",
    description="A powerful Python mathematics engine for computation and education.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rkstudio585/AxiomPy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.0",
    ],
)