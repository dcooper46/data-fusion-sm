from setuptools import find_packages, setup

import datafusionsm


with open("README.md", "r") as fh:
    long_description = fh.read()


required = ["pandas", "numpy", "scipy", "sklearn", "lap"]

setup(
    name='data-fusion-sm',
    version=datafusionsm.__version__,
    author="Daniel Cooper",
    author_email="djcoop46@yahoo.com",
    url="https://github.com/dcooper46/data-fusion-sm",
    project_urls={
        "Documentation": "https://data-fusion-sm.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/dcooper46/data-fusion-sm",
        "Bug Tracker": "https://github.com/dcooper46/data-fusion-sm/issues",
    },
    description="data fusion in python via statistical matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    packages=find_packages(),
    license="BSD-3-Clause",
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True
)
