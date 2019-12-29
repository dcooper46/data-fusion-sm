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
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    packages=find_packages(),
    license="BSD-3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True
)
