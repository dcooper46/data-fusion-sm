from setuptools import find_packages, setup


required = [
    "python-dateutil == 2.8.0",
    "pandas >= 0.24.0",
    "numpy >= 1.16.4",
    "scipy",
    "sklearn",
    "lap"
]

setup(
    name='fusion',
    version='1.0.0',
    author="Daniel Cooper",
    author_email="djcoop46@yahoo.com",
    install_requires=required,
    packages=find_packages(),
    include_package_data=True
)
