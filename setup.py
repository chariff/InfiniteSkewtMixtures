from setuptools import setup, find_packages


def read_me():
    with open('README.md') as f:
        out = f.read()
    return out


setup(
    name='InfiniteSkewtMixtures',
    version='1.0.0',
    url='https://github.com/chariff/BayesianInfiniteMixtures',
    packages=find_packages(),
    author='Chariff Alkhassim',
    author_email='chariff.alkhassim@gmail.com',
    description='Bayesian infinite mixtures.',
    long_description=read_me(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "numpy >= 1.18.5+mkl",
        "scipy >= 1.4.1+mkl",
        "sklearn >= 0.23.2"
    ],
    license='MIT',
)


