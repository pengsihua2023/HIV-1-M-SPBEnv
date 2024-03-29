with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
from setuptools import setup

from HIV_1_M_SPBEnv.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="HIV-1-M-SPBEnv",
    version=__version__,
    license='MIT',
    description="HIV-1-M-SPBEnv: HIV-1 M group subtype prediction using convolutional Autoencoder and full connected neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pengsihua2023/HIV-1-M-SPBEnv",
    author=["Sihua Peng"],
    author_email="Sihua.Peng@uga.edu",
    packages=["HIV_1_M_SPBEnv"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],    
    entry_points={"console_scripts": ["hiv-env=HIV_1_M_SPBEnv.main:run"]},
    py_modules=["HIV_1_M_SPBEnv"],
    install_requires=[
        'scikit-learn == 1.3.2',  
    ],
    zip_safe=False,
    include_package_data=True,
)
