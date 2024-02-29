from setuptools import setup

from classlog.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="HIV-1-M-SPBEnv",
    version=__version__,
    description="Implementation of logistic regression for classification of sequences based on a reference set",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pengsihua2023/HIV-1-M-SPBEnv",
    author=["Sihua Peng"],
    author_email="Sihua.Peng@uga.edu",
    packages=["HIV-1-M-SPBEnv"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["classlog=classlog.ui:main"]},
    py_modules=["HIV-1-M-SPBEnv"],
    install_requires=[
        'Python == 3.9.18',
        'scikit-learn == 1.3.2',  
    ],
    zip_safe=False,
    include_package_data=True,
)
