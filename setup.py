# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="fosQCA",
    version="0.5.0",
    python_requires=">=3.12",
    author="Tibo Ulens",
    author_email="ulens.tibo@gmail.com",
    description="A QCA tool that can optimise solutions for multiple input sets",
    install_requires=["pandas"],
    license="MIT",
    url="https://github.com/Tibo-Ulens/fosQCA",
    packages=find_packages(),
    platforms=["all"],
    entry_points={"console_scripts": ["fosqca = fosqca.fosqca:main"]},
)
