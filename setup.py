import pathlib
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

HERE = pathlib.Path(__file__).parent

setuptools.setup(
    name="uncertain-trees",
    version="0.0.0",
    author="Ricardo Moreira",
    author_email="ricardo.1315.m@gmail.com",
    description="Measure uncertainty in tree-based Machine Learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ricardo1315m/uncertain_trees",
    project_urls={
        "Bug Tracker": "https://github.com/ricardo1315m/uncertain_trees/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=open(HERE / "src" / "requirements.txt").readlines(),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
