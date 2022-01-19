import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

required_packages = [
    "torch==1.6.0", "torchvision", "scikit-learn==0.24.2", "pandas", "numpy",
    "matplotlib", "biopython", "lxml", "requests", "biovec",
    "CPCProt @ git+https://github.com/amyxlu/CPCProt.git#egg=CPCProt",
    "fair-esm"
]
setuptools.setup(
    name="proval",
    version="0.1",
    author="Philipp Väth,Christoph Raab",
    author_email="philipp.vaeth@fhws.de",
    description=
    "Proval: Evaluation Framework for Protein Sequence Embedding and code submission of paper 'Comparison of Protein Sequence Embeddings to Classify Molecular Functions'",
    license="MIT",
    url="https://github.com/philippvaeth/PROVAL",
    packages=setuptools.find_packages(include=['proval', 'proval.*']),
    python_requires=">=3.6",
    install_requires=required_packages,
    package_data={"": ["README.md", "LICENSE"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ])
