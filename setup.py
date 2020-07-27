import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flowproposal",
    version="0.0.1",
    author="Michael J. Williams",
    url="https://gilsay.physics.gla.ac.uk/gitlab/michael.williams/flowproposal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
        'tqdm',
        'scipy',
        'nflows'
        ],
)
