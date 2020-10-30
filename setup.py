import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as requires_file:
    requirements = requires_file.read().split('\n')

with open('dev_requirements.txt', 'r') as dev_requires_file:
    dev_requirements = dev_requires_file.read().split('\n')

setuptools.setup(
    name='nessai',
    version='0.0.1',
    descrption='NesSAI: Nested Sampling with Aritifical Intelligence',
    long_description=long_description,
    author='Michael J. Williams',
    author_emial='m.williams.4@research.gla.ac.uk',
    url='https://github.com/mj-will/nessai',
    project_urls={
        'Documentation': '',
        'Paper': ''
    },
    license='MIT',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords=['nested sampling normalising flows machine learning'],
    packages=['nessai'],
    install_requires=requirements,
    test_require=dev_requirements,
    dev_require=dev_requirements,
    test_suite='tests'
)
