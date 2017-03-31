from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyGPGO',
    version='__version__',
    description='Bayesian Optimization tools in Python',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords = ['machine-learning', 'optimization', 'bayesian'],
    url='https://github.com/hawk31/pyGPGO',
    author='Jose Jimenez',
    author_email='jose.jimenez@upf.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'joblib'
    ],
    zip_safe=False)
