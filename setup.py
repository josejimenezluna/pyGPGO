from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyGPGO',
    version='0.1.0.dev1',
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
    packages=['pyGPGO'],
    install_requires=[
        'numpy',
        'scipy',
        'joblib',
        'cma'
    ],
    zip_safe=False)
