from setuptools import find_packages,setup
from typing import List

classifiers = [  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python'
  ]


setup(
    name='Lazyme',
    version='0.0.1',
    description='This package directly gives you output performance on different models',
    url='',
    license='MIT',
    long_description=open('README.md').read() + '\n\n' + open('ChangeLog.txt').read(),
    author='pratik',
    classifiers=classifiers,
    author_email='pratikvdatey@gmail.com',
    keywords='Lazyme',
    packages=find_packages(),
    install_requires=['sklearn','lightgbm','catboost','xgboost']
)
