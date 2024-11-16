from setuptools import find_packages,setup
from typing import List

classifiers = [  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
  ]


setup(
    name='Pratik_model',
    version='0.1.6',
    description='This package directly gives you output performance on 12 different algorithms',
    url='',
    license='MIT',
    author='pratik',
    long_description = open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=classifiers,
    author_email='pratikvdatey@gmail.com',
    keywords='Pratik_model',
    install_requires=['scikit-learn','catboost','xgboost'],
    project_urls= {'source_code' : "https://github.com/pratikdatey/Pratik_model"
                   }
   
)
