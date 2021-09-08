from setuptools import setup, find_packages

setup(
  name = 'dropconnect-tensorflow',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Drop Connect - Tensorflow',
  author = 'Arya Aftab',
  author_email = 'arya.aftab@gmail.com',
  url = 'https://github.com/AryaAftab/dropconnect-tensorflow',
  keywords = [
    'deep learning',
    'tensorflow',
    'drop connect'    
  ],
  install_requires=[
    'tensorflow>=2.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
