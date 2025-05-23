from setuptools import setup, find_packages

with open('README.md', 'r') as f:
  long_description = f.read()

setup(
  name='gym-games',
  version='1.0.4',
  keywords=['AI', 'Reinforcement Learning', 'Games', 'Pygame', 'MinAtar'],
  description='This is a gym version of various games for reinforcenment learning.',
  url='https://github.com/qlan3/gym-games',
  author='qlan3',
  author_email='qlan3@ualberta.ca',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=find_packages(),
  python_requires='>=3.5',
  install_requires=[
    'numpy>=1.16.4',
    'gymnasium',
    'setuptools>=65.5.1',
    'pygame>=1.9.6',
    'MinAtar @ git+https://github.com/kenjyoung/MinAtar@master#egg=MinAtar',
    'ple @ git+https://github.com/ntasfi/PyGame-Learning-Environment@master#egg=ple'
  ]
)
