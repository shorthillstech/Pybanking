from setuptools import setup, find_packages


setup(
    name='pybanking',
    version='0.1',
    license='MIT',
    author="Shorthills Tech",
    author_email='apurv@shorthillstech.com',
    packages=find_packages('pybanking'),
    package_dir={'': 'pybanking'},
    url='',
    keywords='banking project',
    install_requires=[
          'scikit-learn',
      ],

)
