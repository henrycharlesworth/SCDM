from setuptools import setup
import setuptools

setup(name='SCDM',
      version='0.1.0',
      description='Challenging extensions to openAI Gyms hand manipulation environments',
      url='http://github.com/henrycharlesworth/SCDM',
      author='Henry Charlesworth',
      author_email='H.Charlesworth@warwick.ac.uk',
      packages=setuptools.find_packages(),
      package_data={'SCDM.TD3_plus_demos': [
          'demonstrations/*'
      ]},
      zip_safe=False)
