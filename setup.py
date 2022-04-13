from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
	long_description = fh.read()

setup(name="ott_visualization", version=1.0, 
      package_dir={"": "lib"},
      packages=find_packages(), 
      author="Charles Blakemore", 
      author_email="chas.blakemore@gmail.com",
      description='Python-based Visualization of Computations from the MATLAB library "Optical Tweezers Toolbox"',
      long_description=long_description,
      url="https://github.com/charlesblakemore/ott_visualization")

