from setuptools import setup, find_packages

setup(name="calibrationtfg",
      packages=find_packages(),
      package_dir={"": "src"},
      description="Runs tag frequency grouping calibration experiments",
      author="Michael Kranzlein",
      version="1.0.0",
      project_urls={
          "github": "https://github.com/nert-nlp/calibration_tfg"
      })
