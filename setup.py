from setuptools import setup

setup(
    name="InjectTF",
    version="1.0.0",
    description="A fault injection framework for TensorFlow",
    author="Michael Beyer",
    packages=["InjectTF"],
    install_requires=["pyyaml"],
)
