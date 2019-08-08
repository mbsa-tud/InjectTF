## InjectTF - a fault injection framework for TensorFlow

<p align="left">
<a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square"></a>
</p>

InjectTF is a Python framework for fault injection into TensorFlow models. It builds an fault-injected counterpart to a given neural network, based on a user-defined configuration file. Using the framework, it is possible to inject faults into TensorFlow operations within the network and analyze the performance of the model when exposed to faults.

### Usage (WIP)

Please refer to the examples in the `example` folder for details on how to use the framework.


### Class diagram
![InjectTF class diagram](Readme_images/Class_diagram.svg)

### Tests

To run the tests execute the shell script `run_unittests.sh`.
