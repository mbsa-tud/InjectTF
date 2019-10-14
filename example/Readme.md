# InjectTF example

This example illustrates the usage of the fault injection framework. A basic neural network is trained to classify digits using the MNIST dataset. Afterwards a fault injection experiment is conducted.

> Note:<br/>
> The framework has been tested with TensorFlow version 1.13.1 and will most definetly not work with TensorFlow 2.0. However, TF 2.0 is currently being evaluated. In the near future, InjectTF will be updated to work with TF 2.0.

### Usage:

> Consider using `docker` for an easier setup.<br/>[This](https://hub.docker.com/r/nvaitc/ai-lab) image is quite large, but can be considered as an all-in-one development environment.  

Using the docker image mentioned above, execute the following command in the root directory of the repository to open a shell inside the docker container:
```shell
$ docker run --rm \
          -it \
          -p 8888:8888 \
          -e JUPYTER_ENABLE_LAB=yes \
          --name injectTF_dev \
          -v "$(pwd):/home/jovyan/InjectTF_dev" \
          nvaitc/ai-lab:0.9 \
          bash

```

Now `cd` into `InjectTF_dev`. You should now be in the root directory of the repository. Run
```shell
$ python setup.py install
```
or
```shell
$ python setup.py develop
```
to install InjectTF.

Afterwards the simple model in the `example` subdirectory can be trained with:
```shell
$ python simple_model.py
```
After running the command above, the checkpoint and meta files of the simple model can be found in the `model` directory.
InjectTF works with frozen TensorFlow models, however, it is also possible to provide a non-frozen model. In the latter case, InjecTF will freeze the model during initialization of the framework.

Within the configuration file `InjectTF_example_conf.yml` the fault injection framework can be configured.

The following parameters can be set:

- Seed<br/>
By using this parameter, a seed for the random number generator used in the program can be set. This enables a deterministic number generation.

- ScalarFaultType<br/>
The fault type for scalar vaules. Possible values are `None`, `Rand`, `Zero`, and `BitFlip`.

- TensorFaultType<br/>
The fault type for tensor vaules. Possible values are `None`, `Rand`, `Zero`, and `BitFlip`.

- Ops<br/>
This parameter accepts a list of key-value pairs (e.g. `Add: 0.1`). Each key corresponds to a TensorFlow operation that should be injected, each value to its corresponding probability for injection ranging from `0.0` (== 0%) to `1.0` (== 100%). Refer to the provided example configuration file for more details. All currently implemented operations can be found in the file `InjectedFunctions.py` located in the `InjectTF` directory.

Run
```shell
$ python InjectTF_example.py
```
to start the injection experiment based on the parameters specified in the configration file.


__Note:__<br/>
InjectTF will inject the _result_ of each selected operation based on the specified probability. Since TensorFlow operations also work with tensors, one random element of the tensor is selected and then injected.<br/>
That means, if the input is passed into the network as one batch, it is represented as one huge tensor. Therefore, in case of an injection, only one element of that tensor will be altered according to the selected fault type.
