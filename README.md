# Tensorflow networks in C++

This repository provides instrumentary for inferencing tensorflow networks using Tensorflow bingdings for C++ 
and performing image retrieval with them.

### Requirements

* Tensorflow
* Protobuf
* Opencv (may be remove it later)
* Abseil _(for tests)_

#### Instructions

* Easy way to install tensorflow is by using [this](https://github.com/leggedrobotics/tensorflow-cpp) repository. 
It's uses prebuild binaries so it also very fast to install.

* For Protobuf installing I suggest using [this](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) instructions.
_The version that I'm using is 3.7.0_

* Abseil have classic way to install from [here](https://github.com/abseil/abseil-cpp)

 In order to build the project you have to use Cmake.
 
 Standard way of building the project:

 `mkdir build && cd build`

 `cmake .. && make -j`
 
 A static library `build/tf_wrapper/libTF_WRAPPER_EMBEDDING.a` and also two binary files: an example 
 `build/application/example/TF_WRAPPER_EXAMPLE` that uses this library and a simple console application 
 `build/application/metrics/TF_WRAPPER_METRICS` for accuracy calculation.

#### Setting

To configure library you should use `config.json` file.

* Parameter `"input size"` sets the size to which input image is resized to. It depends on the net that you are going
to use.
* Parameter `"datafile_path"` is a path to a .txt file where you want processed image's embeddings to be stored.
* Parameter `"images_path"` is a path to images which are forming your database, in which network is going to find a 
a match to a query image.
* Parameter `"pb_path"` is a path to .pb(protobuf) file, in which structure and weights of a trained networks a stored. 
Please note that as of now the checkpoints are not provided in this repo.
* Parameter `"input_node"` is a name of an input node of a trained network.
* Parameter `"output_node"` is a name of an output node of a trained network.
* Parameter `"top_n"` is the number of images closest (according to the Euclidean norm) to the query image
network will return to you.

### API
In order to interact with a library you should connect it via Cmake and include `"wrapper_base.h"`

1. Create an object of the `WeapperBase` class.
2. Call a `prepare_for_inference()` method.
3. Call a `inference_and_matching(img_path)` method with the argument that is the path to a query image. This method

Method _`inference_and_matching`_ will return `std::vector<EmbeddingsBase::distance>` - a vector with `"top_n"` image
 paths representing images closest to a query image. 

#### Example
The `TF_WRAPPER_EXAMPLE` file is an example of library usage. To run it you should pass one argument:

* `-img` - path to a query image

#### Metrics
The `TF_WRAPPER_METRICS` is a simple console application to calculate accuracy of matching. To run it you should pass two arguments:

* `--test_path` - path to a directory with query images that you'd like to match with the images in `"images_path"` directory
* `--top_n_classes` - number of plausible classes for a query. If the query image is among `-top_n_classes` unique classes
the match will be considered correct in metrics calculation.

### Dataset structure
The paths that you pass through `--test_path` or `"images_path"` should point to directories of the following structure:

```
dataset
+--building_1
|   +-- build1_1.jpg
|   +-- build1_2.jpg
|   +-- ...
+-- building_2
|   +-- build2_1.jpg
|   +-- build2_2.jpg
|   +-- ...
+-- ...

```


