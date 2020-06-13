# Tensorflow networks in C++

This repository provides instrumentary for inferencing tensorflow networks using Tensorflow bingdings for C++ 
and performing image retrieval with them.

### Requirements

* Tensorflow
* Protobuf
* Opencv (may be remove it later)
* Abseil _(for tests)_

Alternatively, you can skip installation of the requirements and use 
[this](https://hub.docker.com/r/uclh/tensorflow-cpp-opencv) docker container (with a wrapper-v2 tag). 
#### Instructions

* Easy way to install tensorflow is by using [this](https://github.com/leggedrobotics/tensorflow-cpp) repository. 
It's uses prebuild binaries so it also very fast to install.

* For Protobuf installing I suggest using [this](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) instructions.
_The version that I'm using is 3.7.0_

* Abseil has a classic way to install from [here](https://github.com/abseil/abseil-cpp)

 In order to build the project you have to use Cmake.
 
 Standard way of building the project:

 `mkdir build && cd build`

 `cmake .. && make -j`
 
 A static library `build/tf_wrapper/libTF_WRAPPER_EMBEDDING.a` and also three binary files: two examples  
 `build/application/embedings/TF_EMBEDDINGS_EXAMPLE` and `build/application/segmentation/TF_SEGMENTATION_EXAMPLE` 
 for embedding calculation and segmentation respectively and a console application 
 `build/application/metrics/TF_WRAPPER_METRICS` that performs image retrieval and calculates it's accuracy.

#### Setting

To configure library you should use .json files. The library provides inferencing tools for embedding nets for image 
retrieval and segmentation nets. You should use separate .json files for each type of network.

##### Common part:
* Parameter `"input_size"` sets the size to which input image is resized to. It depends on the net that you are going
to use.
* Parameter `"images_path"` is a path to images which are forming your database, in which network is going to find a 
a match to a query image.
* Parameter `"pb_path"` is a path to .pb(protobuf) file, in which structure and weights of a trained networks a stored. 
Please note that as of now the checkpoints are not provided in this repo.
* Parameter `"input_node"` is a name of an input node of a trained network.
* Parameter `"output_node"` is a name of an output node of a trained network.

##### Embeddings only:
* Parameter `"datafile_path"` is a path to a .txt file where you want processed image's embeddings to be stored.
* Parameter `"top_n"` is the number of images closest (according to the Euclidean norm) to the query image
network will return to you.

##### Segmentation only:
* Parameter `"colors_path"` is a path to a .csv file where the data for colored mask construction is stored. 

### API
In order to interact with a library you should connect it via Cmake and include `"segmentation_base.h"` 
or/and `"embeddings_base.h"`.

#### Embeddings

1. Include `"embeddings_base.h"`.
2. Create an object of the `EmbeddingsWrapper` class.
3. Call `prepare_for_inference(config_path)` method with the argument that is the path to a config file 
 described bellow.
4. Call `inference_and_matching(img_path)` method with the argument that is the path to a query image.

Method _`inference_and_matching`_ will return `std::vector<EmbeddingsWrapper::distance>` - a vector with `"top_n"` image
 paths representing images closest to a query image. 

#### Segmentation

1. Include `"segmentation_base.h"`
2. Create an object of the `SegmentationWrapper` class.
3. Call `prepare_for_inference(config_path)` method with the argument that is the path to a config file 
 described bellow.
4. Call `process_images()` method. 
5. Call `get_indexed()` method if you need mask of indexes
6. Call `get_colored()` method if you need colored mask
7. Call `get_masked(classes_to_mask)` where `classes_to_mask` is a set of int coresseponding to segmentation classes
 defined in `"classes.csv"`, if you need to cut `classes_to_mask` segmentation classes out of the images.

Methods _`get_colored`, `get_indexed`, `get_masked`_ will return `std::vector<cv::Mat>` - a vector of processed 
images

### Applications

#### Embeddings
The `TF_EMBEDDINGS_EXAMPLE` file is an example of `"embeddings_base.h"` usage. To run it you should pass 
one argument:

* `-img` - path to a query image

#### Metrics
The `TF_WRAPPER_METRICS` is a simple console application to calculate accuracy of matching. To run it you should pass two arguments:

* `--test_path` - path to a directory with query images that you'd like to match with the images in `"images_path"` directory
* `--top_n_classes` - number of plausible classes for a query. If the query image is among `-top_n_classes` unique classes
the match will be considered correct in metrics calculation.
* `--use_segmentation` - whether to mask images before matching

#### Segmentation
The `TF_SEGMENTATION_EXAMPLE` file is an example of `"segmentation_base.h"` usage. To run it you should pass 
one argument:

* `-img` - path to an input image

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


