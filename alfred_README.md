# WELCOME TO DEEPNANO DEMO


## License

Use of this software implies accepting all the terms and conditions described in
the
[license](https://gitlab.kaust.edu.sa/makam0a/deepnano/-/blob/master/LICENSE)
document available in this repository.  We remind users that the use of this
software is permitted for non-commercial applications, and proper credit must be
given to the authors whenever this software is used.

## Overall description

This repository contains a demonstration of the flat optics design software ALFRED described in detail in the publication: 

*Broadband vectorial ultrathin optics with experimental efficiency up to 99% in the visible via universal approximators*

available as an open access article at [Light: Science & Applications volume 10, Article number: 47 (2021)](https://www.nature.com/articles/s41377-021-00489-7). 

The code makes use of the theory described in the publication:

*Generalized Maxwell projections for multi-mode network Photonics* [Scientific Reports volume 10, Article number: 9038 (2020)](https://doi.org/10.1038/s41598-020-65293-6)

Users are encouraged to read both publications and familiarize themselves with the underlying theory and logic behind the  software.

#### Alfred
ALFRED stands for Autonomous Learning Framework for Rule-based Evolutionary Design, it is an inverse design software platform 
intended for the design of high efficiency flat optics. Given a desired optical response as the input ALFRED will find the
nanoscale geometry of the device that best approximates this response. 

The program is composed of two parts: A particle swarm optimizer and a neural network prediction unit

![](https://github.com/makamoa/alfred/blob/assets/Alfred_overview.png)

Alfred works by launching the particles into a multidimensional search space containing a very large number of possible
nanostructure geometries. Each particle evaluates the performance of a candidate geometry and explores the search space
according to the values assigned to its inertia, social and memory parameters. The behaviour of the particles is
intended to resemble the behaviour of social insects, such as ants or bees, in the sense that the exploration of the 
environment is carried out by individuals that share information with each other. For fast evaluation of the performance
of a possible candidate geometry each particle is equipped with a neural network prediction unit. The unit has been trained
on a set of FDTD simulations to be able to quickly and accurately predict the optical response of candidate geometries.  
Structurally, the predictor consists of the combination of a convolutional neural netork (CNN) based on the ResNet18 architecture 
and a series of fully connected networks (FCN) at the output. The CNN extracts features from an image representing the candidate
geometry and feeds this information to one of the FCNs, which returns the predicted optical response. The choice of FCN depends
on the thickness of the candidate structure, as each FCN has been trained using a specific thickness value.

In a typical search scenario ALFRED begins by launching a swarm of particles equipped with predictor units to quickly explore
the solution space. Once the particles converge to a candidate solution, ALFRED launches a second set of particles around it 
but with the predictor unit removed. These particles then execute full FDTD simulations to refine the candidate into the final
solution structure.

## Limitations of the provided software

The version of ALFRED provided here is a base version of a demo specifically designed to run on a desktop. As this software can be used to produce commercial 
devices, in order to protect the financial interests of the authors the final optimization routine has been removed. Any interested
parties who which to use this software with full optimizations for commercial applications can contact the authors to work out a licensing agreement.


# Getting started

## Requierements

### Hardware

The codes provided are optimized for running on a CUDA capable NVIDIA GPU.
While not strictly required, the user is advised that the neural network training
process can take several hours when running on the GPU and may become prohibitively
long if running on a single CPU. 

### Software

The use of a Linux based operating system is strongly recommended. 
All codes were tested on a Ubuntu 18.04 system.

A working distribution of python 3.8 or higher is required.
The use of the [anaconda python distribution](https://www.anaconda.com/) is recommended
to ease the installation of the required python packages.

Examples of use of this software are provided as Jupyter notebooks and as such 
it requires the [Jupyter notebook](https://jupyter.org/) package. Note that this package
is included by default in the anaconda distribution.


## Initial set up

The usage of an Ubuntu 18.04 system or similar with a CUDA capable GPU and the anaconda python
distribution is assumed for the rest of this document. 

### Obtaining the code

Begin by cloning this project. In a terminal, type:

```sh
$ git clone https://gitlab.kaust.edu.sa/primalight/deepnano
```

### Obtaining the dataset

A large (2 GB) dataset for training ALFRED is maintained as a compressed zip file [here](https://drive.google.com/uc?export=download&id=1nwy3SE8Vstj_AsZ-iMygCsfu4IFi7fAw)

From the terminal, you can download this dataset using the python utility [gdown](https://github.com/wkentaro/gdown)

```bash
$ pip install gdown
$ gdown https://drive.google.com/uc?export=download&id=1nwy3SE8Vstj_AsZ-iMygCsfu4IFi7fAw
```

To extract the zip file, the following command may be used

```bash
$ python -c "from zipfile import PyZipFile; PyZipFile( '''alfred_data.zip''' ).extractall()";
```


### System setup

The use of a separate python virtual environment is recommended for running the provided
programs. The file "deepnano.yml" is provided to quickly setup this environment in Linux
systems. To create an environment using the provided file and activate it do:

```bash
$ cd deepnano
$ conda env create -f deepnano.yml
$ conda activate deepnano
```
Note 'deepnano.yml' is intended to be used only with Linux systems.
Should the user experience problems with this file or be using another system 
a full list of requierements for running the code is available in the file
'requierements.txt' of the repository.

Note 'deepnano.yml' is intended to be used only with Linux systems.
Should the user experience problems with this file or be using another system, 
a full list of requierements for running the code is available in the file
'requierements.txt' of the repository.


To use a Jupyter notebook inside the created virtual environment, type the following code:

```bash
pip install ipykernel ipython kernel install --user --name=deepnano
```
## Usage

Usage instructions are provided in the jupyter notebook files of the repository. The user is advised to first go through the 
file 'Demo.ipynb' as it explains how ALFRED handles data, the training process of the predictor and how to replicate the results
of the manuscript. The notebook can be viewed by executing the following commands:

```bash
$ jupyter notebook Demo.ipynb
```
Please ensure the kernel is the correct one once the notebook starts running.
 
## Citing

When making use of the provided codes in this repository for your own work please ensure you reference the publication 
[Light: Science & Applications volume 10, Article number: 47 (2021)](https://www.nature.com/articles/s41377-021-00489-7).

The following biblatex entry may be used for this purpose.

```
@article{Getman2021,
  title = {Broadband Vectorial Ultrathin Optics with Experimental Efficiency up to 99\% in the Visible Region via Universal Approximators},
  author = {Getman, F. and Makarenko, M. and {Burguete-Lopez}, A. and Fratalocchi, A.},
  year = {2021},
  month = mar,
  volume = {10},
  pages = {1--14},
  publisher = {{Nature Publishing Group}},
  issn = {2047-7538},
  doi = {10.1038/s41377-021-00489-7},
  copyright = {2021 The Author(s)},
  journal = {Light: Science \& Applications},
  language = {en},
  number = {1}
}
```