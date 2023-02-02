# StoragetVET 2.0

StorageVET 2.0 is a valuation model for analysis of energy storage technologies and some other energy resources paired with storage. The tool can be used as a standalone model, or integrated with other power system models, thanks to its open-source Python framework. Download the executable environment and learn more at https://www.storagevet.com.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites & Installing

#### 1. Install [Anaconda](https://www.anaconda.com/download/) for python 3.**

#### 2. Open Anaconda Prompt

#### 3. Activate Python 3.8 environment

    It is recommended that the latest Python 3.8 version be used. As of this writing, that version is Python 3.8.16
    We give the user 2 routes to create a python environment for python 3.8.16
   >Most Windows users have success with the Conda route.

    Each route results in a siloed python environment, but with different properties.
    Choose the conda OR pip route and stick to it. Commands are not interchangeable.
    >Please remember the route which created the python environment in order to activate it again later.
   > **You will need to activate the python environment to run the model, always.**

    **Conda Route - Recommended route for Windows OS**

Note that the python version is specified, meaning conda does not have to be associated with a python 3.8
```
conda create -n storagevet-venv python=3.8.16
conda activate storagevet-venv
```

**Pip Route**

    If you have Python 3.8.16 installed directly on your computer, then we recommend trying this route.
   >This route lets you to open the prompt of your choice.
Note that pip should be associated to a python 3.8 installation

On Linux/Mac

```
pip install virtualenv
virtualenv storagevet-venv
source storagevet-venv/bin/activate
```
On Windows

```
pip install virtualenv
virtualenv storagevet-venv
"./storagevet-venv/Scripts/activate"
```

#### 3. Install project dependencies

**Conda Route**
```
pip install setuptools==52.0.0
conda install conda-forge::blas=*=openblas --file requirements.txt --file requirements-dev.txt
pip install numpy_financial==1.0.0
```

**Pip Route**
```
pip install setuptools==52.0.0
pip install -r requirements.txt -r requirements-dev.txt
pip install numpy_financial==1.0.0
```

## Running the tests

To run tests, activate Python environment. Then enter the following into your terminal:
```
python -m pytest test
```

## Deployment

To use this project as a dependency in your own, clone this repo directly into the root of your project.
Open terminal or command prompt from your project root, and input the following command:
```
pip install -e ./storagevet
```

## Versioning

For the versions available, please
see the [list of releases](https://github.com/epri-dev/StorageVET/releases) on out GitHub repository.
This is version 1.2.3

## Authors

* **Miles Evans**
* **Andres Cortes**
* **Halley Nathwani**
* **Ramakrishnan Ravikumar**
* **Evan Giarta**
* **Thien Nguyen**
* **Micah Botkin-Levy**
* **Yekta Yazar**
* **Kunle Awojinrin**
* **Giovanni Damato**
* **Andrew Etringer**

## License

This project is licensed under the BSD (3-clause) License - see the [LICENSE.txt](./LICENSE.txt) file for details

