# StoragetVET 2.0

StorageVET 2.0 is a valuation model for analysis of energy storage technologies and some other energy resources paired with storage. The tool can be used as a standalone model, or integrated with other power system models, thanks to its open-source Python framework. Download the executable environment and learn more at https://www.storagevet.com.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites & Installing

#### 1. Install [Anaconda](https://www.anaconda.com/download/) for python 3.**

#### 2. Open Anaconda Prompt

#### 3. Activate Python 3.6 environment

On Linux/Mac   
Note that pip should be associated to a python 3.6 installation  
```
pip install virtualenv
virtualenv storagevet-venv
source storagevet-venv/bin/activate
```
On Windows  
Note that pip should be associated to a python 3.6 installation    
```
pip install virtualenv
virtualenv storagevet-venv
"./storagevet-venv/Scripts/activate"
```
With Conda
Note that the python version is specified, meaning conda does not have to be associated with a python 3.6
```
conda create -n storagevet-venv python=3.6
conda activate storagevet-venv
```

#### 3. Install project dependencies
 
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
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

We use [Gitlab](https://gitlab.epri.com/storagevet/storagevet) for versioning.
This is version 1.1.0.

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

## License

This project is licensed under the BSD (3-clause) License - see the [LICENSE.txt](./LICENSE.txt) file for details

