# Comparing DP Logistic Regression

This project repo highlights several different methods of achieving differentially private logistic regression.
The models covered are **Noisy Gradient Descent**, **DP Stochastic Gradient Descent**, **DP Follow the Regularized Leader**, **Objective Perturbation**, and **Output Perturbation**.

This repo builds several plots for comparing and assessing the privacy and utility of the different methods for differentially private machine learning.


## Retrieving the data locally:

(1) Download the `dataset.csv` data file for satellite telemetry, from the OSSAT-AD dataset here: https://zenodo.org/records/12588359

(2) Edit the file: __config.json__ to include the path/location of the downloaded data in the value of the __fp__ key


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

  
### Building the project stages using `run.py`

* To run the file, in your terminal, run `python run.py [target arguments]`.
    * List of target arguments: `gd`, `sgd`, `ftrlm`, `objpert`, `outpert`.
* Some of the methods take a while to train to effectively compare the data. If you use no arguments,
the script will run all five methods. If you include one or more arguments, the script will only run
the methods inputted into the target arguments.