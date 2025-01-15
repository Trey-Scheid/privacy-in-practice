# Novel Techniques in Private Data Analysis

This project repo is a part of an HDSI Data Science Capstone prohject (link) by names ____. The goal of the project is to privatize common data tasks using an example domain: Telemetry!

## Data Access:

Much of the data is not accessible to the public domain due to confidentiality. You may run the project on a synthetic demo dataset (follow Demo), or view our resulting plots without running the code (follow 'run.py')


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`


### Demo

* To run the demo on your local machine...

  
### Building the project stages using `run.py`

* To run the file, in your terminal, run `python run.py [target arguments]`.
    * List of target arguments: `gd`, `sgd`, `ftrlm`, `objpert`, `outpert`.
* Some of the methods take a while to train to effectively compare the data. If you use no arguments,
the script will run all five methods. If you include one or more arguments, the script will only run
the methods inputted into the target arguments.
