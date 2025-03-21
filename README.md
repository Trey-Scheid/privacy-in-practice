# 📦 privacy-in-practice
## The Feasibility of Differential Privacy for Telemetry Analysis

This project is a part of an [HDSI](https://datascience.ucsd.edu) Data Science [Capstone Project](https://trey-scheid.github.io/privacy-in-practice/) by Trey, Chris, Tyler, and Bradley. The goal of the project is to privatize common data tasks and compare their performance to non-private baselines. We do our analysis in one domain: Telemetry!

Within you will find we worked off 4 research papers with 4 well known data tasks: computing conditional probabilities, Kmeans clustering, Lasso Regresssion, and Logistic Regression significance tests. They are based off other HDSI research completed on the Intel Telemetry Database. 

Please read our paper [report.pdf](https://github.com/Trey-Scheid/privacy-in-practice/blob/main/report/report.pdf), check out our [poster](https://github.com/Trey-Scheid/privacy-in-practice/blob/main/poster.pdf) and run the code yourself (run.py)!

## Data Access:

The database our analsis was completed on is confidential, so only the resulting plots are available. You may run the project on a synthetic demo dataset with argument `test`


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `conda env create -f environment.yml`
  
### Building the project stages using `run.py`

To run the file, in your terminal, run `python run.py` followed by specific task names, or blank and let task in the run.json config take care of it! Choose however many from:
- `lasso`
- `kmeans`
- `cond_prob`
- `lr_pval`
<!---
[target arguments]`.
- `all` runs `all` targets from scratch (
  - clean → data → build → test
- `test` runs `all` targets on unit test data.
   - clean → data → build → test_build
- `clean` deletes all built files, so that you can build your project from scratch
    - It reverts to a *clean* repository.
- `data` processese raw parquets into processed parquets
- `build` executes code for specified tasks in config from processed data
- `test_build` runs unit tests on models (must be already built)
- `report` runs all but with save figs on and shows output.
-->
* Some of the methods take a while to train to effectively compare the data. If you use no arguments,
the script will run all five methods. If you include one or more arguments, the script will only run
the methods inputted into the target arguments.


```
.
├─ LICENSE
├─ README.md
├─ run.py                <- Catch all function to build project from scratch
│
├─ src                   <- all source code
│  ├─ Task KMEANS        <- each Task corresponds to Privatized Reseach paper
│  │  ├─ run.py          <- build for this task only, called by global run.py
│  │  ├─ inv.ipynb       <- play friendly example investigation of task
│  │  └─ src/            <- ETL and other .py
│  ├─ Task LASSO
│  │  ├─ run.py
│  │  ├─ inv.ipynb
│  │  └─ src/
│  ├─ Task LR_PVAL
│  │  ├─ run.py
│  │  ├─ inv.ipynb
│  │  └─ src/
│  └─ Task COND_PROB
│     ├─ run.py
│     ├─ inv.ipynb
│     └─ src/
│
├─ report                <- Research paper, detailed methods + results
│  ├─ report.pdf
│  └─ report.tex
│
├─ poster.pdf            <- Conference Poster
│
├─ environment.yml       <- Built with conda for executing run.py
│
├─ config/               <- parameters for run.py and each other .py
│
├─ viz
│  ├─ dynamic_output/    <- Stores output plots and tables from run.py
│  └─ static_output/     <- Output plots/tables from Intel Telemetry Data 
│
├─ dummy_data/           <- synthetic (random noise) data according to schema
│  ├─ raw/               <- "read only" parquets used by tasks
│  └─ processed/         <- results of etl processing
│
├─ .github/workflows/nextjs.yml      <- Builds github.io website from branch│
│
└─ deliverable_assets/   <- website imgs, trained models from tasks etc
```

Note that the website is deployed from the website branch.
