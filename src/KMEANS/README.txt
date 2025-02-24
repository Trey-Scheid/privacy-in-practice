# Device Usage K-means Clustering

## Project Overview

This project analyzes Intel device usage patterns (non-privatized and privatized). It utilizes K-Means clustering to group devices based on their usage patterns, using standardized Z-scores and L1 distances for clustering.
I reference and replicate the paper PC Health Impact White Paper Tool 2: Clustering Devices Together To Detect Change Patterns which implements normal K-means clustering.
## Table of Contents
```
- [File Structure]
    .
    ├── data                      
    │   ├── out
    │   └── raw
    |        ├── 0007_part_09_limit_1000000 <-sample dataset, will use more parts for final submission
    ├── src
    │   ├── kmeans.py                 <- Data processing and K-means
    │   └── kmeans_private.py         <- Private K-means [TODO]
    ├── requirements.txt              
    └── README.md
    
- [Dependencies](#dependencies)
- [Docker Setup](#docker-setup)
```
## Project Setup

This project is organized into the following components:
- **Source Code**: Located in the `src/` directory. Contains Python classes/scripts for analysis.
- **Data**: Raw and/or processed data can be found in the `data/` directory.
- `run.py` is the entry point for executing the analysis.

## Environment Setup

To recreate the development environment, you can use either **Conda** or **Docker**. 