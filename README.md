# DataMigrations

## Note

This repository contains code related to the paper "Data Migrations: Exploring the Use of Social Media Data as
Evidence for Human Rights Advocacy". The purpose of this code is to assist the community in their future efforts to analyze disappearance data obtained from social media posts. It consists of scripts for data processing and model-building to classify posts according to whether their subject matter relates to disappearances. Sample notebooks are also provided to assist users with model-building and comparison of predictions against human-labeled samples. 

However, no datasets are provided due to the sensitive and personal nature of the data collected. Due to fact that this code is intended only as a sample for working with similar datasets in the future, it cannot be used to directly replicate the results provided in the paper. 

## Explanation of each file

### Load_Data.py

Can be run from the command-line to parse a set of data files. Takes a series of arguments specifying the locations of various necessary data files - see script for details. Should be run before Crime_Types_Model.ipynb.

### Crime_Types_Model.ipynb

Uses data output from Load_Data.py. Builds and validates two models for predicting the presence and type of a crime in a social media post, and another model for predicting disappearances specifically. 

### Official_Disappearance_Notice_Model.ipynb

Uses raw data downloaded from NodeXL, although it could easily be retooled to use data output from Load_Data.py. Builds and validates a model for classifying whether an image in a social media post contains an official disappearance notification issues by the Mexican government. 

### Caption_scraping_script.py

This script is used to gather image caption data from social media for use in the model.
