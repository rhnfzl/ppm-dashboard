# Streamlit based Predictive Process Monitoring Dashboard for Emulating LSTM Models

This repository has been put together with the intention of hosting. More details may be obtained at [Dashboard for Emulating LSTM Models over Business Processes and its Qualitative Evaluation](https://github.com/rhnfzl/business-process-dashboard-for-lstm), which is the original code repository. The code has been simplified, and features has been stripped down in order to make it as light as possible for the hosting server, it has the next event prediction part using the best LSTM model found by experiments on [Sepsis Event Log](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639/1).

## Getting Started

These instructions will help you setup the project on your local machine.

#### Prerequisites

- First install [Anaconda](https://www.anaconda.com/products/individual) on your system.
- Change the directory to desired location where you would like to clone the repo, and then clone it.
- Create Conda virtual Environment using ```conda create -n <env name>```
- Activate the Virtual Env : ```conda activate <env name>```
- Install from the [requirement.txt](https://github.com/rhnfzl/streamlit-predictive-process-monitoring-dashboard-using-lstm/blob/master/requirements.txt) file using ```pip install -r requirements.txt``` from the root directory.

## Running the script

Once you've established the environment, you should be able to run the dashboard using the ```streamlit run dashboard.py``` from the root directory.

## Application is Hosted on

[Streamlit Share](https://share.streamlit.io/rhnfzl/ppm-dashboard/dashboard.py)

## Known Issues

- In *Single Event Processing* : **Execution Mode** and **What-If Mode** doesn't show results on the hosted dashboard. It seems some issue with [Streamlit Session State API](https://docs.streamlit.io/en/stable/session_state_api.html) (```st.session_state```) communication with multiple ```.py``` files. Although both the modes work on local system setup along with other features.
- Can't be deployed on Heroku because it is exceeding the [Slug Size](https://devcenter.heroku.com/articles/slug-compiler#slug-size) of 500 MB (Compression Error), although necessary files has been added for the deployment.
- Under Batch - Pre-select prefix mode the *Generative* is not working on the streamlit share, but working on local system setup.

## Future Enhancement

- Adding more models for different publicly available datasets.
- Incorporating KPI's.
- Adding tree based classifier to predict the labels after provided number of events of the trace.
- Single activity based execution based on event stream along with individual case id.
