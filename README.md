# Streamlit based Predictive Process Monitoring powered by LSTM

This project is the created for the purpose of hosting it sharing platform. The original repository is [Dashboard for Emulating LSTM Models over Business Processes and its Qualitative Evaluation](https://github.com/rhnfzl/business-process-dashboard-for-lstm) and more information can be found there. The code has been stripped down to keep it light weight for the server.

## Getting Started

These instructions will help you set up run the project on your local machine.

### Prerequisites

- First install [Anaconda](https://www.anaconda.com/products/individual) on your system.
- Change the directory to desired location where you would like to clone the repo, and then clone it.
- Create Conda virtual Environment using ```conda create -n <env name>```
- Activate the Virtual Env : ```conda activate <env name>```
- Install from the requirement.txt file using ```pip install -r requirements.txt```


## Running the script

Once you've established the environment, you may spin the dashboard using the ```streamlit run dashboard.py```


## App is Hosted on

[Streamlit Share](https://share.streamlit.io/rhnfzl/streamlit-predictive-process-monitoring-dashboard-using-lstm/)


### Known Issues

- In *Single Event Processing* : **Execution Mode** and **What-If Mode** doesn't show results on the hosted dashboard. It seems some issue with [Streamlit Session State API](https://docs.streamlit.io/en/stable/session_state_api.html) (```st.session_state```) communication with multiple ```.py``` files. Although both the modes work on local system.
