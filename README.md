# GNSS-ML Project
###### Using ML models to classify GNSS measurements as NLOS and LOS.

### GNSS Dataset
Data is available from here: https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en#Datasets

### Project Setup
1) Clone the repository
2) Setup virtual environment and install necessary libraries
3) Run main.py (I used python 3.12)
   - All raw data and processed data has been included in this repo
   - Raw data was processed in dataExtract.py

    
### Setup a virtual environment

Following commands are for Windows. For macOS/Unix and additional information: 
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

To create a virtual environment, go to your project’s directory and run the following command. This will create a new virtual environment in a local folder named .venv:

	py -m venv .venv

Activate a virtual environment:
	
	.venv\Scripts\activate


Prepare pip and make sure it is up-to-date:

	py -m pip install --upgrade pip
	py -m pip --version

Install packages using pip:

    pip install pandas
    pip install -U scikit-learn
    pip install -U matplotlib
    pip install plotly    #if you wish to map
    

To confirm successful installation, check all packages installed in the active virtual env:

    python -m pip freeze


When finished with installs, deactivate the virtual environment:

	deactivate



