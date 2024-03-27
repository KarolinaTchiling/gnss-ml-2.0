# GNSS-ML Project
###### Using ML models to classify GNSS measurements as NLOS and LOS.

### GNSS Dataset
Data is available from here: https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en#Datasets

### Setup a virtual environment

Following commands are for Windows. For macOS/Unix and additional information: 
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

To create a virtual environment, go to your project’s directory and run the following command. This will create a new virtual environment in a local folder named .venv:

	py -m venv .venv

Activate a virtual environment:
	
	.venv\Scripts\activate

###### _Side Note:_

###### If using Powershell and you have the following error:

    \.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. 

###### Enter this command:

    Set-ExecutionPolicy Unrestricted -Scope Process

To confirm the virtual environment is activated:

	where python
    # While the virtual environment is active, the above command will output a 
    # filepath that includes the .venv directory, by ending with the following:
	.venv\Scripts\python


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

### Process Explained
- DataExtract Module: 
  - Creates a new csv from SmartLoc dataset which will include only required information for this study
  - Extracts columns of data from a csv and stores them in a list
- Plots Module:
  - Plots graphs and maps
- Main:
  - Runs the ML modelling

