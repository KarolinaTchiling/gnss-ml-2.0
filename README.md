# gnss-ml2.0

## Setup a virtual environment

Following commands are for Windows. For macOS/Unix and additional information: 
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

_If using Powershell and you have the following error:_

    \.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. 

_Enter this command:_

    Set-ExecutionPolicy Unrestricted -Scope Process


To create a virtual environment, go to your project’s directory and run the following command. This will create a new virtual environment in a local folder named .venv:

	py -m venv .venv

Activate a virtual environment:
	
	.venv\Scripts\activate

To confirm the virtual environment is activated:

	where python

While the virtual environment is active, the above command will output a filepath that includes the .venv directory, by ending with the following:

	.venv\Scripts\python


Prepare pip and make sure it is up to date:

	py -m pip install --upgrade pip
	py -m pip --version

Install packages using pip:

    pip install -U scikit-learn
    pip install -U matplotlib
    

To confirm successful installation, check all packages installed in the active virtual env:

    python -m pip freeze


When finished with installs, deactivate the virtual environment:

	deactivate


