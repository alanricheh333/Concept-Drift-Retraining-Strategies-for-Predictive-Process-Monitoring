# Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring
The implementation of the thesis Concept Drift Driven Retraining strategies for Predictive Process Monitoring.


1. Clone the repository and enter it.

	1.1 Either you clone with ssh key (you need to generate an ssh key please follow the steps here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
	and add it to your account settings. Then clone and enter the repo:

    ```
    git clone git@github.com:alanricheh333/Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring.git
    cd Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring
    ```
	1.2 Or you can clone by https.

2. Set up a virtual environment for this project, and activate it:

    ```
    python -m venv venv
    # On macOS / Linux / "Git Bash" on Windows:
    source venv/bin/activate
    # On Windows (PowerShell)
    .\venv\Scripts\activate.ps1
    ```
   
   You may have to replace `python` with the absolute path to your interpreter, in case you have several interpreters
   installed and `python` does not point to the right one.

3. Install all required python packages with pip.

    # For windows or linux
    ```
    pip install -r requirements.txt
    ```
    # For mac
    ```
    pip install -r requirements-max.txt
    ```

    - If you're using Mac M1/M2 then you need to follow the instructions here to install tensorflow: https://www.youtube.com/watch?v=5DgWvU0p2bk&ab_channel=JeffHeaton .
    - If you're using Mac then you need to follow these steps to install CVXOPT properly:
        ```
        pip uninstall cvxopt
        export CVXOPT_BUILD_GLPK=1
        pip install cvxopt --no-binary cvxopt
        ```
	
4. Install the ProDrift2.5.jar tool in order to run the drift detection method:
    - Go to: `https://drive.google.com/file/d/1eiG2ShFukTkjlzbcSrMYOZ9wwPO8uVQM/view?usp=sharing`
    - Install the ProDrift2.5.jar
    - Add the file to this folder: `/Concept-Drift-Retraining-Strategies-for-Predictive-Process-Monitoring/core/concept_drift/`
    - Then you can run the method smoothly


## Running the project
To run the project please follow these steps:
1. Activate the virtual environemnt:
	```
	# On macOS / Linux / "Git Bash" on Windows:
   	source venv/bin/activate
   	# On Windows (PowerShell)
   	.\venv\Scripts\activate.ps1
	```

2. Specify the function you want to run from main.py.

3. Run the project with the command:
    ```
	python main.py
	```

4. You can also call the functions from the cli.py.
    ```
    python cli.py {{the function to be used from cli.py}}
    ```