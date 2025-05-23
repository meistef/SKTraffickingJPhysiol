# SK Channel Trafficking

# In-silico modeling of atrial SK channel trafficking.

This model is part of the Journal of Physiology (2025) submission:
"Computational Modeling of the Pro- and Antiarrhythmic Effects of Atrial High Rate-Dependent Trafficking of Small-Conductance Calcium-Activated Potassium Channels"
by Stefan Meier, Dobromir Dobrev, Paul G.A. Volders, Jordi Heijman.

This repository contains the computational framework used to model the high-rate dependent trafficking of SK (small-conductance calcium-activated potassium) channels in atrial cardiomyocytes. The model captures both pro- and antiarrhythmic consequences of altered SK current densities under different pacing conditions.

:file_folder: The [MMT](https://github.com/HeijmanLab/SK-trafficking/tree/main/MMT) folder contains the adapted Human Atrial Cardiomyocyte Model (HACM) with the SK trafficking model and calcium sensor implementation.
 
:computer: :snake: The Python script to create the simulations and figures used in the paper can be found in [SKTraffickingModel](https://github.com/HeijmanLab/SK-trafficking/blob/main/SKTraffickingModel.py) and the 2D simulations can be found in [SK2D_scripts](https://github.com/HeijmanLab/SK-trafficking/blob/main/SK2D_scripts.py).

:computer: :snake: The functions used for the above-mentioned simulations can be found in [SKTraffickingModelFunctions](https://github.com/HeijmanLab/SK-trafficking/blob/main/SKTraffickingModelFunctions.py).


## Virtual environment (Instructions for pip):

Follow the below mentioned steps to re-create te virtual environment with all the correct package versions for this project.

:exclamation: **Before creating a virtual environment please make sure you fully installed Python >3.9.5 (for Linux: -dev version) and myokit (v. 1.37) already. Please follow these steps carefully: http://myokit.org/install.** :exclamation:


***1. Clone the repo:***

https://github.com/HeijmanLab/SK-trafficking.git or git clone git@github.com:HeijmanLab/SK-trafficking.git

***2. Create virtual environment:***

This re-creates a virtual environment with all the correct packages that were used to create the model and run the simulations. 

- Set the directory:

cd SK-trafficking

- Create the virtual environment:

python3 -m venv SK_env

- Activate the environment:

On Windows: SK_env\Scripts\activate

On macOS/Linux: source SK_env/bin/activate

- Install packages from requirements.txt:

pip install -r requirements.txt

***3. Setup and launch spyder (or any other IDE of your liking) from your anaconda prompt:***

spyder
