# ohbm2021-nimare-tutorial

[![Binder](https://binder-mcgill.conp.cloud/badge_logo.svg)](https://binder-mcgill.conp.cloud/v2/gh/neurolibre/ohbm2021-nimare-tutorial/main?filepath=notebooks%2Ftutorial.ipynb)
[![Binder Backup](https://img.shields.io/badge/launch-backup--binder-orange.svg)](https://binder.conp.cloud/v2/gh/neurostuff/ohbm2021-nimare-tutorial/main?filepath=notebooks%2Ftutorial.ipynb)

Materials for the OHBM 2021 NiMARE tutorial

## To run this tutorial on Binder

### 1. Follow the Binder badge above

### 2. Run the notebook

## To run this tutorial locally

### 0. Requirements
You must have git, Python3, and bash.


### 1. Clone this repo
In a bash terminal, type
```
git clone https://github.com/neurostuff/ohbm2021-nimare-tutorial.git
```

### 2. Create a virtual environment with the requirements
```
cd ohbm2021-nimare-tutorial
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r binder/requirements.txt
```
### 3. Open the notebook
```bash
jupyter notebook
```
When the Jupyter Notebook page opens in your browser, navigate to `notebooks/tutorial.ipynb`

### 4. Download the data
Run the first code cell to start downloading the data. You can run the cell by putting your cursor somewhere in the cell, then hitting `shift + enter`.
