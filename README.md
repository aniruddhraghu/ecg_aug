# Getting started

Start off by downloading the PTB-XL dataset from [this link](https://physionet.org/content/ptb-xl/1.0.1/). Then, edit the `PTBXL_PATH` variable in  `ptbxl_dataset.py` to point to where you downloaded the data.

Next, setup the conda environment. Once conda is installed, run: `conda env create -f environment.yml`. Then, you can activate the environment using `conda activate aug`. 

# Training models

## No Augmentations
To train a baseline model on the MI task with no augmentations on GPU 0, run:

```python baseline.py --gpu 0 --task MI ```

See the `baseline.py` script for other configuration options.


## TaskAug
To train a TaskAug model on MI on GPU 0, run:

```python taskaug.py --gpu 0 --task MI```

More configuration options are described  in the `taskaug.py` script.