# EXERCISE for Remote Sensing Course

This repository contains the exercices for the EPFL Remote Sensing Course.

The exercises can be executed directly on the EPFL VMware Horizon platform  `ENAC-SSIE-Ubuntu-20-04` after following these steps:

0. [Download the RS2025 GitHub repository](https://github.com/ltelab/RS2025/archive/refs/heads/main.zip)

1. Unzip the `RS2025-main.zip` file and move the `RS2025` directory wherever you prefer (i.e. within the `/home/ghiggi/courses/` directory).

2. Open a terminal and activate the `lte` environment with:

```sh
micromamba activate lte
```
  
3. Create the `lte` Jupyter Notebook environment with:

```sh
python -m ipykernel install --user --name=lte
```

4. Launch the Jupyter Notebook interface with:

```sh
jupyter notebook
```

----------------

Alternatively, you can clone the [RS2025 repository](https://github.com/ltelab/RS202) on your laptop and install the required environment using conda/mamba or micromamba:  

1. Go to the directory where you want to clone the repository. As an example:

```sh
cd /home/ghiggi/courses
```

2. Clone this repository:

```sh
git clone git@github.com:ltelab/RS2025.git
cd RS2025
```

3. Install the dependencies using conda:

```sh
micromamba env create -f environment.yml
```

3. Activate the `lte` conda environment:

```sh
conda activate lte
```

4. Create the `lte` Jupyter Notebook environment with:

```sh
python -m ipykernel install --user --name=lte
```

6. Launch the Jupyter Notebook interface with:

```sh
jupyter notebook
```

----------------


Note that the installation of the dependencies on your laptop might cause conflicts; in case you encounter such issues and cannot fix them, please contact the TA team.

The latest version of the required packages can be installed using the following command:

```sh
conda install numpy pandas xarray dask rasterio rioxarray scikit-learn matplotlib-base seaborn colorcet pywavelets pillow jupyter
```
