# Post-processed GalactISM simulation data
This repository contains all methods required to post-process the data from the GalactISM simulations, and to reproduce Figures 10-14 of [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract). These are the figures validating the pressure-regulated, feedback-modulated (PRFM) theory of star formation, as presented in [Ostriker & Kim 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ%E2%80%A6936..137O/abstract). A full description of the simulations and their production can be found in [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract), along with the equations and descriptions for each of the post-processed quantities.

## Dataset download
The post-processed dataset can be found [on Kaggle](https://www.kaggle.com/datasets/sarahjeffreson/galactism-analysis-prfm), along with a description of its structure.

## Repository structure
The following files are used to post-process the raw simulation data (outputs from the moving-mesh code [Arepo](https://ui.adsabs.harvard.edu/abs/2010MNRAS.401..791S/abstract)). They can be _ignored_ for the purposes of reproducing the Figures, as the post-processed data can simply be downloaded from Kaggle. However, the code is well-commented and can be used to post-process any other isolated galaxy simulations produced in Arepo. If you prefer to do this yourself, then you will use the following:
* `PRFMData.py`: Class containing all post-processing functions.
* `Comp_PRFMData.py`: Interface between config files and PRFMData class.
* `Jeffreson+LtU_24b/config/*`: Example config files for various subsets of quantities you may want to compute.
* `Jeffreson+LtU_24b/submit/*`: Example SLURM submission files for running the post-processing code.

The code is designed to be flexible (can be used to compute any subset of quantities you would like). However, if you want to define the galactic mid-plane via the minimum of the gravitational potential (rather than simply setting it to z=0), you MUST complete a run with the `Jeffreson+LtU_24b/config/config_weights.ini` config file, before calculating anything else. This will compute the z-indices of the potential minimum.

## Creating your own config files
The config file options are as follows:
* `EXCLUDE_TEMP` (K): Temperature above which to exclude gas (20,000K for cool-warm ISM). If set to 'None', no gas is excluded.
* `EXCLUDE_AVIR`: Virial parameter below which to exclude gas (2 for cool-warm ISM). If set to 'None', no gas is excluded.
* `TOT_HEIGHT` (kpc): Gas at this distance above/below the galactic mid-plane is excluded to speed computation.
* `RMAX` (kpc): Gas outside this galactocentric radius is excluded to speed computation.
* `MIDPLANEIDCS`: Set to `True` if you have already computed mid-plane indices according to the potential minimum, which you want to use. Set to False if you wish to define the mid-plane at z=0, OR you are computing the mid-plane indices themselves.
* `BEGSNAPNO`: First snapshot number to analyze
* `ENDSNAPNO`: Last snapshot number to analyze
* `ROOT_DIR`: Your data root directory
* `SAVE_DIR`: Your root directory for saving new, post-processed data
* `SUBDIR`: The subdirectory for any one galaxy, should be INSIDE both `ROOT_DIR` and `SAVE_DIR`
* `SNAPSTR`: The string (excluding numbers) defining each snapshot. If in standard Arepo format, this will just be `snap`.

## Test notebooks
There are three notebooks to test whether the `PRFMData` class is working as expected on your galaxy:
* `Jeffreson+LtU_24b/Check_PRFMData.ipynb`: Test notebook for the PRFMData functions.
* `Jeffreson+LtU_24b/Check_weight-z-dstbn.ipynb`: Test notebook for the PRFMData functions, specifically the computation of the galactic mid-plane.
* `Jeffreson+LtU_24b/Check_asymmetric-forces-2D.ipynb`: Test notebook for the PRFMData functions, specifically the symmetry of the forces acting at the galactic mid-plane.

## Notebook to combine data produced in different post-processing runs
If you have executed several post-processing runs for different quantities and you wish to combine them, you may do so with:
* `Jeffreson+LtU_24b/Combine_data.ipynb`: Combines the batches of data produced by `Comp_PRFMData.py` to produce the dataset published on Kaggle.

## Notebooks to produce figures from Jeffreson+ LtU, 2024b
The remaining files in the `Jeffreson+LtU_24b` directory are used to produce Figures 10-14 in [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract). Any external data used for these figures is stored at `Jeffreson+LtU_24b/ext-data`. Each notebook is labeled according to its Figure number.
