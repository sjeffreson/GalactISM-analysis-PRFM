# Post-processed GalactISM simulation data
This repository contains all methods required to post-process the data from the GalactISM simulations, and to reproduce Figures 10-14 of [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract). These are the figures validating the pressure-regulated, feedback-modulated (PRFM) theory of star formation, as presented in [Ostriker & Kim 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ%E2%80%A6936..137O/abstract). A full description of the simulations and their production can be found in [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract), along with the equations and descriptions for each of the post-processed quantities.

### Dataset download
The final post-processed dataset can be found [on Kaggle](https://www.kaggle.com/datasets/sarahjeffreson/galactism-analysis-prfm), along with a description of its structure.

### Repository structure
The following files are used to post-process the raw simulation data (outputs from the moving-mesh code [Arepo](https://ui.adsabs.harvard.edu/abs/2010MNRAS.401..791S/abstract)). They can be _ignored_ for the purposes of reproducing the Figures, as the post-processed data can simply be downloaded from Kaggle. However, the code is well-commented and can be used to post-process any other isolated galaxy simulations produced in Arepo:
* `PRFMData.py`: Class containing all post-processing functions.
* `Comp_PRFMData.py`: Uses the PRFMData class to compute all quantities relevant to PRFM, and store them in batches.
* `Jeffreson+LtU_24b/Combine_data.ipynb`: Combines the batches of data produced by `Comp_PRFMData.py` to produce the dataset published on Kaggle.
* `Jeffreson+LtU_24b/Check_PRFMData.ipynb`: Test notebook for the PRFMData functions.
* `Jeffreson+LtU_24b/Check_weight-z-dstbn.ipynb`: Test notebook for the PRFMData functions, specifically the computation of the galactic mid-plane.
* `Jeffreson+LtU_24b/Check_asymmetric-forces-2D.ipynb`: Test notebook for the PRFMData functions, specifically the symmetry of the forces acting at the galactic mid-plane.
* `Jeffreson+LtU_24b/config/*`: Config files for the post-processing of the Arepo data.
* `Jeffreson+LtU_24b/submit/*`: Example SLURM submission files for the post-processing of the Arepo data.

The remaining files in the `Jeffreson+LtU_24b` directory are then used to produce Figures 10-14 in [Jeffreson et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240909114J/abstract). Any external data used for these figures is stored at `Jeffreson+LtU_24b/ext-data`. Each notebook is labeled according to its Figure number.
