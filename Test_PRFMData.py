import numpy as np
import h5py
import glob, os, re, sys

from PRFMData import PRFMDataset

ETG_vlM_rot = PRFMDataset(
    galaxy_type="ETG-vlM",
    total_height=0.3, # kpc
    Rmax=2., # kpc
    phibin_sep=np.pi/12., # rad
    snapname="snap-DESPOTIC_100.hdf5",
    realign_galaxy=True, # according to angular momentum vector of gas
    required_particle_types=[0,1,2,3,4], # just gas by default
)

weight = ETG_vlM_rot.get_weight_Rphi()