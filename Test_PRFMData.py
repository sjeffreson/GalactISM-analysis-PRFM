import numpy as np
import h5py
import glob, os, re, sys

from PRFMData import PRFMDataset

NGC300 = PRFMDataset(
    galaxy_type="MW",
    total_height=0.3, # kpc
    Rmax=13., # kpc
    phibin_sep=np.pi/12., # rad
    snapname="snap-DESPOTIC_600.hdf5",
    realign_galaxy=True, # according to angular momentum vector of gas
    required_particle_types=[0,1,2,3,4], # just gas by default
)

props = NGC300.get_prop_by_keyword('Weight')
props = NGC300.get_prop_by_keyword('SigmaSFR')