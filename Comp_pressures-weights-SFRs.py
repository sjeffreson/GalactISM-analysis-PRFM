import numpy as np
import h5py
from PRFMData import PRFMDataset
import astro_helper as ah

import glob, os, re, sys
from pathlib import Path
import configparser
config = configparser.ConfigParser()
import pickle
import argparse, logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''Compute all values for the properties to be plotted in Pressures-weights-SFRs_fig.ipynb,
for one galaxy at a time.'''
PROPS = ['Ptot', 'Ptherm', 'Pturb', 'Weight', 'SigmaSFR']

config.read('config.ini')
galname = sys.argv[1]
params = config[galname]

snapnames = [
    "snap-DESPOTIC_{0:03d}.hdf5".format(i) for i in 
    range(params.getint('BEGSNAPNO'), params.getint('ENDSNAPNO')+1)
]

props_3D = {key: None for key in PROPS}
for snapname in snapnames:
    gal = PRFMDataset(
        galaxy_type=galname,
        total_height=params.getfloat('TOT_HEIGHT'), # kpc
        Rmax=params.getfloat('RMAX'), # kpc
        phibin_sep=np.pi/12.,
        snapname=snapname,
        exclude_temp_above=params.getfloat('EXCLUDE_TEMP'),
        exclude_avir_below=params.getfloat('EXCLUDE_AVIR'),
        realign_galaxy=True, # according to angular momentum vector of gas
        required_particle_types=[0,1,2,3,4], # just gas by default
    )

    for prop in PROPS:
        if props_3D[prop] is None:
            props_3D[prop] = gal.get_prop_by_keyword(prop)
        elif props_3D[prop].ndim == 2:
            props_3D[prop] = np.stack((props_3D[prop], gal.get_prop_by_keyword(prop)), axis=2)
        else:
            props_3D[prop] = np.concatenate((props_3D[prop], gal.get_prop_by_keyword(prop)[:,:,np.newaxis]), axis=2)
        logger.info("Adding {0}, shape of {1}: {2}".format(snapname, prop, props_3D[prop].shape))

    '''Save the dictionary to a temporary pickle at regular intervals'''
    if (props_3D[PROPS[-1]].ndim > 2) & (props_3D[PROPS[-1]].shape[-1] % 25 == 0):
        filesavedir = Path(config['DEFAULT']['ROOT_DIR']) / params['SUBDIR']
        filesavename = filesavedir / "pressures-weights-SFRs_{:s}_{:s}_T{:.1e}_avir{:.1e}.pkl".format(
            re.search(r'\d+', snapnames[0]).group(),
            galname, params.getfloat('EXCLUDE_TEMP'), params.getfloat('EXCLUDE_AVIR')
        )
        with open(filesavename, "wb") as f:
            pickle.dump(props_3D, f)
        logger.info("Saved: {:s}".format(str(filesavename)))