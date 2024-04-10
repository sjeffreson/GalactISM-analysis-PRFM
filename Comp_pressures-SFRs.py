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

'''Compute counts and properties of non-ionized gas, to be plotted in Pressures-weights-SFRs_fig.ipynb,
for one galaxy at a time.'''
PROPS = ['count', 'midplane-count', 'Ptot', 'Ptherm', 'Pturb', 'SigmaSFR']

config.read(sys.argv[1])
galname = sys.argv[2]
params = config[galname]
savestring = ""
EXCLUDE_TEMP = params.get('EXCLUDE_TEMP')
if EXCLUDE_TEMP == 'None':
    EXCLUDE_TEMP = None
else:
    EXCLUDE_TEMP = float(EXCLUDE_TEMP)
    savestring += "_T{:.1e}".format(EXCLUDE_TEMP)
EXCLUDE_AVIR = params.get('EXCLUDE_AVIR')
if EXCLUDE_AVIR == 'None':
    EXCLUDE_AVIR = None
else:
    EXCLUDE_AVIR = float(EXCLUDE_AVIR)
    savestring += "_avir{:.1e}".format(EXCLUDE_AVIR)

snapnames = [
    "snap-DESPOTIC_{0:03d}.hdf5".format(i) for i in 
    range(params.getint('BEGSNAPNO'), params.getint('ENDSNAPNO')+1)
]

props_3D = {key: None for key in PROPS}
for snapname in snapnames:
    gal = PRFMDataset(
        params=params,
        galaxy_type=galname,
        total_height=params.getfloat('TOT_HEIGHT'), # kpc
        Rmax=params.getfloat('RMAX'), # kpc
        phibin_sep=np.pi/12.,
        snapname=snapname,
        exclude_temp_above=EXCLUDE_TEMP,
        exclude_avir_below=EXCLUDE_AVIR,
        exclude_HII=True,
        realign_galaxy_to_gas=True, # according to angular momentum vector of gas
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

    '''Save the Rbin_centers as a temporary pickle, if it does not already exist'''
    filesavedir = Path(params['ROOT_DIR']) / params['SUBDIR']
    filesavename = str(filesavedir / "Rbin_centers_{:s}".format(galname)) + savestring + ".pkl"
    if gal.Rbin_centers is not None and not (Path(filesavename)).exists():
        with open(filesavename, "wb") as f:
            pickle.dump(gal.Rbin_centers, f)
        logger.info("Saved: {:s}".format(str(filesavename)))

    '''Save the dictionary to a temporary pickle at regular intervals'''
    if (props_3D[PROPS[-1]].ndim > 2) & ((props_3D[PROPS[-1]].shape[-1] % 25 == 0) | (snapname == snapnames[-1])):
        filesavename = str(filesavedir / "pressures-SFRs-no-HII_{:s}_{:s}".format(
            re.search(r'\d+', snapname).group(), galname
        )) + savestring + ".pkl"
        with open(filesavename, "wb") as f:
            pickle.dump(props_3D, f)
        logger.info("Saved: {:s}".format(str(filesavename)))

    '''Delete the instance of the class to free up memory'''
    del gal