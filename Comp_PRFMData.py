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

'''Compute quantites needed to produce the figures in Jeffreson+ 2024 for one galaxy at a time.'''
config.read(sys.argv[1])
galname = sys.argv[2]
params = config[galname]
PROPS = config['GENERAL']['PROPS'].split(', ')
print(PROPS)
PROPSSTRING = config['GENERAL']['PROPSSTRING']

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
snapstr = params.get('SNAPSTR')

snapnames = [
    snapstr + "_{0:03d}.hdf5".format(i) for i in 
    range(params.getint('BEGSNAPNO'), params.getint('ENDSNAPNO')+1)
]
midplane_idcs_arraynames = glob.glob(str(Path(params['ROOT_DIR']) / params['SUBDIR'] / "weights_*_{:s}.pkl".format(galname)))

props_3D = {key: None for key in PROPS}
for snapname in snapnames:
    '''Open the pickle that stores the mid-plane idces, closest number larger than snapno stores values at time==snapno.'''
    snapno = re.search(r'\d+', snapname).group()
    midplane_idcs_arrayname = min(
        midplane_idcs_arraynames,
        key=lambda x: int(re.search(r'\d+', x.rsplit('/')[-1]).group()) - int(snapno) if
        int(re.search(r'\d+', x.rsplit('/')[-1]).group()) > int(snapno)-1 else np.inf
    )
    midplane_idcs_no = re.search(r'\d+', midplane_idcs_arrayname.rsplit('/')[-1]).group()
    with open(midplane_idcs_arrayname, "rb") as f:
        whole_dict = pickle.load(f)
    timelen_idcs = whole_dict['PtlMinIdcs'].shape[-1]
    midplane_idcs = whole_dict['PtlMinIdcs'][:,:,int(snapno)-1-(int(midplane_idcs_no)-timelen_idcs)]
    del whole_dict

    '''Load the gal, feed it these mid-plane idcs'''
    gal = PRFMDataset(
        params=params,
        galaxy_type=galname,
        total_height=params.getfloat('TOT_HEIGHT'), # kpc
        Rmax=params.getfloat('RMAX'), # kpc
        phibin_sep=np.pi/12.,
        snapname=snapname,
        midplane_idcs=midplane_idcs, # we calculated the mid-plane indices via the minimum of the vertical grav. ptl.
        exclude_temp_above=EXCLUDE_TEMP,
        exclude_avir_below=EXCLUDE_AVIR,
        exclude_HII=True,
        realign_galaxy_to_gas=True, # according to angular momentum vector of gas
    )

    for prop in PROPS:
        if prop in ['Kappa', 'Vcirc', 'Omega']:
            axisnum = 1
        else:
            axisnum = 2
        if props_3D[prop] is None:
            props_3D[prop] = gal.get_prop_by_keyword(prop)
        elif props_3D[prop].ndim == axisnum:
            props_3D[prop] = np.stack((props_3D[prop], gal.get_prop_by_keyword(prop)), axis=axisnum)
        else:
            props_3D[prop] = np.concatenate((props_3D[prop], gal.get_prop_by_keyword(prop)[...,np.newaxis]), axis=axisnum)
        logger.info("Adding {0}, shape of {1}: {2}".format(snapname, prop, props_3D[prop].shape))

    '''Save the Rbin_centers as a temporary pickle, if it does not already exist'''
    filesavedir = Path(params['SAVE_DIR']) / params['SUBDIR']
    filesavename = str(filesavedir / "Rbin_centers_{:s}".format(galname)) + savestring + ".pkl"
    if gal.Rbin_centers is not None and not (Path(filesavename)).exists():
        with open(filesavename, "wb") as f:
            pickle.dump(gal.Rbin_centers, f)
        logger.info("Saved: {:s}".format(str(filesavename)))

    '''Save the dictionary to a temporary pickle at regular intervals'''
    if (props_3D[PROPS[-1]].ndim > axisnum) & ((props_3D[PROPS[-1]].shape[-1] % 25 == 0) | (snapname == snapnames[-1])):
        filesavename = str(filesavedir / PROPSSTRING) + "_{:s}_{:s}".format(
            re.search(r'\d+', snapname).group(), galname
        ) + savestring + ".pkl"
        with open(filesavename, "wb") as f:
            pickle.dump(props_3D, f)
        logger.info("Saved: {:s}".format(str(filesavename)))
        for prop in PROPS: # memory
            del props_3D[prop]
            props_3D[prop] = None
    del gal # memory