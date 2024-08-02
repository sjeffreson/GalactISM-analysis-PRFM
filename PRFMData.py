from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from scipy.signal import savgol_filter as sg
from scipy.stats import binned_statistic_2d
from rbf.interpolate import RBFInterpolant

import h5py
import astro_helper as ah

import configparser
import argparse, logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PRFMDataset:
    '''Calculate physical properties for validation of pressure-regulated star formation,
    using Arepo simulation data, in bins of R and phi (cylindrical coordinates).'''

    def __init__(
        self,
        params: configparser.SectionProxy,
        galaxy_type: str,
        total_height: float = 1.5, # kpc
        Rmax: float = 15., # kpc
        Rmin: float = 0.3, # kpc
        Rbin_width: float = 0.5, # kpc
        Rbin_sep: float = 0.2, # kpc
        phibin_sep: float = np.pi/4., # rad
        zbin_sep: float = 10., # pc
        zbin_sep_ptl: float = 10., # pc, for computation of the potential
        exclude_temp_above: float = None, # K
        exclude_avir_below: float = None, # virial parameter
        exclude_HII: bool = False, # whether to completely exclude ionized gas
        snapname: str = "snap-DESPOTIC_300.hdf5",
        midplane_idcs: np.array = None, # if mid-plane already known
        realign_galaxy_to_gas: bool=True, # according to angular momentum vector of gas
        realign_galaxy_to_disk: bool=False, # according to angular momentum vector of entire disk system
        required_particle_types: List[int] = [0], # just gas by default
    ):
        self.ROOT_DIR = Path(params['ROOT_DIR'])
        self.galaxy_type = galaxy_type
        self.galaxy_dir = params['SUBDIR']

        self.total_height = total_height * ah.kpc_to_cm
        self.Rmax = Rmax * ah.kpc_to_cm
        self.Rmin = Rmin * ah.kpc_to_cm
        self.Rbin_width = Rbin_width * ah.kpc_to_cm
        self.Rbin_sep = Rbin_sep * ah.kpc_to_cm
        self.phibin_sep = phibin_sep
        self.zbin_sep = zbin_sep * ah.pc_to_cm
        self.zbin_sep_ptl = zbin_sep_ptl * ah.pc_to_cm
        self.exclude_temp_above = exclude_temp_above
        self.exclude_avir_below = exclude_avir_below
        self.exclude_HII = exclude_HII
        self.snapname = snapname
        self.midplane_idcs = midplane_idcs
        self.realign_galaxy_to_gas = realign_galaxy_to_gas
        self.realign_galaxy_to_disk = realign_galaxy_to_disk

        '''Load all required data'''
        self.data = {}
        for part_type in required_particle_types:
            self.data[part_type] = self._read_snap_data(part_type)

        '''Sixth PartType for all stars'''
        present_stellar_types = [key for key, value in self.data.items() if key in [2, 3, 4] and value is not None]
        if len(present_stellar_types) > 0:
            self.data[5] = {key: np.concatenate([self.data[i][key] for i in present_stellar_types]) for key in self.data[present_stellar_types[0]]}

        '''Seventh PartType for gas that's cut to parameter thresholds, create new variable
        for this, as we don't want to cut out these gas cells for every method.'''
        self.data[6] = None
        cnd = np.ones(len(self.data[0]["R_coords"]), dtype=bool)
        if exclude_temp_above is not None:
            cnd = cnd & (self.data[0]["temps"] < exclude_temp_above)
        if exclude_avir_below is not None:
            cnd = cnd & ~((self.data[0]["AlphaVir"] < exclude_avir_below) & (self.data[0]["AlphaVir"] > 0.))
        self.data[6] = {key: value[cnd] for key, value in self.data[0].items()}
        if exclude_HII:
            self.data[6]["masses"] = self.data[6]["masses"] * (1. - self.data[6]["xHP"])
            self.data[6]["Density"] = self.data[6]["Density"] * (1. - self.data[6]["xHP"])

        '''Realign the galaxy according to the gas or gas+stellar disk'''
        if self.realign_galaxy_to_gas & self.realign_galaxy_to_disk:
            raise ValueError("Galaxy cannot be realigned to both gas and disk. Please choose one.")
        if self.realign_galaxy_to_gas:
            if 0 not in self.data:
                raise ValueError("Gas data must be loaded to realign the galaxy to the gas.")
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self._get_gas_disk_COM()
            self.Lx, self.Ly, self.Lz = self._get_gas_disk_angmom()
            self.data = {key: self._set_realign_galaxy(value) for key, value in self.data.items()}
        elif self.realign_galaxy_to_disk:
            if 0 not in self.data or 2 not in self.data:
                raise ValueError("Gas and stellar disk particles must be loaded to realign the galaxy to the disk.")
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self._get_disk_COM()
            self.Lx, self.Ly, self.Lz = self._get_disk_angmom()
            self.data = {key: self._set_realign_galaxy(value) for key, value in self.data.items()}

        '''Grid on which to compute arrays'''
        self.Rbinno = int(np.rint((self.Rmax-self.Rmin)/(self.Rbin_sep)))
        self.Rbin_edges = np.linspace(self.Rmin, self.Rmax, self.Rbinno+1)
        self.Rbin_centers = (self.Rbin_edges[1:]+self.Rbin_edges[:-1])/2.
        self.Rbin_mins = self.Rbin_centers - self.Rbin_width/2.
        self.Rbin_maxs = self.Rbin_centers + self.Rbin_width/2.

        self.phibinno = int(np.rint(2.*np.pi/self.phibin_sep))
        self.phibin_edges = np.linspace(-np.pi, np.pi, self.phibinno+1)
        self.phibin_centers = (self.phibin_edges[1:]+self.phibin_edges[:-1])/2.

        self.Rbin_centers_2d, self.phibin_centers_2d = np.meshgrid(self.Rbin_centers, self.phibin_centers)
        self.Rbin_centers_2d = self.Rbin_centers_2d.T
        self.phibin_centers_2d = self.phibin_centers_2d.T

        self.zbinno = int(np.rint(2.*self.total_height/self.zbin_sep))
        self.zbin_edges = np.linspace(-self.total_height, self.total_height, self.zbinno+1)
        self.zbin_centers = (self.zbin_edges[1:]+self.zbin_edges[:-1])/2.

        '''Finer z-grid for the potential'''
        self.zbinno_ptl = int(np.rint(2.*self.total_height/self.zbin_sep_ptl))
        self.zbin_edges_ptl = np.linspace(-self.total_height, self.total_height, self.zbinno_ptl+1)
        self.zbin_centers_ptl = (self.zbin_edges_ptl[1:]+self.zbin_edges_ptl[:-1])/2.
        self.Rbin_centers_3d_ptl, self.phibin_centers_3d_ptl, self.zbin_centers_3d_ptl = np.meshgrid(self.Rbin_centers, self.phibin_centers, self.zbin_centers_ptl)
        self.Rbin_centers_3d_ptl = np.transpose(self.Rbin_centers_3d_ptl, axes=(1, 0, 2))
        self.phibin_centers_3d_ptl = np.transpose(self.phibin_centers_3d_ptl, axes=(1, 0, 2))
        self.zbin_centers_3d_ptl = np.transpose(self.zbin_centers_3d_ptl, axes=(1, 0, 2))

        '''Keywords to access methods'''
        self._cached_force = None
        self._method_map = {
            'count': lambda: self.get_count_Rphi(PartType=6),
            'midplane-count': lambda: self.get_count_midplane_Rphi(PartType=6),
            'midplane-dens': lambda: self.get_gas_midplane_density_Rphi(PartType=6),
            'Ptherm': lambda: self.get_gas_midplane_thermpress_Rphi(PartType=6) / ah.kB_cgs,
            'Pturb': lambda: self.get_gas_midplane_turbpress_Rphi(PartType=6),
            'Ptot': lambda: self.get_gas_midplane_thermpress_Rphi(PartType=6) + self.get_gas_midplane_turbpress_Rphi(PartType=6),
            'Weight': lambda: self.get_weight_Rphi() / ah.kB_cgs,
            'Force': lambda: self.get_force_Rphi() / ah.kB_cgs,
            'ForceLeft': lambda: self._get_cached_force()[0] / ah.kB_cgs,
            'ForceRight': lambda: self._get_cached_force()[1] / ah.kB_cgs,
            'PtlMinIdcs': lambda: self._get_cached_force()[2],
            'SigmaSFR': lambda: self.get_SFR_surfdens_Rphi() / ah.Msol_to_g * ah.kpc_to_cm**2 * ah.yr_to_s,
            'SigmaGas': lambda: self.get_surfdens_Rphi(PartType=6) / ah.Msol_to_g * ah.pc_to_cm**2,
            'SigmaH2Gas': lambda: self.get_component_surfdens_Rphi(component='xH2') / ah.Msol_to_g * ah.pc_to_cm**2,
            'SigmaStar': lambda: self.get_surfdens_Rphi(PartType=5) / ah.Msol_to_g * ah.pc_to_cm**2,
            'Omega': lambda: self.get_Omegaz_R() * ah.Myr_to_s,
            'Kappa': lambda: self.get_kappa_R() * ah.Myr_to_s,
            'Vcirc': lambda: self.get_rotation_curve_R() / ah.kms_to_cms
        }

    def get_prop_by_keyword(self, keyword: str) -> np.array:
        '''Get the physical property array by keyword'''

        if keyword not in self._method_map:
            raise ValueError("Keyword {:s} not found in method map.".format(keyword))
        return self._method_map[keyword]()

    def _get_cached_force(self):
        '''Avoid recomputation of this expensive method'''
        if self._cached_force is None:
            self._cached_force = self.get_int_force_left_right_Rphi()
        return self._cached_force

    def _read_snap_data(
        self,
        PartType: int,
    ) -> Dict[str, np.array]:
        """Read necessary information about a given particle type, from Arepo snapshot
        Args:
            snapshot (int): snapshot number
            PartType (int): particle type, as in Arepo snapshot
        Returns:
            Dict: dictionary with only the relevant gas information, in cgs units
        """
        snapshot = h5py.File(self.ROOT_DIR / self.galaxy_dir / self.snapname, "r")
        header = snapshot["Header"]
        if "PartType"+str(PartType) not in snapshot:
            return None
        else:
            PartType_data = snapshot["PartType"+str(PartType)]

        snap_data = {}
        snap_data["x_coords"] = (PartType_data['Coordinates'][:,0] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["y_coords"] = (PartType_data['Coordinates'][:,1] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["R_coords"] = np.sqrt(snap_data["x_coords"]**2 + snap_data["y_coords"]**2)
        snap_data["phi_coords"] = np.arctan2(snap_data["y_coords"], snap_data["x_coords"])
        snap_data["z_coords"] = (PartType_data['Coordinates'][:,2] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["velxs"] = PartType_data['Velocities'][:,0] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["velys"] = PartType_data['Velocities'][:,1] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["velzs"] = PartType_data['Velocities'][:,2] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["Potential"] = PartType_data['Potential'][:] * PartType_data['Potential'].attrs['to_cgs']
        if 'Masses' in PartType_data:
            snap_data["masses"] = PartType_data['Masses'][:] * PartType_data['Masses'].attrs['to_cgs']
        else:
            snap_data["masses"] = np.ones(len(snap_data["x_coords"])) * header.attrs['MassTable'][PartType] * snapshot['PartType0/Masses'].attrs['to_cgs']
        if PartType != 0:
            snapshot.close()
            return snap_data
        else:
            # standard Arepo
            snap_data["U"] = PartType_data['InternalEnergy'][:] * PartType_data['InternalEnergy'].attrs['to_cgs']
            snap_data["temps"] = (ah.gamma - 1.) * snap_data["U"] / ah.kB_cgs * ah.mu * ah.mp_cgs
            snap_data["Density"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["SFRs"] = PartType_data['StarFormationRate'][:] * PartType_data['StarFormationRate'].attrs['to_cgs']
            # specific to Jeffreson et al. runs
            try:
                snap_data["xH2"] = PartType_data['ChemicalAbundances'][:,0] * 2.
                snap_data["xHP"] = PartType_data['ChemicalAbundances'][:,1]
                snap_data["xHI"] = 1. - snap_data["xH2"] - snap_data["xHP"]
                snap_data["AlphaVir"] = PartType_data['AlphaVir'][:]
            except KeyError:
                snap_data["xH2"] = np.zeros(len(snap_data["x_coords"]))
                snap_data["xHP"] = np.zeros(len(snap_data["x_coords"]))
                snap_data["xHI"] = np.ones(len(snap_data["x_coords"]))
                snap_data["AlphaVir"] = np.zeros(len(snap_data["x_coords"]))
            snapshot.close()
            return snap_data
    
    def _cut_out_particles(self, PartType: int=0) -> Dict[str, np.array]:
        '''Cut out most of the gas cells that are in the background grid, not the disk'''

        cnd = (self.data[PartType]["R_coords"] < self.Rmax) & (np.fabs(self.data[PartType]["z_coords"]) < self.Rmax)
        return {key: value[cnd] for key, value in self.data[PartType].items()}

    def _get_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas/stellar disk.'''

        gasstar_data = {key: np.concatenate([self.data[2][key], self._cut_out_particles(PartType=0)[key]]) for key in self.data[2]}

        x_CM = np.average(gasstar_data["x_coords"], weights=gasstar_data["masses"])
        y_CM = np.average(gasstar_data["y_coords"], weights=gasstar_data["masses"])
        z_CM = np.average(gasstar_data["z_coords"], weights=gasstar_data["masses"])
        vx_CM = np.average(gasstar_data["velxs"], weights=gasstar_data["masses"])
        vy_CM = np.average(gasstar_data["velys"], weights=gasstar_data["masses"])
        vz_CM = np.average(gasstar_data["velzs"], weights=gasstar_data["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def _get_gas_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas disk'''

        gas_data_cut = self._cut_out_particles(PartType=0)

        x_CM = np.average(gas_data_cut["x_coords"], weights=gas_data_cut["masses"])
        y_CM = np.average(gas_data_cut["y_coords"], weights=gas_data_cut["masses"])
        z_CM = np.average(gas_data_cut["z_coords"], weights=gas_data_cut["masses"])
        vx_CM = np.average(gas_data_cut["velxs"], weights=gas_data_cut["masses"])
        vy_CM = np.average(gas_data_cut["velys"], weights=gas_data_cut["masses"])
        vz_CM = np.average(gas_data_cut["velzs"], weights=gas_data_cut["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def _get_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the disk'''

        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self._get_disk_COM()
        gasstar_data = {key: np.concatenate([self.data[2][key], self._cut_out_particles(PartType=0)[key]]) for key in self.data[2]}

        Lx = np.sum(
            gasstar_data["masses"]*((gasstar_data["y_coords"]-y_CM)*(gasstar_data["velzs"]-vz_CM) -
            (gasstar_data["z_coords"]-z_CM)*(gasstar_data["velys"]-vy_CM))
        )
        Ly = np.sum(
            gasstar_data["masses"]*((gasstar_data["z_coords"]-z_CM)*(gasstar_data["velxs"]-vx_CM) -
            (gasstar_data["x_coords"]-x_CM)*(gasstar_data["velzs"]-vz_CM))
        )
        Lz = np.sum(
            gasstar_data["masses"]*((gasstar_data["x_coords"]-x_CM)*(gasstar_data["velys"]-vy_CM) -
            (gasstar_data["y_coords"]-y_CM)*(gasstar_data["velxs"]-vx_CM))
        )
        return Lx, Ly, Lz

    def _get_gas_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the gas disk'''

        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self._get_gas_disk_COM()
        gas_data_cut = self._cut_out_particles(PartType=0)

        Lx = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["y_coords"]-y_CM)*(gas_data_cut["velzs"]-vz_CM) -
            (gas_data_cut["z_coords"]-z_CM)*(gas_data_cut["velys"]-vy_CM))
        )
        Ly = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["z_coords"]-z_CM)*(gas_data_cut["velxs"]-vx_CM) -
            (gas_data_cut["x_coords"]-x_CM)*(gas_data_cut["velzs"]-vz_CM))
        )
        Lz = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["x_coords"]-x_CM)*(gas_data_cut["velys"]-vy_CM) -
            (gas_data_cut["y_coords"]-y_CM)*(gas_data_cut["velxs"]-vx_CM))
        )
        return Lx, Ly, Lz

    def _set_realign_galaxy(self, snap_data: Dict[str, np.array]) -> Dict[str, np.array]:
        '''Realign the galaxy according to the center of mass and the angular momentum
        vector of the gas disk'''

        if snap_data is None:
            return None

        # new unit vectors
        zu = np.array([self.Lx, self.Ly, self.Lz])/np.sqrt(self.Lx**2+self.Ly**2+self.Lz**2)
        xu = np.array([-self.Ly, self.Lx, 0.]/np.sqrt(self.Lx**2+self.Ly**2))
        yu = np.array([-self.Lx*self.Lz, -self.Ly*self.Lz, self.Lx**2+self.Ly**2])/np.sqrt((-self.Lx*self.Lz)**2+(-self.Ly*self.Lz)**2+(self.Lx**2+self.Ly**2)**2)

        # new co-ordinates
        x = snap_data["x_coords"] - self.x_CM
        y = snap_data["y_coords"] - self.y_CM
        z = snap_data["z_coords"] - self.z_CM
        vx = snap_data["velxs"] - self.vx_CM
        vy = snap_data["velys"] - self.vy_CM
        vz = snap_data["velzs"] - self.vz_CM
        snap_data['x_coords'] = xu[0]*x + xu[1]*y + xu[2]*z
        snap_data['y_coords'] = yu[0]*x + yu[1]*y + yu[2]*z
        snap_data['R_coords'] = np.sqrt(snap_data['x_coords']**2 + snap_data['y_coords']**2)
        snap_data['z_coords'] = zu[0]*x + zu[1]*y + zu[2]*z
        snap_data['velxs'] = xu[0]*vx + xu[1]*vy + xu[2]*vz
        snap_data['velys'] = yu[0]*vx + yu[1]*vy + yu[2]*vz
        snap_data['velzs'] = zu[0]*vx + zu[1]*vy + zu[2]*vz

        snap_data['R_coords'] = np.sqrt(snap_data['x_coords']**2 + snap_data['y_coords']**2)
        snap_data['phi_coords'] = np.arctan2(snap_data['y_coords'], snap_data['x_coords'])

        return snap_data

    def get_data(self) -> Dict[int, Dict[str, np.array]]:
        return self.data

    def get_grid(self) -> Tuple[np.array, np.array, np.array]:
        return self.Rbin_centers, self.phibin_centers, self.zbin_centers

    def get_keys(self):
            for key in vars(self).keys():
                print(key)

    def get_rotation_curve_R(self) -> np.array:
        '''Get the 1D rotation curve of the galaxy, within the gas disk, in cm/s.'''

        vcs = []
        for Rbmin, Rbmax in zip(self.Rbin_edges[:-1], self.Rbin_edges[1:]):
            cnd = (self.data[0]['R_coords'] > Rbmin) & (self.data[0]['R_coords'] < Rbmax)
            if(len(self.data[0]['R_coords'][cnd])>0):
                vcs.append(np.average(
                    -self.data[0]['y_coords'][cnd]/self.data[0]['R_coords'][cnd] * self.data[0]['velxs'][cnd] +
                    self.data[0]['x_coords'][cnd]/self.data[0]['R_coords'][cnd] * self.data[0]['velys'][cnd],
                    weights=self.data[0]['masses'][cnd]))
            else:
                vcs.append(np.nan)

        return np.array(vcs)

    def get_Omegaz_R(self) -> np.array:
        '''Get the 1D galactic angular velocity in the z-direction, in /s.'''

        vcs = self.get_rotation_curve_R()
        Omegazs = vcs / self.Rbin_centers

        return Omegazs
    
    def get_kappa_R(self, polyno: int=2, wndwlen: int=5) -> np.array:
        '''Get the 1D epicyclic frequency in the z-direction, in /s.'''

        vcs = self.get_rotation_curve_R()
        Omegazs = vcs / self.Rbin_centers

        dR = sg(self.Rbin_centers, wndwlen, polyno, deriv=1)
        dvc = sg(vcs, wndwlen, polyno, deriv=1)
        betas = dvc/dR * self.Rbin_centers/vcs
        kappas = Omegazs * np.sqrt(2.*(1.+betas))

        return kappas

    def get_density_Rphiz(self, zbinsep: float=None, PartType: int=None) -> np.array:
        '''Get the 3D mid-plane density in cgs'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_density_Rphiz.")
        if zbinsep==None:
            zbinsep = self.zbin_sep

        densities = np.zeros((self.Rbinno, self.phibinno, self.zbinno)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                densities[:,:,k] = np.zeros((self.Rbinno, self.phibinno)) * np.nan
                continue
            dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            densities[:,:,k] = dens / (self.Rbin_width * self.zbin_sep * self.Rbin_centers_2d*self.phibin_sep)

        return densities

    def get_gas_midplane_density_Rphi(self, PartType: int=None) -> np.array:
        '''Get the 3D mid-plane density in cgs.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_midplane_turbpress_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_midplane_turbpress_Rphi (gas particles only).")

        dens_3D = self.get_density_Rphiz(PartType=PartType)
        if self.midplane_idcs is not None:
            return self._select_predef_midplane(dens_3D)
        else: # estimate the mid-plane by returning the maximum value along the z-axis
            return np.nanmax(dens_3D, axis=2)

    def get_gas_av_vel_xyz_Rphi(
        self,
        z_min: float=None,
        z_max: float=None,
        PartType: int=None
    ) -> Tuple[np.array, np.array, np.array]:
        '''Get the average 2D gas velocity components in the x, y, and z directions, in cm/s.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_av_vel_xyz_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_av_vel_xyz_Rphi (gas particles only).")

        if z_min==None:
            z_min = -self.total_height
        if z_max==None:
            z_max = self.total_height

        cnd = (self.data[PartType]["z_coords"] > z_min) & (self.data[PartType]["z_coords"] < z_max)
        if len(self.data[PartType]["R_coords"][cnd]) == 0:
            logger.critical("No gas cells found in the given z-range.")

        summass, R_edge, phi_edge, binnumbers = binned_statistic_2d(
            self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd],
            bins=(self.Rbin_edges, self.phibin_edges), expand_binnumbers=True,
            statistic='sum'
        )

        meanvels = []
        for velstring in ["velxs", "velys", "velzs"]:
            meanvel, R_edge, phi_edge, binnumbers = binned_statistic_2d(
                self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType][velstring][cnd],
                bins=(self.Rbin_edges, self.phibin_edges), expand_binnumbers=True,
                statistic='sum'
            )
            meanvel /= summass
            meanvels.append(meanvel)

        return tuple(meanvels)

    def get_gas_veldisps_xyz_Rphi(
        self,
        meanvels_xyz: Tuple=None,
        z_min: float=None,
        z_max: float=None,
        PartType: int=None,
    ) -> Tuple[np.array, np.array, np.array]:
        '''2D Gas velocity dispersion components in cgs. Distinct from the mid-plane turbulent
        velocity dispersion, this is the velocity dispersion along columns in z.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_veldisps_xyz_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_veldisps_xyz_Rphi (gas particles only).")

        if z_min==None:
            z_min = -self.total_height
        if z_max==None:
            z_max = self.total_height

        cnd = (self.data[PartType]["z_coords"] > z_min) & (self.data[PartType]["z_coords"] < z_max)
        if len(self.data[PartType]["R_coords"][cnd]) == 0:
            logger.critical("No gas cells found in the given z-range.")

        summass, R_edge, phi_edge, binnumbers = binned_statistic_2d(
            self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd],
            bins=(self.Rbin_edges, self.phibin_edges), expand_binnumbers=True,
            statistic='sum'
        )

        '''Subtract the mean velocity in each bin from the velocity components, then take the
        root-mean-square of the differences to get the velocity dispersions.'''
        if meanvels_xyz==None:
            meanvels = self.get_gas_av_vel_xyz_Rphi(z_min=z_min, z_max=z_max, PartType=PartType)
        else:
            meanvels = meanvels_xyz

        maxbinnumber_x = len(R_edge)-1
        maxbinnumber_y = len(phi_edge)-1
        veldisps_xyz = []
        for velstring, meanvel in zip(["velxs", "velys", "velzs"], meanvels):
            bn_x, bn_y = binnumbers.copy()
            bn_x[bn_x>maxbinnumber_x] = maxbinnumber_x
            bn_y[bn_y>maxbinnumber_y] = maxbinnumber_y
            bn_x[bn_x<1] = 1
            bn_y[bn_y<1] = 1
            bn_x -= 1
            bn_y -= 1
            vel_minus_mean = self.data[PartType][velstring][cnd]-meanvel[bn_x, bn_y]

            sumveldisp, R_edge, phi_edge, binnumbers_ = binned_statistic_2d(
                self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd]*vel_minus_mean**2,
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            veldisps_xyz.append(np.sqrt(sumveldisp/summass))

        return tuple(veldisps_xyz)

    def _get_gas_turbpress_Rphiz(self, PartType: int=None) -> np.array:
        '''3D Gas turbulent pressure in cgs units, divided by the Boltzmann constant.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_turbpress_Rphiz.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_turbpress_Rphiz (gas particles only).")

        density = self.get_density_Rphiz(PartType=PartType)

        veldisps_z = np.zeros((self.Rbinno, self.phibinno, self.zbinno)) * np.nan
        meanvels = self.get_gas_av_vel_xyz_Rphi(z_min=-self.total_height, z_max=self.total_height, PartType=PartType)
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                veldisps_z[:,:,k] = np.ones((self.Rbinno, self.phibinno)) * np.nan
                continue
            veldisps_x, veldisp_y, veldisp_z = self.get_gas_veldisps_xyz_Rphi(meanvels_xyz=meanvels, z_min=zbmin, z_max=zbmax, PartType=PartType)
            veldisps_z[:,:,k] = veldisp_z
        
        turbpress_3D = veldisps_z**2 * density / ah.kB_cgs

        return turbpress_3D

    def _select_predef_midplane(self, input_array: np.array) -> np.array:
        '''If the mid-plane of the galaxy is already known from a previous calculation (e.g. from the minimum
        of the gravitational potential), select the values at the mid-plane of the array.'''

        midplane_value = np.zeros_like(self.midplane_idcs) * np.nan
        for i in range(self.Rbinno):
            for j in range(self.phibinno):
                midplane_value[i,j] = input_array[i,j,self.midplane_idcs[i,j]]

        return midplane_value

    def get_gas_midplane_turbpress_Rphi(self, PartType: int=None) -> np.array:
        '''Mid-plane gas turbulent pressure (2D) in the vertical/plane-perpendicular direction, in cgs units,
        divided by the Boltzmann constant.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_midplane_turbpress_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_midplane_turbpress_Rphi (gas particles only).")

        turbpress_3D = self._get_gas_turbpress_Rphiz(PartType=PartType)
        if self.midplane_idcs is not None:
            return self._select_predef_midplane(turbpress_3D)
        else: # estimate the mid-plane by returning the maximum value along the z-axis
            return np.nanmax(turbpress_3D, axis=2)

    def get_gas_midplane_thermpress_Rphi(self, PartType: int=None) -> np.array:
        '''Gas midplane thermal pressure (2D) in cgs units.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_midplane_thermpress_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_midplane_thermpress_Rphi (gas particles only).")

        Pth = np.zeros((self.Rbinno, self.phibinno, self.zbinno)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                Pth[:,:,k] = np.ones((self.Rbinno, self.phibinno)) * np.nan
                continue
            Pth_bin, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType]["U"][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            Pth[:,:,k] = Pth_bin / (self.Rbin_width * self.zbin_sep * self.Rbin_centers_2d*self.phibin_sep) # mass * U --> dens * U

        if self.midplane_idcs is not None:
            return self._select_predef_midplane(Pth)
        else:
            return np.nanmax(Pth * (ah.gamma-1.) / ah.kB_cgs, axis=2) # dens * U --> Ptherm

    def get_count_Rphi(self, PartType: int=None) -> np.array:
        '''Get the number of particles being sampled per bin.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_count_Rphi.")

        count, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.data[PartType]["R_coords"], self.data[PartType]["phi_coords"], self.data[PartType]["R_coords"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='count'
        )

        return count

    def _get_count_Rphiz(self, PartType: int=None) -> np.array:
        '''Get the number of particles being sampled per 3D bin.'''

        if PartType==None:
            logger.critical("Please specify a particle type for _get_count_Rphiz.")

        count_3D = np.zeros((self.Rbinno, self.phibinno, self.zbinno))
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                continue
            count, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.data[PartType]["R_coords"][cnd], self.data[PartType]["phi_coords"][cnd], self.data[PartType]["R_coords"][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='count'
            )
            count_3D[:,:,k] = count
        
        return count_3D

    def get_count_midplane_Rphi(self, PartType: int=None) -> np.array:
        '''Get the number of particles being sampled per bin, in the mid-plane, using the pressure
        computation.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_count_midplane_Rphi.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_count_midplane_Rphi (gas particles only).")

        turbpress_3D = self._get_gas_turbpress_Rphiz(PartType=PartType)
        count_3D = self._get_count_Rphiz(PartType=PartType)
        count = np.zeros((self.Rbinno, self.phibinno))
        for i in range(self.Rbinno):
            for j in range(self.phibinno):
                try:
                    midplane_idx = np.nanargmax(turbpress_3D[i,j,:])
                    count[i,j] = count_3D[i,j,midplane_idx]
                except ValueError as e:
                    if str(e) == "All-NaN slice encountered":
                        count[i,j] = 0

        if self.midplane_idcs is not None:
            return self._select_predef_midplane(count_3D)
        else:
            return count

    def get_surfdens_Rphi(self, PartType: int=None) -> np.array:
        '''Surface density in cgs units.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_surfdens_Rphi.")

        dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
            self.data[PartType]["R_coords"], self.data[PartType]["phi_coords"], self.data[PartType]["masses"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        return dens / (self.Rbin_width * self.Rbin_centers_2d*self.phibin_sep)

    def get_component_surfdens_Rphi(self, component: str=None) -> np.array:
        '''Surface density in cgs units, of a particular gas component'''

        if component==None:
            logger.critical("Please specify a gas component for get_component_surfdens_Rphi.")

        dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
            self.data[0]["R_coords"], self.data[0]["phi_coords"], self.data[0]["masses"] * self.data[0][component],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        return dens / (self.Rbin_width * self.Rbin_centers_2d*self.phibin_sep)

    def get_SFR_surfdens_Rphi(self) -> np.array:
        '''Gas star formation rate surface density in cgs units.'''

        Sfr, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.data[0]["R_coords"], self.data[0]["phi_coords"], self.data[0]["SFRs"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        return Sfr / (self.Rbin_width * self.Rbin_centers_2d*self.phibin_sep)

    def get_weight_integrand_Rphiz(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=0) -> np.array:
        '''Get the integrand for the weight function, in cgs units.'''

        rho_grid = self.get_density_Rphiz(zbinsep=self.zbin_sep_ptl, PartType=PartType)
        ptl_grid = self._get_potential_Rphiz(PartTypes=PartTypes)

        dz = np.gradient(self.zbin_centers_3d_ptl, axis=2)
        dPhi = np.gradient(ptl_grid, axis=2)
        dPhidz = dPhi/dz

        integrand = rho_grid * dPhidz * self.zbin_sep_ptl
        return integrand

    def get_weight_Rphi(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=0) -> np.array:
        '''Get the weights for the interstellar medium, based on the density and potential
        grids, assuming the potential is symmetrical about the mid-plane of the disk.'''

        integrand = self.get_weight_integrand_Rphiz(PartTypes=PartTypes, PartType=PartType)
        return np.nansum(np.fabs(integrand)/2., axis=2)

    def get_int_force_left_right_Rphi(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=0) -> Tuple[np.array, np.array, np.array]:
        '''Get integrated force per unit area, separated into its components above and below
        the mid-plane of the disk.'''

        integrand = self.get_weight_integrand_Rphiz(PartTypes=PartTypes, PartType=PartType)
        z_mp_idcs = np.nanargmin(np.nancumsum(integrand, axis=2), axis=2)

        integrand_left = np.zeros_like(integrand)
        integrand_right = np.zeros_like(integrand)
        for i in range(self.Rbinno):
            for j in range(self.phibinno):
                integrand_left[i,j,:z_mp_idcs[i,j]] = integrand[i,j,:z_mp_idcs[i,j]]
                integrand_right[i,j,z_mp_idcs[i,j]:] = integrand[i,j,z_mp_idcs[i,j]:]
                
        return np.nansum(integrand_left, axis=2), np.nansum(integrand_right, axis=2), z_mp_idcs
    
    def get_force_Rphi(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=0) -> np.array:
        '''Get the net force resulting from an asymmetric potential, in cgs units.'''

        integrand = self.get_weight_integrand_Rphiz(PartTypes=PartTypes, PartType=PartType)
        return np.nansum(integrand, axis=2)

    def _get_potential_Rphiz(
        self,
        PartTypes: List[int] = [0, 1, 2, 3, 4],
        sigma: float=0., # smoothing parameter, defaults to 0.
        eps: float=1., # shape parameter, defaults to 1.
        phi: str="phs3", # 3rd-order polyharmonic spline
        order: int=5, # at least q-1, where q is the order of the RBF of type phi
        neighbors: int=150, # interpolant at each eval point uses this many observations, defaults to all
    ):
        '''Get the potential array from the gas cells, in cgs units. This uses the RBF
        interpolation class at https://rbf.readthedocs.io/en/latest/.
        Details about the optimization parameters given as keyword arguments in this
        function can be found in these docs.'''

        try:
            x_all = np.concatenate([self.data[i]["x_coords"] for i in PartTypes if self.data[i] is not None])
            y_all = np.concatenate([self.data[i]["y_coords"] for i in PartTypes if self.data[i] is not None])
            z_all = np.concatenate([self.data[i]["z_coords"] for i in PartTypes if self.data[i] is not None])
            ptl_all = np.concatenate([self.data[i]["Potential"] for i in PartTypes if self.data[i] is not None])
            coords_all = np.array([x_all, y_all, z_all]).T
        except KeyError:
            logger.critical("Requested particle types not loaded: {:s}".format(str([i for i in PartTypes if i not in self.data])))
            sys.exit(1)

        interp = RBFInterpolant(
            coords_all,
            ptl_all,
            sigma=sigma,
            eps=eps,
            phi=phi,
            order=order,
            neighbors=neighbors
        )

        coords_interp = np.array([self.Rbin_centers_3d_ptl.flatten(), self.phibin_centers_3d_ptl.flatten(), self.zbin_centers_3d_ptl.flatten()]).T
        return interp(coords_interp).reshape(self.Rbinno, self.phibinno, self.zbinno_ptl)

    

    