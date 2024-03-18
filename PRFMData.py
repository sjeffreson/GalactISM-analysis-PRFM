from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from scipy.signal import savgol_filter as sg
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d

import h5py
import astro_helper as ah
from invdisttree import Invdisttree

DEFAULT_ROOT_DIR = Path("/n/holystore01/LABS/itc_lab/Users/sjeffreson/")
class PRFMDataset:
    '''Calculate physical properties for validation of pressure-regulated star formation,
    using Arepo simulation data, in bins of R and phi (cylindrical coordinates).'''

    def __init__(
        self,
        galaxy_type: str,
        total_height: float = 1.5, # kpc
        Rmax: float = 15., # kpc
        Rmin: float = 0.3, # kpc
        Rbin_width: float = 0.5, # kpc
        Rbin_sep: float = 0.2, # kpc
        phibin_sep: float = np.pi/4., # rad
        zbin_sep: float = 10., # pc
        zbin_sep_ptl: float = 10., # pc, for computation of the potential
        root_dir: Path = DEFAULT_ROOT_DIR,
        snapname: str = "snap-DESPOTIC_300.hdf5",
        realign_galaxy: bool=True, # according to angular momentum vector of gas
        required_particle_types: List[int] = [0], # just gas by default
    ):
        
        self.galaxy_type = galaxy_type
        self.total_height = total_height * ah.kpc_to_cm
        self.Rmax = Rmax * ah.kpc_to_cm
        self.Rmin = Rmin * ah.kpc_to_cm
        self.Rbin_width = Rbin_width * ah.kpc_to_cm
        self.Rbin_sep = Rbin_sep * ah.kpc_to_cm
        self.phibin_sep = phibin_sep
        self.zbin_sep = zbin_sep * ah.pc_to_cm
        self.zbin_sep_ptl = zbin_sep_ptl * ah.pc_to_cm
        self.root_dir = root_dir
        self.snapname = snapname
        self.realigned = realign_galaxy

        '''Load all required data'''
        self.data = {}
        for part_type in required_particle_types:
            self.data[part_type] = self.read_snap_data(part_type)

        '''PartType 6 for all stars'''
        present_stellar_types = [i for i in range(2, 5) if i in self.data]
        if len(present_stellar_types) > 0:
            self.data[6] = {key: np.concatenate([self.data[i][key] for i in present_stellar_types]) for key in self.data[present_stellar_types[0]]}

        if realign_galaxy:
            if 0 not in self.data:
                raise ValueError("Gas data must be loaded to realign the galaxy.")
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self.get_gas_disk_COM()
            self.Lx, self.Ly, self.Lz = self.get_gas_disk_angmom()
            self.data = {key: self.set_realign_galaxy(value) for key, value in self.data.items()}

        '''Grid on which to compute arrays'''
        self.Rbinno = int(np.rint((self.Rmax-self.Rmin)/(self.Rbin_sep)))
        self.Rbin_edges = np.linspace(self.Rmin, self.Rmax, Rbinno+1)
        self.Rbin_centers = (self.Rbin_edges[1:]+self.Rbin_edges[:-1])/2.
        self.Rbin_mins = self.Rbin_centers - self.Rbin_width/2.
        self.Rbin_maxs = self.Rbin_centers + self.Rbin_width/2.

        self.phibinno = int(np.rint(2.*np.pi/self.phibin_sep))
        self.phibin_edges = np.linspace(-np.pi, np.pi, phibinno+1)

        self.zbinno = int(np.rint(2.*self.total_height/self.zbin_sep))
        self.zbin_edges = np.linspace(-self.total_height, self.total_height, zbinno+1)
    
    def get_grid(self) -> Tuple[np.array, np.array, np.array]:
        return self.Rbin_centers, self.phibin_edges, self.zbin_edges

    def read_snap_data(
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
        snapshot = h5py.File(DEFAULT_ROOT_DIR / self.galaxy_type / self.snapname, "r")
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
            return snap_data
        else:
            snap_data["Density"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["U"] = PartType_data['InternalEnergy'][:] * PartType_data['InternalEnergy'].attrs['to_cgs']
            snap_data["voldenses"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["temps"] = (ah.gamma - 1.) * snap_data["U"] / ah.kB_cgs * ah.mu * ah.mp_cgs
            snap_data["SFRs"] = PartType_data['StarFormationRate'][:] * PartType_data['StarFormationRate'].attrs['to_cgs']
            snap_data["AlphaVir"] = PartType_data['AlphaVir'][:]
            return snap_data
    
    def cut_out_gas_disk(self) -> Dict[str, np.array]:
        '''Cut out most of the gas cells that are in the background grid, not the disk'''

        cnd = (self.gas_data["R_coords"] < self.total_width/2.) & (np.fabs(self.gas_data["z_coords"]) < self.total_width/2.)
        return {key: value[cnd] for key, value in self.gas_data.items()}

    def get_gas_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas disk'''

        gas_data_cut = self.cut_out_gas_disk()

        x_CM = np.average(gas_data_cut["x_coords"], weights=gas_data_cut["masses"])
        y_CM = np.average(gas_data_cut["y_coords"], weights=gas_data_cut["masses"])
        z_CM = np.average(gas_data_cut["z_coords"], weights=gas_data_cut["masses"])
        vx_CM = np.average(gas_data_cut["velxs"], weights=gas_data_cut["masses"])
        vy_CM = np.average(gas_data_cut["velys"], weights=gas_data_cut["masses"])
        vz_CM = np.average(gas_data_cut["velzs"], weights=gas_data_cut["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def get_gas_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the gas disk'''

        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self.get_gas_disk_COM()
        gas_data_cut = self.cut_out_gas_disk()

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

    def set_realign_galaxy(self, snap_data: Dict[str, np.array]) -> Dict[str, np.array]:
        '''Realign the galaxy according to the center of mass and the angular momentum
        vector of the gas disk'''

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

    def get_rotation_curve_R(self) -> Tuple[np.array, np.array]:
        '''Get the 1D rotation curve of the galaxy, in cm/s. DeltaR is the size of the radial bins
        (pc) used to compute the rotation curve at all azimuthal angles, which is then interpolated
        to obtain the grid.'''

        vcs = []
        for Rbmin, Rbmax in zip(self.Rbin_edges[:-1], self.Rbin_edges[1:]):
            cnd = (self.gas_data['R_coords'] > Rbmin) & (self.gas_data['R_coords'] < Rbmax)
            if(len(self.gas_data['R_coords'][cnd])>0):
                vcs.append(np.average(
                    -self.gas_data['y_coords'][cnd]/self.gas_data['R_coords'][cnd] * self.gas_data['velxs'][cnd] +
                    self.gas_data['x_coords'][cnd]/self.gas_data['R_coords'][cnd] * self.gas_data['velys'][cnd],
                    weights=self.gas_data['masses'][cnd]))
            else:
                vcs.append(np.nan)

        return vcs

    def get_Omegaz_R(self) -> np.array:
        '''Get the 1D galactic angular velocity in the z-direction, in /s.'''

        vcs = self.get_rotation_curve()
        Omegazs = vcs / self.Rbin_centers

        return Omegazs
    
    def get_kappa_R(self, polyno: int=2, wndwlen: int=9) -> np.array:
        '''Get the 1D epicyclic frequency in the z-direction, in /s.'''

        vcs = self.get_rotation_curve()
        Omegazs = vcs / self.Rbin_centers

        dR = sg(self.Rbin_centers, wndwlen, polyno, deriv=1)
        dvc = sg(vcs, wndwlen, polyno, deriv=1)
        betas = dvc/dR * self.Rbin_centers/vcs
        kappas = Omegazs * np.sqrt(2.*(1.+betas))

        return kappas

    def get_gas_density_Rphiz(self) -> np.array:
        '''Get the 3D gas mid-plane density in cgs'''

        for zbmin, zbmax in zip(self.zbin_edges[:-1], self.zbin_edges[1:]):
            cnd = (self.gas_data["z_coords"] > zbmin) & (self.gas_data["z_coords"] < zbmax)
            dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )

        R_edges_2d, phi_edges_2d = np.meshgrid(R_edge, phi_edge)
        R_centers_2d = (R_edges_2d[:-1]+R_edges_2d[1:])/2.
        return dens / (self.Rbin_width*ah.kpc_to_cm * self.zbin_sep*ah.pc_to_cm * R_centers_2d*self.phibin_sep*ah.pc_to_cm)

    def get_gas_av_vel_xyz_Rphi(
        self,
        z_min: float=-self.total_height,
        z_max: float=self.total_height
    ) -> Tuple[np.array, np.array, np.array]:
        '''Get the average 2D gas velocity components in the x, y, and z directions, in cm/s.'''

        cnd = (self.gas_data["z_coords"] > z_min) & (self.gas_data["z_coords"] < z_max)

        summass, R_edge, phi_edge, binnumber = binned_statistic_2d(
            self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        meanvels = []
        for velstring in ["velxs", "velys", "velzs"]:
            meanvel, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd]*self.gas_data[velstring][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            meanvel /= summasses
            meanvels.append(meanvel)
            binnumbers.append(binnumber)

        return tuple(meanvels)

    def get_gas_veldisps_xyz_Rphi(
        self,
        Tuple: meanvels_xyz=None,
        z_min: float=-self.total_height,
        z_max: float=self.total_height
    ) -> Tuple[np.array, np.array, np.array]:
        '''2D Gas velocity dispersion components in cgs. Distinct from the mid-plane turbulent
        velocity dispersion, this is the velocity dispersion along columns in z.'''

        cnd = (self.gas_data["z_coords"] > z_min) & (self.gas_data["z_coords"] < z_max)

        summass, x_edge, y_edge, binnumbers = binned_statistic_2d(
            self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd],
            bins=(self.Rbin_edges, self.phibin_edges), expand_binnumbers=True,
            statistic='sum'
        )

        '''Subtract the mean velocity in each bin from the velocity components, then take the
        root-mean-square of the differences to get the velocity dispersions.'''
        if meanvels_xyz==None:
            meanvels = self.get_gas_av_vel_xyz(z_min=z_min, z_max=z_max)
        else:
            meanvels = meanvels_xyz

        maxbinnumber = (len(self.Rbin_edges)-1)*(len(self.phibin_edges)-1)
        veldisps_xyz = []
        for velstring, meanvel, binnumber in zip(["velxs", "velys", "velzs"], meanvels, binnumbers):
            bn_x, bn_y = binnumber
            bn_x[bn_x>maxbinnumber] = maxbinnumber
            bn_y[bn_y>maxbinnumber] = maxbinnumber
            bn_x[bn_x<1] = 1
            bn_y[bn_y<1] = 1
            bn_x -= 1
            bn_y -= 1
            vel_minus_mean = self.gas_data[velstring][cnd]-meanvel[bn_x, bn_y]

            sumveldisp, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd]*vel_minus_mean**2,
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            veldisps_xyz.append(np.sqrt(sumveldisp/summasses))

        return tuple(veldisps_xyz)

    def get_gas_midplane_turbpress_Rphi(self) -> Tuple[np.array, np.array, np.array]:
        '''Mid-plane gas turbulent pressure (2D) in the vertical/plane-perpendicular direction, in cgs units,
        divided by the Boltzmann constant.'''

        density = self.get_gas_density_Rphiz()

        veldisps_z = np.zeros((self.Rbinno, self.phibinno, self.zbinno)) * np.nan
        meanvels, binnumbers = self.get_gas_av_vel_xyz(z_min=-self.total_height, z_max=self.total_height)
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.gas_data["z_coords"] > zbmin) & (self.gas_data["z_coords"] < zbmax)
            veldisps_x, veldisp_y, veldisp_z = self.get_gas_veldisps_xyz(meanvels_xyz=meanvels, z_min=zbmin, z_max=zbmax)
            veldisps_z[:,:,k] = veldisps_z
        
        turbpress_3D = veldisps_z**2 * density / ah.kB_cgs

        return np.nanmax(turbpress_3D, axis=2)

    def get_gas_midplane_thermpress_Rphi(self) -> np.array:
        '''Gas midplane thermal pressure (2D) in cgs units.'''

        Pth = np.zeros((self.Rbinno, self.phibinno, self.zbinno)) * np.nan
        for zbmin, zbmax in zip(self.zbin_edges[:-1], self.zbin_edges[1:]):
            cnd = (self.gas_data["z_coords"] > zbmin) & (self.gas_data["z_coords"] < zbmax)
            Pth_bin, R_edge, phi_edge, binnumber = binned_statistic_2d(
                self.gas_data["R_coords"][cnd], self.gas_data["phi_coords"][cnd], self.gas_data["masses"][cnd]*self.gas_data["U"][cnd],
                bins=(self.Rbin_edges, self.phibin_edges),
                statistic='sum'
            )
            Pth[:,:,k] = Pth_bin

        R_edges_2d, phi_edges_2d = np.meshgrid(R_edge, phi_edge)
        R_centers_2d = (R_edges_2d[:-1]+R_edges_2d[1:])/2.
        Pth = Pth / (self.Rbin_width*ah.kpc_to_cm * self.zbin_sep*ah.pc_to_cm * R_centers_2d*self.phibin_sep*ah.pc_to_cm) # mass * U --> dens * U

        return np.nanmax(Pth * (ah.gamma-1.) / ah.kB_cgs * ah.mu * ah.mp_cgs, axis=2) # dens * U --> Ptherm

    def get_SFR_surfdens_array_Rphi(self) -> np.array:
        '''Gas star formation rate surface density in cgs units.'''

        Sfr, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["R_coords"], self.gas_data["phi_coords"], self.gas_data["SFRs"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        R_edges_2d, phi_edges_2d = np.meshgrid(R_edge, phi_edge)
        R_centers_2d = (R_edges_2d[:-1]+R_edges_2d[1:])/2.
        return Sfr / (self.Rbin_width*ah.kpc_to_cm * R_centers_2d*self.phibin_sep*ah.pc_to_cm)

    def get_gas_surfdens_Rphi(self) -> np.array:
        '''Gas surface density in cgs units.'''

        dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
            self.gas_data["R_coords"], self.gas_data["phi_coords"], self.gas_data["masses"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        R_edges_2d, phi_edges_2d = np.meshgrid(R_edge, phi_edge)
        R_centers_2d = (R_edges_2d[:-1]+R_edges_2d[1:])/2.
        return dens / (self.Rbin_width*ah.kpc_to_cm * R_centers_2d*self.phibin_sep*ah.pc_to_cm)

    def get_stellar_surfdens_Rphi(self) -> np.array:
        '''Stellar surface density in cgs units.'''

        if 6 not in self.data:
            raise ValueError("Stellar data must be loaded to use this function.")

        dens, R_edge, phi_edge, binnumber = binned_statistic_2d(
            self.data[6]["R_coords"], self.data[6]["phi_coords"], self.data[6]["masses"],
            bins=(self.Rbin_edges, self.phibin_edges),
            statistic='sum'
        )

        R_edges_2d, phi_edges_2d = np.meshgrid(R_edge, phi_edge)
        R_centers_2d = (R_edges_2d[:-1]+R_edges_2d[1:])/2.
        return dens / (self.Rbin_width*ah.kpc_to_cm * R_centers_2d*self.phibin_sep*ah.pc_to_cm)

### TO DO: Change this to the RBF interpolation method
# def get_potential_z_grid(self) -> Tuple[int, np.array]:
#     ptl_z_binnum = int(np.ceil(2.*self.total_height/self.ptl_resolution))
#     ptl_z_bin_edges = np.linspace(-self.total_height, self.total_height, ptl_z_binnum+1)
#     ptl_z_bin_centers = (ptl_z_bin_edges[1:]+ptl_z_bin_edges[:-1])*0.5
#     return ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers

# def get_potential_array(self, leafsize: int=10, eps: float=6., p: int=1) -> np.array:
#     '''Get the potential array from the gas cells, in cgs units. Leafsize, eps and p
#     are performance parameters for the inverse distance interpolation. See description
#     in invdisttree.py'''

#     if self.ptl_resolution == None:
#         raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

#     '''the z-grid for calculation of the potential needs to be sufficiently-fine to capture
#     density fluctations near the simulation resolution'''
#     ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()

#     snapdata_all = [self.read_snap_data(i) for i in range(5) if self.read_snap_data(i) != None]
#     snapdata_all = [self.set_realign_galaxy(snapdata) for snapdata in snapdata_all]
#     x_all = np.concatenate([snapdata["x_coords"] for snapdata in snapdata_all])
#     y_all = np.concatenate([snapdata["y_coords"] for snapdata in snapdata_all])
#     z_all = np.concatenate([snapdata["z_coords"] for snapdata in snapdata_all])
#     ptl_all = np.concatenate([snapdata["Potential"] for snapdata in snapdata_all])

#     x_grid, y_grid, z_grid = np.meshgrid(self.xbin_centers, self.ybin_centers, ptl_z_bin_centers)

#     invdisttree = Invdisttree(np.array([x_all, y_all, z_all]).T, ptl_all, leafsize=leafsize, stat=1)
#     ptl_grid = invdisttree(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T, nnear=8, eps=eps, p=p)
#     ptl_grid = np.reshape(ptl_grid, np.shape(x_grid))

#     return ptl_grid

# def get_rho_potential_array(self) -> np.array:
#     '''Get the density array corresponding to the potential grid, for computation of
#     interstellar medium weights, in cgs units.'''

#     if self.ptl_resolution == None:
#         raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

#     ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()

#     cnd = (self.gas_data["AlphaVir"] > 2.) | (self.gas_data["AlphaVir"] <= 0.) # cut out gas that's in gravitationally-bound clouds
#     x = self.gas_data["x_coords"][cnd]
#     y = self.gas_data["y_coords"][cnd]
#     z = self.gas_data["z_coords"][cnd]
#     mass = self.gas_data["masses"][cnd]

#     rho3Ds = np.zeros((self.xy_binnum, self.xy_binnum, ptl_z_binnum))
#     for zbmin, zbmax, k in zip(ptl_z_bin_edges[:-1], ptl_z_bin_edges[1:], range(ptl_z_binnum)):
#         cnd = (z > zbmin) & (z < zbmax)
#         mass_bin = mass[cnd]
#         x_bin = x[cnd]
#         y_bin = y[cnd]

#         masssum, xedges, yedges, binnumber = binned_statistic_2d(
#             x_bin, y_bin,
#             mass_bin,
#             bins=[self.xbin_edges, self.ybin_edges],
#             statistic='sum')
#         rho3Ds[:,:,k] = masssum/self.resolution/self.resolution/self.ptl_resolution
    
#     return rho3Ds

# def get_weight_array(self, polyno: int=2, wndwlen: int=9) -> np.array:
#     '''Get the weights for the interstellar medium, based on the density and potential
#     grids. Polyno and wndlen are the parameters for the Savitzky-Golay filter, which
#     is used to take the z-derivative of the potential grid. Output in cgs units.'''

#     if self.ptl_resolution == None:
#         raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

#     ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()
#     deltaz = ptl_z_bin_centers[1]-ptl_z_bin_centers[0]
#     x_grid, y_grid, z_grid = np.meshgrid(self.xbin_centers, self.ybin_centers, ptl_z_bin_centers)

#     rho_grid = self.get_rho_potential_array()
#     ptl_grid = self.get_potential_array()
    
#     dz = sg(z_grid, wndwlen, polyno, deriv=1, axis=2)
#     dPhi = sg(ptl_grid, wndwlen, polyno, deriv=1, axis=2)
#     dPhidz = dPhi/dz

#     return np.sum(np.fabs(rho_grid*dPhidz*deltaz)/2., axis=2)