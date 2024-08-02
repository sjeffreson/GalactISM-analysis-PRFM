import h5py
import numpy as np
from scipy.stats import binned_statistic_2d
import sys
sys.path.append('../')
import astro_helper as ah

def collect_tau_deps(files):
    width_kpc = 30.
    tau_deps, SFRs = [], []
    for file in files:
        f = h5py.File(file, "r")
        SFR = f['PartType0']['StarFormationRate'][:] * f['PartType0']['StarFormationRate'].attrs['to_cgs']
        mass = f['PartType0']['Masses'][:] * f['PartType0']['Masses'].attrs['to_cgs']
        x = (f['PartType0']['Coordinates'][:,0] - f['Header'].attrs['BoxSize'] / 2.) * f['PartType0']['Coordinates'].attrs['to_cgs']
        y = (f['PartType0']['Coordinates'][:,1] - f['Header'].attrs['BoxSize'] / 2.) * f['PartType0']['Coordinates'].attrs['to_cgs']
        R = np.sqrt(x**2 + y**2)

        cnd = (SFR > 0) & (R/ah.kpc_to_cm < width_kpc/2.)
        mass = mass[cnd]
        SFR = SFR[cnd]

        tau_dep = mass / SFR
        SFRs.extend(list(SFR))
        tau_deps.extend(list(tau_dep))
    SFRs = np.array(SFRs)
    tau_deps = np.array(tau_deps)
    return tau_deps, SFRs

def collect_tau_dyns_prfm(files):
    width_kpc = 30.
    tau_dyns = []
    for file in files:
        f = h5py.File(file, "r")
        SFR = f['PartType0']['StarFormationRate'][:] * f['PartType0']['StarFormationRate'].attrs['to_cgs']
        x = ((f['PartType0']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])
        y = ((f['PartType0']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])
        R = np.sqrt(x**2 + y**2)
        PDE = f['PartType0/PDE'][:]
        SigmaGas = f['PartType0/SigmaGas'][:]
        sigmaEff = f['PartType0/Veldisp'][:]
        
        cnd = (SFR > 0) & (R/ah.kpc_to_cm < width_kpc/2.)
        PDE = PDE[cnd]
        SigmaGas = SigmaGas[cnd]
        sigmaEff = sigmaEff[cnd]

        tau_dyn = SigmaGas*sigmaEff/PDE * f['Header'].attrs['UnitLength_in_cm'] / f['Header'].attrs['UnitVelocity_in_cm_per_s']
        tau_dyns.extend(list(tau_dyn))
    tau_dyns = np.array(tau_dyns)
    return tau_dyns

def collect_tau_dyns(files, mode='SH03'):
    width_kpc = 30.
    x_bins = np.linspace(-width_kpc/2., width_kpc/2., int(np.ceil(width_kpc/0.5)))
    y_bins = np.linspace(-width_kpc/2., width_kpc/2., int(np.ceil(width_kpc/0.5)))

    tau_dyns = []
    for file in files:
        f = h5py.File(file, "r")
        SFR = f['PartType0']['StarFormationRate'][:] * f['PartType0']['StarFormationRate'].attrs['to_cgs']
        x = ((f['PartType0']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])
        y = ((f['PartType0']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])
        R = np.sqrt(x**2 + y**2)
        x_s = np.concatenate((
            f['PartType2']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize'],
            f['PartType3']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize']
        )) * f['PartType2']['Coordinates'].attrs['to_cgs']
        y_s = np.concatenate((
            f['PartType2']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize'],
            f['PartType3']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize']
        )) * f['PartType2']['Coordinates'].attrs['to_cgs']
        R_s = np.sqrt(x_s**2 + y_s**2)
        vx = f['PartType0']['Velocities'][:,0] * f['PartType0']['Velocities'].attrs['to_cgs']
        vy = f['PartType0']['Velocities'][:,1] * f['PartType0']['Velocities'].attrs['to_cgs']
        vtheta = -y*vx + x*vy
        Omega = vtheta / (x**2 + y**2)
        mass = f['PartType0']['Masses'][:] * f['PartType0']['Masses'].attrs['to_cgs']
        try:
            mass_s = np.concatenate((
                f['PartType2']['Masses'][:] * f['PartType2']['Masses'].attrs['to_cgs'],
                f['PartType3']['Masses'][:] * f['PartType3']['Masses'].attrs['to_cgs']
            ))
        except KeyError:
            mass_s = np.concatenate((
                np.ones_like(f['PartType2']['Coordinates'][:,0]) * f['Header'].attrs['MassTable'][2],
                np.ones_like(f['PartType3']['Coordinates'][:,0]) * f['Header'].attrs['MassTable'][3]
            ))

        U = (f['PartType0/InternalEnergy'][:] * f['PartType0/InternalEnergy'].attrs['to_cgs'])
        sigmaEff = (ah.gamma - 1.) * U

        cnd = (SFR > 0) & (R/ah.kpc_to_cm < width_kpc/2.)
        x = x[cnd]
        y = y[cnd]
        Omega = Omega[cnd]
        mass = mass[cnd]
        sigmaEff = sigmaEff[cnd]
        SFR = SFR[cnd]

        cnd_s = (R_s/ah.kpc_to_cm < width_kpc/2.)
        x_s = x_s[cnd_s]
        y_s = y_s[cnd_s]
        mass_s = mass_s[cnd_s]

        # calculate surface densities
        mass_2D, x_edges, y_edges = np.histogram2d(
            x/ah.kpc_to_cm, y/ah.kpc_to_cm,
            bins=(x_bins, y_bins),
            weights=mass/ah.Msol_to_g
        )
        area = np.outer(np.diff(x_edges), np.diff(y_edges)) * ah.kpc_to_cm**2
        mass_2D /= area

        mass_s_2D = np.histogram2d(
            x_s/ah.kpc_to_cm, y_s/ah.kpc_to_cm,
            bins=(x_bins, y_bins),
            weights=mass_s/ah.Msol_to_g
        )[0]
        mass_s_2D /= area

        # calculate the bin of each gas cell using binned_statistic_2d while returning binnumber
        bin_numbers = binned_statistic_2d(
            x/ah.kpc_to_cm, y/ah.kpc_to_cm,
            values=x,
            statistic='count',
            bins=(x_bins, y_bins),
            expand_binnumbers=True
        )[3]
        surfdens_g = mass_2D[bin_numbers[0]-1, bin_numbers[1]-1]
        surfdens_s = mass_s_2D[bin_numbers[0]-1, bin_numbers[1]-1]

        tau_dyn = 2.*sigmaEff / (
            np.pi * ah.G_cgs * (surfdens_g + surfdens_s) + ((np.pi * ah.G_cgs)**2 * (surfdens_g + surfdens_s)**2 + 8./3. * Omega**2 * sigmaEff**2)**0.5
        )
        tau_dyns.extend(list(tau_dyn))
    tau_dyns = np.array(tau_dyns)
    return tau_dyns

def collect_tau_dyns_resample(files, width_kpc=30., scale_kpc=0.5):
    '''For the high-res galaxies'''
    x_bins = np.linspace(-width_kpc/2., width_kpc/2., int(np.ceil(width_kpc/scale_kpc)))
    y_bins = np.linspace(-width_kpc/2., width_kpc/2., int(np.ceil(width_kpc/scale_kpc)))

    tau_dyns = []
    for file in files:
        f = h5py.File(file, "r")
        SFR = f['PartType0']['StarFormationRate'][:] * f['PartType0']['StarFormationRate'].attrs['to_cgs']
        x = ((f['PartType0']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])[SFR>0]
        y = ((f['PartType0']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize']) * f['PartType0']['Coordinates'].attrs['to_cgs'])[SFR>0]
        x_s = np.concatenate((
            f['PartType2']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize'],
            f['PartType3']['Coordinates'][:,0] - 0.5 * f['Header'].attrs['BoxSize']
        )) * f['PartType2']['Coordinates'].attrs['to_cgs']
        y_s = np.concatenate((
            f['PartType2']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize'],
            f['PartType3']['Coordinates'][:,1] - 0.5 * f['Header'].attrs['BoxSize']
        )) * f['PartType2']['Coordinates'].attrs['to_cgs']
        vx = f['PartType0']['Velocities'][:,0][SFR>0] * f['PartType0']['Velocities'].attrs['to_cgs']
        vy = f['PartType0']['Velocities'][:,1][SFR>0] * f['PartType0']['Velocities'].attrs['to_cgs']
        vtheta = -y*vx + x*vy
        Omega = vtheta / (x**2 + y**2)
        mass = f['PartType0']['Masses'][:][SFR>0] * f['PartType0']['Masses'].attrs['to_cgs']
        mass_s = np.concatenate((
            f['PartType2']['Masses'][:] * f['PartType2']['Masses'].attrs['to_cgs'],
            f['PartType3']['Masses'][:] * f['PartType3']['Masses'].attrs['to_cgs']
        ))

        sigmaEff = f['PartType0']['VelDisp'][:][SFR>0]

        # Using Equation (22) from old version of Sultan's paper, i.e. same equilibrium assumptions
        # as PRFM to compute the dynamical time
        sigmaEff_2D, _, _, _ = binned_statistic_2d(
            x/ah.kpc_to_cm, y/ah.kpc_to_cm,
            sigmaEff,
            bins=(x_bins, y_bins), statistic='mean'
        )

        Omega_2D, _, _, _ = binned_statistic_2d(
            x/ah.kpc_to_cm, y/ah.kpc_to_cm,
            Omega,
            bins=(x_bins, y_bins), statistic='mean'
        )

        mass_2D, x_edges, y_edges = np.histogram2d(
            x/ah.kpc_to_cm, y/ah.kpc_to_cm,
            bins=(x_bins, y_bins),
            weights=mass/ah.Msol_to_g
        )
        area = np.outer(np.diff(x_edges), np.diff(y_edges)) * ah.kpc_to_cm**2
        mass_2D /= area

        mass_s_2D = np.histogram2d(
            x_s/ah.kpc_to_cm, y_s/ah.kpc_to_cm,
            bins=(x_bins, y_bins),
            weights=mass_s/ah.Msol_to_g
        )[0]
        mass_s_2D /= area

        tau_dyn = 2.*sigmaEff_2D / (
            np.pi * ah.G_cgs * (mass_2D + mass_s_2D) + ((np.pi * ah.G_cgs)**2 * (mass_2D + mass_s_2D)**2 + 8./3. * Omega_2D**2 * sigmaEff_2D**2)**0.5
        )

        tau_dyn = np.ravel(tau_dyn)
        tau_dyn = tau_dyn[np.isfinite(tau_dyn)]
        tau_dyns.extend(list(tau_dyn))
    tau_dyns = np.array(tau_dyns)
    return tau_dyns