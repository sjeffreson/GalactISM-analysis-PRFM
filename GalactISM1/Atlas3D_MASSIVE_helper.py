# Helper functions and unit conversions for MASSIVE and ATLAS-3D data
import numpy as np

# get the expected weights for the ETGs in the ATLAS-3D/MASSIVE samples
# (and compare to their stellar surface densities)
def calc_ang_size(extent, distance, unit): #make sure that your distances are both in the same unit! 
    size = extent/distance
    
    if "rad" in unit:
        return size
    elif "deg" in unit:
        return size*(180/np.pi)
    elif "arcsec" in unit:
        return size*(180/np.pi)*3600

def calc_physical_size(angular_size, distance, unit): #will return result in the same unit as distance
    physical_size = angular_size*distance
    
    if "rad" in unit:
        return physical_size
    elif "deg" in unit:
        return physical_size*(np.pi/180.)
    elif "arc" in unit:
        return physical_size*(np.pi/180.)/3600.
    
def calc_c(mhalo_msun, h):
    return 10**(0.905-0.101*np.log10(mhalo_msun/(1e12*(1/h))))

def multf(conc):
    num = np.sqrt(2*(np.log(1+conc)-(conc/(1+conc))))
    fac = num/conc
    pf = 1+ fac
    return pf*pf

def m200(mhalo, conc):
    return mhalo/multf(conc)

def fc(c):
    return c * (0.5 - 0.5 / pow(1 + c, 2) - np.log(1 + c) / (1 + c)) / pow(np.log(1 + c) - c / (1 + c), 2)

def r200(vc,h):
    return vc/(10*h)

def scale_length(c, LAMBDA, vc, h):
    return sqrt(2.0) / 2.0 * LAMBDA / fc(c) * r200(vc, h)

def vc(mhalo, c, h):
    m2 = m200(mhalo, c)
    return pow(10*m2*4300*h, 1./3.)

def calc_M200(v200, G, h):
    return 1e9 * pow(v200, 3) / (G * h)

def calc_logMstar(MKS):
    return 10.58 - 0.44 * (MKS+23)