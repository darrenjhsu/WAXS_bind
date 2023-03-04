

import sys
import importlib
import numpy as np
from saxstats import saxstats
from Ligand import Ligand
#from WAXS import *
#from PDB import PDB
from Geometry import *
import rdkit
from rdkit import Chem
import time
#from xs_helper import xray_scatter 
from array import array
from scipy.optimize import minimize
from spyrmsd import rmsd as srmsd


lig_gt = Ligand('1FIN_ligand.sdf')
lig_gt.set_coordinates(lig_gt.get_coordinates(0) - lig_gt.get_coordinates(0).mean(0)) # center
lig_coord_gt = lig_gt.get_coordinates()

lig = Ligand('1FIN_ligand.sdf')
lig.set_coordinates(lig_gt.get_coordinates(0)) # center
sp_np = {'t': np.random.random(3) * 0,
         'r': np.random.random(3) * 360 - 180,
         'a': np.random.random(lig_gt.num_angle) * 10 - 5,
         'd': np.random.random(lig_gt.num_torsion) * 360 - 180}
lig.set_coordinates(lig.transform(0, sp_np))


def x_to_dict(x, lig):
    return {'t': x[:3], 'r': x[3:6], 'a': x[6:6+lig.num_angle], 'd': x[6+lig.num_angle:]}
def myminfunc(x, lig, lig_coord_gt, isomorphism=None):
    return symmRMSD(lig.transform(0, x_to_dict(x, lig)), lig_coord_gt, lig.A, lig.G, isomorphism=isomorphism)[0]
def myminfunc_symmRMSD(x, lig, lig_coord_gt, isomorphism=None, return_isomorphism=False):
    x_dict = x_to_dict(x, lig)
    if return_isomorphism:
        return symmRMSD(lig.transform(0, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)
    else:
        return symmRMSD(lig.transform(0, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)[0]

rmsd, iso = myminfunc_symmRMSD(np.zeros(6+lig.num_angle+lig.num_torsion),
                          lig, lig_coord_gt, isomorphism=None, return_isomorphism=True)

res = minimize(myminfunc, np.zeros(6+lig.num_torsion+lig.num_angle), args=(lig, lig_coord_gt, iso), method='L-BFGS-B', options={'iprint':50, 'maxfun': 100000})
               #bounds=[(None, None), (None, None), (None, None), (0, 180), (0, 360), 
                       #*([(None, None)] * lig.num_torsion)], options={'iprint': 10})
print('sp_np')
print(sp_np)
print('result')
print(x_to_dict(res.x, lig))

print("Ground truth")
print(lig_coord_gt)

print("Minimized result")
print(lig.transform(0, res.x))

print("Initial Kabsch RMSD")
print(Kabsch_RMSD(lig_coord_gt, lig.get_coordinates())[0])

print("Final RMSD")
print(pureRMSD(lig_coord_gt, lig.transform(0, res.x)))

print("Final Kabsch RMSD")
print(Kabsch_RMSD(lig_coord_gt, lig.transform(0, res.x))[0])
