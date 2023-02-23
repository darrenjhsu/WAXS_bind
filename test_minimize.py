

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


lig_gt = Ligand('1FIN_ligand.sdf')
sp = np.zeros(6+lig_gt.num_torsion)
sp = np.random.random(6 + lig_gt.num_torsion) * 180
#lig_gt.set_coordinates(lig_gt.transform(0, sp))
lig_coord_gt = lig_gt.get_coordinates()

sp = np.zeros(6+lig_gt.num_torsion)
lig = Ligand('1FIN_ligand.sdf')
lig.generate_conformers(1)
sp = np.random.random(6 + lig.num_torsion) * 180
lig.set_coordinates(lig.transform(0, sp))

def myminfunc(x, lig, lig_coord_gt):
    return pureRMSD(lig.transform(0, x), lig_coord_gt)

res = minimize(myminfunc, np.zeros(6+lig.num_torsion), args=(lig, lig_coord_gt), method='L-BFGS-B', options={'iprint':10, 'maxfun': 100000})
               #bounds=[(None, None), (None, None), (None, None), (0, 180), (0, 360), 
                       #*([(None, None)] * lig.num_torsion)], options={'iprint': 10})

print(res)

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
