
import sys
import importlib
import numpy as np
from saxstats import saxstats
from Ligand import Ligand
from WAXS import *
from PDB import PDB
from Geometry import *
import rdkit
from rdkit import Chem
import time
from xs_helper import xray_scatter 
from array import array


lig = Ligand('1FIN_ligand.sdf')
print(lig.get_coordinates())
sp = np.zeros(5+lig.num_torsion)
print("Zero everything")
print(lig.transform(structure_parameters=sp))
#sp[1] += 1
#print(lig.transform(structure_parameters=sp))
#sp[1] = 0
#sp[3] = 90
#print(lig.transform(structure_parameters=sp))
sp = np.zeros(5+lig.num_torsion)
sp[6] = 90
print(lig.transform(structure_parameters=sp, debug=True))
