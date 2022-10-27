
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


lig = Ligand('1FIN_ligand.sdf')
lig_coord_gt = lig.get_coordinates(0)

lig2 = Ligand('1FIN_ligand.sdf')
lig2.generate_conformers(1000)

for i in range(1000):
    lig_coord = lig2.get_coordinates(i)
    print(f'{i}, {Kabsch_RMSD(lig_coord, lig_coord_gt)[0]:.3f} A')
    # Kabsch align

#Chem.MolToPDBFile(lig2.molecule, 'lig2.pdb') 
