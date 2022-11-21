
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
from rdkit.Chem import AllChem

try:
    num_conformers = int(sys.argv[1])
except:
    num_conformers = 50
print(f'Set number of conformers to {num_conformers}')

lig = Ligand('1FIN_ligand.sdf', addHs=True)
lig_coord_gt = lig.get_coordinates(0)

lig2 = Ligand('1FIN_ligand.sdf', addHs=True)
print('Generating confs ...')
lig2.generate_conformers(num_conformers)
print('Optimize all confs ...')
AllChem.MMFFOptimizeMoleculeConfs(lig2.molecule)

for i in range(num_conformers):
    lig_coord = lig2.get_coordinates(i)
    print(f'{i}, {Kabsch_RMSD(lig_coord, lig_coord_gt)[0]:.3f} A')
    # Kabsch align

#Chem.MolToPDBFile(lig.molecule, 'lig1.pdb') 
#Chem.MolToPDBFile(lig2.molecule, 'lig2.pdb') 
