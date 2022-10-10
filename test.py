
from saxstats import saxstats
from Ligand import Ligand
from WAXS import *
from PDB import PDB
import rdkit
from rdkit import Chem


pro = PDB('1FIN_apo.pdb')
#lig = Ligand('c1ccccc1CCC')
lig = Ligand('1FIN_ligand.pdb')
#lig.generate_conformers()
Chem.MolToPDBFile(lig.molecule, 'test.pdb')
print(lig.get_coordinates(0))

pv, ps, lv, ls = overlap_grid(pro, lig, grid_spacing=0.5, radius=1.0)

saxstats.write_mrc(pv, ps, 'test_protein_convolved.mrc')
saxstats.write_mrc(lv, ls, 'test_ligand.mrc')
