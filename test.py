
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


pro = PDB('1FIN_apo.pdb')
lig = Ligand('1FIN_ligand.pdb')
print(lig.elements, lig.electrons)
pocket = PDB('test_fpocket/1FIN_apo_out/1FIN_apo_out.pdb')

t0 = time.time()
pv, pg, ps, lv, lg, ls = overlap_grid(pro, lig, conformerID=0, pocket=pocket, 
                                      grid_spacing=1.0, radius=1.0, write_grid=False, timing=True)
t1 = time.time()
print(f'Time elapsed: {t1-t0:.3f} s')

rmat = rotation_sampling(8)
num_conditions = 1 * len(rmat)
t0 = time.time()
t1 = time.time()
condition_idx = 0
XS_calc_num = 0
for confID in range(1):
    for idx, rot in enumerate(rmat):
        if idx > 0 and idx % 5 == 0:
            print(f'Conformation {confID}, orientation {idx}')
        pv, pg, ps, lv, lg, ls = overlap_grid(pro, lig, conformerID=confID, rotation=rot, pocket=pocket, 
                                              grid_spacing=1.0, radius=1.0, write_grid=False, timing=False, 
                                              printing=False)
        XS_calc_num += np.sum(pv)
        condition_idx += 1
        t2 = time.time()
        if (t2-t1) > 5:
            print(f'Elapsed: {t2-t0:.3f} s. Estimated remaining: {(num_conditions - condition_idx) * (t2-t0) / condition_idx}')
            t1 = t2


