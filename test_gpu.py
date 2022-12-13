
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


pro = PDB('1FIN_apo.pdb')
lig_gt = Ligand('1FIN_ligand.sdf', addHs=True)
lig_coord_gt = lig_gt.get_coordinates(0)
lig = Ligand('1FIN_ligand.sdf', addHs=True)
lig.generate_conformers(26)
lig.optimize_all_conformers()
print(lig.elements, lig.electrons)
pocket = PDB('1FIN_apo_out/1FIN_apo_out.pdb')

t0 = time.time()
pv, pg, ps, lv, lg, ls = overlap_grid(pro, lig, conformerID=0, pocket=pocket, 
                                      grid_spacing=1.0, radius=1.0, write_grid=False, timing=True)
t1 = time.time()
print(f'Time elapsed: {t1-t0:.3f} s')

scat = Scatter(c2 = 2)


S_calc_pro = scat.scatter(protein=pro)
S_calc_lig = scat.scatter(ligand=lig)
S_calc_complex_gt = np.array(scat.scatter(protein=pro, ligand=lig))


rmat = rotation_sampling(9)
test_conf_list = range(10)
num_conditions = len(test_conf_list) * len(rmat)
t0 = time.time()
t1 = time.time()
condition_idx = 0
XS_calc_num = 0
XS_calc_time = 0
rmsd_list = []
chi_list = []
other_info = []
for confID in test_conf_list:
    for idx, rot in enumerate(rmat):
        pv, pg, ps, lv, lg, ls = overlap_grid(pro, lig, conformerID=confID, rotation=rot, pocket=pocket, 
                                              grid_spacing=1.0, radius=1.0, write_grid=False, timing=False, 
                                              printing=False)
        if idx > 0 and idx % 5 == 0:
            print(f'Conformation {confID}, orientation {idx}, {np.sum(pv)} points to try')

        for xyz in pg[pv.flatten()]:
            ligand_coords = rotate_then_center(lig.get_coordinates(confID), rot, xyz)
            ts0 = time.time()
            S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=ligand_coords))
            ts1 = time.time()
            XS_calc_time += (ts1-ts0)
            rmsd = pureRMSD(ligand_coords, lig_coord_gt) #np.sqrt(np.mean(((ligand_coords - lig_coord_gt)**2).sum(1)))
            #chi = np.sum(((np.log(S_calc_complex_gt) - np.log(S_calc)) / (np.log(0.003 * S_calc_complex_gt)))**2)
            chi = np.sum(((S_calc_complex_gt - S_calc) / (0.003 * S_calc_complex_gt))**2)
            rmsd_list.append(rmsd)
            chi_list.append(chi)
            other_info.append([confID, idx, *xyz])
            
        XS_calc_num += np.sum(pv)
        condition_idx += 1
        t2 = time.time()
        if (t2-t1) > 5:
            print(f'Elapsed: {t2-t0:.3f} s. Estimated remaining: {(num_conditions - condition_idx) * (t2-t0) / condition_idx:.3f} s')
            t1 = t2

rmsd_list = np.array(rmsd_list)
chi_list = np.array(chi_list)
good_idx = np.argsort(rmsd_list)
print('Sort by RMSD')
for idx in good_idx:
    print(f'{rmsd_list[idx]:10.3f}, {chi_list[idx]:10.3f}, {other_info[idx]}')

good_idx = np.argsort(chi_list)
print('Sort by chi2')
for idx in good_idx:
    print(f'{rmsd_list[idx]:10.3f}, {chi_list[idx]:10.3f}, {other_info[idx]}')

print(f'We ran {XS_calc_num} simulations at {XS_calc_time / XS_calc_num * 1000:.3f} ms per calc')
