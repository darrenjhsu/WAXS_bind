

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
from scipy.optimize import *

pro = PDB('1FIN_apo.pdb')
lig_gt = Ligand('1FIN_ligand.sdf', addHs=True)
lig_coord_gt = lig_gt.get_coordinates(0)
print(lig_coord_gt)
print(lig_gt.rbond, lig_gt.rgroup)
#lig_gt.set_coordinates(lig_gt.get_coordinates(0) + 1)
#print("Overlap")
#print(overlap(pro, lig_gt, timing=True))
#exit()
lig = Ligand('1FIN_ligand.sdf', addHs=True)
sp = np.random.random(6+lig.num_torsion) * 20 - 10

#lig.set_coordinates(lig.get_coordinates(0) + 10)
lig.generate_conformers(1)
lig.set_coordinates(lig.transform(0, sp))
#print(lig.elements, lig.electrons)
pocket = PDB('1FIN_apo_out/1FIN_apo_out.pdb')

scat = Scatter(c2=0)

S_calc_complex_gt = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig_coord_gt))
print(np.sum(((S_calc_complex_gt) / (0.003 * S_calc_complex_gt))**2))

def myminfunc_RMSD(x, lig, lig_coord_gt):
    return pureRMSD(lig.transform(0, x), lig_coord_gt)

res_RMSD = minimize(myminfunc_RMSD, np.zeros(6+lig.num_torsion), args=(lig, lig_coord_gt), method='L-BFGS-B', options={'iprint':10})

S_calc_RMSD_gt = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(0, res_RMSD.x)))

def myminfunc_scatter(x, lig, pro, scat, S_calc_complex_gt):
    new_ligand_coords = lig.transform(0, x)
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    chi_squared = np.log(np.sum(((S_calc_complex_gt - S_calc) / (0.003 * S_calc_complex_gt))**2))
    op = overlap(protein=pro, ligand=lig, ligand_coords=new_ligand_coords)
    overlap_penalty = 50 / (1 + np.exp(-op + 10))
    print(f'{np.exp(chi_squared):12.3f}, {chi_squared:10.3f}, {overlap_penalty:10.3f}')
    return chi_squared + overlap_penalty

print('Ground Truth')
print(res_RMSD.x)
print()

sp0 = np.concatenate([pocket.pdb.coords[pocket.elements=='Ve'].mean(0), [0] * (3+lig.num_torsion)])
#sp0 = np.concatenate([res_RMSD.x[:3], [0] * (3+lig.num_torsion)])
print('Initial Guess')
print(sp0)
print()
spb = [(x-10, x+10) if idx < 0 else (None, None) for idx, x in enumerate(sp0)]
print('Bounds')
print(spb) 

res = minimize(myminfunc_scatter, sp0, args=(lig, pro, scat, S_calc_complex_gt), method='L-BFGS-B', 
               bounds=spb, 
               options={'iprint': 10, 'eps':0.01})#, 'ftol': 1e-6, 'gtol': 1e-3}) 

def earlystop(x, f, accept):
    print(f'Current f is {f}')
    if np.exp(f) < 50:
        return True
    else:
        return False


#res_pre = basinhopping(myminfunc_scatter, sp0, niter=50, stepsize=4, T=0.2, disp=True, interval=5,
#                    minimizer_kwargs={
#                        'method':'L-BFGS-B', 
#                        'args': (lig, pro, scat, S_calc_complex_gt), 
#                        'bounds': spb, 
#                        'options': {'iprint': 10, 'eps':0.01, 'ftol': 1e-6, 'gtol': 1e-3}},
#                    callback=earlystop) 
#
#res = minimize(myminfunc_scatter, res_pre.x, args=(lig, pro, scat, S_calc_complex_gt), method='L-BFGS-B', 
#               bounds=spb, 
#               options={'iprint': 10, 'eps':0.001})#, 'ftol': 1e-6, 'gtol': 1e-3}) 

S_calc_min_chi = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(0, res.x)))

print("chi2 minimization")
print(res)
print()
print("RMSD minimization")
print(res_RMSD)

print("Initial Kabsch RMSD:", end=' ')
print(f'{Kabsch_RMSD(lig_coord_gt, lig.get_coordinates())[0]:.3f} A')

print("Final RMSD:", end=' ')
print(f'{pureRMSD(lig_coord_gt, lig.transform(0, res.x)):.3f} A')

print("Final Kabsch RMSD:", end=' ')
print(f'{Kabsch_RMSD(lig_coord_gt, lig.transform(0, res.x))[0]:.3f} A')

print('chi2 at low RMSD:', end=' ')
print(f'{np.sum(((S_calc_complex_gt - S_calc_RMSD_gt) / (0.003 * S_calc_complex_gt))**2):.3f}', end=' ')
print(f'at {res_RMSD.fun:.3f} A and {pureRMSD(lig_coord_gt, lig.transform(0, res_RMSD.x)):.3f} A')

print('chi2 at low chi2:', end=' ')
print(f'{np.sum(((S_calc_complex_gt - S_calc_min_chi) / (0.003 * S_calc_complex_gt))**2):.3f}', end=' ')
print(f'at {pureRMSD(lig_coord_gt, lig.transform(0, res.x)):.3f} A', end=' ')
print(f'with overlap = {overlap(protein=pro, ligand=lig, ligand_coords=lig.transform(0, res.x)):.2f}')

# output result
lig.set_coordinates(lig.transform(0, res.x))
Chem.MolToPDBFile(lig.molecule, 'refined_ligand.pdb')
