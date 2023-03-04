

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
import time
from xs_helper import xray_scatter 
from array import array
from scipy.optimize import *
from spyrmsd import rmsd as srmsd

np.random.seed(42)

pro = PDB('1FIN_apo.pdb', generate_params=True)
lig_gt = Ligand('1FIN_ligand.sdf', addHs=False)
lig_coord_gt = lig_gt.get_coordinates(0)
lig = Ligand('1FIN_ligand.sdf', addHs=False)
sp_np = {'t': np.random.random(3) * 0,
         'r': np.random.random(3) * 90,
         'a': np.random.random(lig_gt.num_angle) * 10,
         'd': np.random.random(lig_gt.num_torsion) * 180}

lig.generate_conformers(1)
#lig.set_coordinates(lig.transform(0, sp_np))
AllChem.MMFFOptimizeMolecule(lig.molecule)

pocket = PDB('1FIN_apo_out/1FIN_apo_out.pdb')

#print('Ligand energy')
#print(lig.calculate_energy(lig.get_coordinates(0)))
#print(lig_gt.calculate_energy())
#exit()

scat = Scatter(q = np.linspace(0.03, 1.5, 80))

S_calc_complex_gt = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig_coord_gt))
#print(np.sum(((S_calc_complex_gt) / (0.003 * S_calc_complex_gt))**2))


def x_to_dict(x, lig):
    return {'t': x[:3], 'r': x[3:6] * 18, 'a': x[6:6+lig.num_angle], 'd': x[6+lig.num_angle:] * 18}
def myminfunc_RMSD(x, lig, lig_coord_gt):
    x_dict = x_to_dict(x, lig)
    return pureRMSD(lig.transform(0, x_dict), lig_coord_gt)
def myminfunc_symmRMSD(x, lig, lig_coord_gt, isomorphism=None, return_isomorphism=False):
    x_dict = x_to_dict(x, lig)
    if return_isomorphism:
        return symmRMSD(lig.transform(0, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)
    else:
        return symmRMSD(lig.transform(0, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)[0]

rmsd, iso = myminfunc_symmRMSD(np.zeros(6+lig.num_angle+lig.num_torsion),
                          lig, lig_coord_gt, isomorphism=None, return_isomorphism=True)

#res_RMSD = minimize(myminfunc_symmRMSD, np.zeros(6+lig.num_torsion+lig.num_angle), 
#                    args=(lig, lig_coord_gt, iso), method='L-BFGS-B', options={'iprint':0})
#
#print('Ground Truth')
#print(res_RMSD.x)
#print()
#print('Ground Truth RMSD')
#print(res_RMSD.fun)
#print()

#S_calc_RMSD_gt = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(0, x_to_dict(res_RMSD.x, lig))))

#def myminfunc_overlap(x, lig, pro):
#    new_ligand_coords = lig.transform(0, x_to_dict(x, lig))
#    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
#    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt - S_calc) / (0.003 * S_calc_complex_gt))**2)))
#    op, ei = overlap(protein=pro, ligand=lig, ligand_coords=new_ligand_coords)
#    #overlap_penalty = 50 / (1 + np.exp(-op + 10))
#    overlap_penalty=op
#    if np.random.random(1) < 0.5:
#        print(f'{overlap_penalty:10.3f}, {pureRMSD(lig_coord_gt, new_ligand_coords):10.3f}')
#    return overlap_penalty #+ eia + ce
#
#sp_overlap = np.zeros(6+lig.num_angle+lig.num_torsion)
#
#spb = [(0, 0) if idx > 5 else (None, None) for idx in range(6+lig.num_angle+lig.num_torsion)] # only translation and global rotation allowed
#res_overlap = minimize(myminfunc_overlap, sp_overlap, args=(lig, pro), method='L-BFGS-B', 
#               bounds=spb, 
#               options={'iprint': 10, 'eps':0.01, 'ftol': 1e-6, 'gtol': 1e-3}) 
#
#lig.set_coordinates(lig.transform(0, x_to_dict(res_overlap.x, lig)))

def myminfunc_scatter(x, lig, pro, scat, S_calc_complex_gt, lig_coord_gt, isomorphism=None, xyzfh=None):
    new_ligand_coords = lig.transform(0, x_to_dict(x, lig))
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt[:len(S_calc)] - S_calc) / (0.003 * S_calc_complex_gt[:len(S_calc)]))**2)))
    cs_temp = chi_squared**2
    if xyzfh is not None:
        xyzfh.write(f'{len(lig.elements)}\n\n')
        for ele, xyz in zip(lig.elements, new_ligand_coords):
            # Write xyz to the file handle
            xyzfh.write(f'{ele} {xyz[0]} {xyz[1]} {xyz[2]}\n')
    op, ei = overlap(protein=pro, ligand=lig, ligand_coords=new_ligand_coords)
    #eia = ei * 332.07 / 10 # electronic interaction adjusted
    #ce = lig.calculate_energy(new_ligand_coords)
    #print(chi_squared)
    #chi_squared = 200 / (1 + np.exp(-chi_squared / 20 + 5))
    chi_squared = np.log2(chi_squared)
    overlap_penalty = 10 / (1 + np.exp(-op/2 + 10))
    #overlap_penalty = 0
    if np.random.random(1) < 0.05:
        #print(f'{chi_squared**2:10.3f}, {chi_squared:10.3f}, {op:10.3f}, {eia:10.3f}, {ce:10.3f}, {pureRMSD(lig_coord_gt, new_ligand_coords):10.3f}')
        print(f'{cs_temp:10.3f}, {chi_squared:10.3f}, {overlap_penalty:10.3f}, {symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, isomorphism)[0]:10.3f}')
    return chi_squared + overlap_penalty #+ eia + ce


#sp0 = np.concatenate([pocket.pdb.coords[pocket.elements=='Ve'].mean(0), [0] * (3+lig.num_torsion)])
#sp0 = np.concatenate([res_RMSD.x[:3] + np.random.random(3) * 2 -1, np.random.random(3)*6-3, np.random.random(lig.num_torsion)*10-5])
sp0 = np.concatenate([lig_coord_gt.mean(0), np.zeros(3 + lig.num_torsion + lig.num_angle)])
print('Initial Guess')
print(sp0)
print()
#spb = [(x-10, x+10) if idx < 3 else (None, None) for idx, x in enumerate(sp0)]
spb = [(x-10, x+10) for x in sp0[:3]] + [(-10, 10)] * 3 + [(-10, 10)] * lig.num_angle + [(-10, 10)] * lig.num_torsion
#spb = [(0, 0) if idx > 5 else (None, None) for idx, x in enumerate(sp0)]
print('Bounds')
print(spb) 

fh = open('test_traj.xyz','w')

segs = 1
for ii in np.linspace(len(S_calc_complex_gt)/segs, len(S_calc_complex_gt), num=segs, dtype=int):
    print(f'\n\nUsing q range {scat.q[0]:.3f} to {scat.q[ii-1]:.3f} to fit conformations ...\n\n')
    scat_trial = Scatter(q = scat.q[:ii])
    res = minimize(myminfunc_scatter, sp0, args=(lig, pro, scat_trial, S_calc_complex_gt, lig_coord_gt, iso, fh), 
               #method='CG', #
               method='L-BFGS-B', 
               bounds=spb, 
               options={'iprint': 50, 'eps':0.02, 'gtol': 1e-3})
    sp0 = res.x 

fh.close()

def earlystop(x, f, accept):
    print(f'Current f is {f}')
    if np.exp(f) < 50:
        return True
    else:
        return False


#res_pre = basinhopping(myminfunc_scatter, sp0, niter=50, stepsize=4, T=0.2, disp=True, interval=5,
#                    minimizer_kwargs={
#                        'method':'L-BFGS-B', 
#                        'args': (lig, pro, scat, S_calc_complex_gt, lig_coord_gt, iso), 
#                        'bounds': spb, 
#                        'options': {'iprint': 50, 'eps':0.01, 'ftol': 1e-6, 'gtol': 1e-3}},
#                    callback=earlystop) 

#res = minimize(myminfunc_scatter, res_pre.x, args=(lig, pro, scat, S_calc_complex_gt), method='L-BFGS-B', 
#               bounds=spb, 
#               options={'iprint': 10, 'eps':0.001})#, 'ftol': 1e-6, 'gtol': 1e-3}) 

S_calc_min_chi = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(0, x_to_dict(res.x, lig))))

print("chi2 minimization")
print(res)
#print()
#print("RMSD minimization")
#print(res_RMSD)

print("Initial Kabsch RMSD:", end=' ')
print(f'{Kabsch_RMSD(lig_coord_gt, lig.get_coordinates())[0]:.3f} A')

print("Final RMSD:", end=' ')
print(f'{pureRMSD(lig_coord_gt, lig.transform(0, x_to_dict(res.x, lig))):.3f} A')

print("Final Kabsch RMSD:", end=' ')
print(f'{Kabsch_RMSD(lig_coord_gt, lig.transform(0, x_to_dict(res.x, lig)))[0]:.3f} A')

#print('chi2 at low RMSD:', end=' ')
#print(f'{np.sum(((S_calc_complex_gt - S_calc_RMSD_gt) / (0.003 * S_calc_complex_gt))**2):.3f}', end=' ')
#print(f'at {res_RMSD.fun:.3f} A and {symmRMSD(lig.transform(0, x_to_dict(res_RMSD.x, lig)), lig_coord_gt, lig.A, lig.G, iso)[0]:.3f} A')

print('chi2 at low chi2:', end=' ')
print(f'{np.sum(((S_calc_complex_gt - S_calc_min_chi) / (0.003 * S_calc_complex_gt))**2):.3f}', end=' ')
print(f'at {symmRMSD(lig.transform(0, x_to_dict(res.x, lig)), lig_coord_gt, lig.A, lig.G, iso)[0]:.3f} A')
#print(f'with overlap = {overlap(protein=pro, ligand=lig, ligand_coords=lig.transform(0, res.x)):.2f}')


# output result
lig.set_coordinates(lig.transform(0, x_to_dict(res.x, lig)))
#Chem.MolToPDBFile(lig.molecule, 'refined_ligand.pdb')
