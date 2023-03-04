

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

np.random.seed(1010)

pro = PDB('1FIN_apo.pdb', generate_params=True)
lig_gt = Ligand('1FIN_ligand.sdf', addHs=False)
lig_coord_gt = lig_gt.get_coordinates(0)
lig = Ligand('1FIN_ligand.sdf', addHs=False)
sp_np = {'t': np.random.random(3) * 0,
         'r': np.random.random(3) * 180 - 90,
         'a': np.random.random(lig_gt.num_angle) * 10 - 5,
         'd': np.random.random(lig_gt.num_torsion) * 360 - 180}

#print(lig.rgroup)

#print('sp_np')
#print(sp_np)

#print('ligand coord pre')
#print(lig.get_coordinates(0))

num_conf = 1
lig.generate_conformers(num_conf)
#lig.set_coordinates(lig.transform(0, sp_np))
AllChem.MMFFOptimizeMolecule(lig.molecule)

#print('ligand coord post')
#print(lig.get_coordinates(0))

#pocket = PDB('1FIN_apo_out/1FIN_apo_out.pdb')


scat = Scatter(q = np.linspace(0.03, 1.5, 80), use_oa=1)

S_calc_complex_gt = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig_coord_gt))


def x_to_dict(x, lig):
    if len(x) == 5:
        return {'t': x[:3], 'r': np.array([x[3] * 9, x[4] * 18])}
    elif len(x) == 6:
        return {'t': x[:3], 'r': x[3:] * 18}
    elif len(x) == 5 + lig.num_angle + lig.num_torsion:
        return {'t': x[:3], 'r': np.array([x[3] * 9, x[4] * 18]), 'a': x[5:5+lig.num_angle], 'd': x[5+lig.num_angle:] * 18}
    else:
        return {'t': x[:3], 'r': x[3:6] * 18, 'a': x[6:6+lig.num_angle], 'd': x[6+lig.num_angle:] * 18}
def myminfunc_RMSD(x, lig, lig_coord_gt, conf_id=0, iso=None):
    x_dict = x_to_dict(x, lig)
    return pureRMSD(lig.transform(conf_id, x_dict), lig_coord_gt)
def myminfunc_symmRMSD(x, lig, lig_coord_gt, conf_id=0, isomorphism=None, return_isomorphism=False):
    x_dict = x_to_dict(x, lig)
    if return_isomorphism:
        return symmRMSD(lig.transform(conf_id, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)
    else:
        return symmRMSD(lig.transform(conf_id, x_dict), lig_coord_gt, lig.A, lig.G, isomorphism)[0]

rmsd, iso = myminfunc_symmRMSD(np.zeros(5+lig.num_angle+lig.num_torsion),
                          lig, lig_coord_gt, isomorphism=None, return_isomorphism=True)


def myminfunc(x, lig, lig_coord_gt, conf_id=0, isomorphism=None):
    return symmRMSD(lig.transform(conf_id, x_to_dict(x, lig)), lig_coord_gt, lig.A, lig.G, isomorphism=isomorphism)[0]

def myminfunc_scatter(x, lig, pro, scat, S_calc_complex_gt, lig_coord_gt, conf_id=0, isomorphism=None, xyzfh=None):
    #print(x)
    timing = np.random.random(1) < 0.00
    printing = np.random.random(1) < 0.00
    if timing:
        t0 = time.time()
    new_ligand_coords = lig.transform(conf_id, x_to_dict(x, lig))
    if timing:
        t1 = time.time()
        print(f'Transform = {(t1-t0)*1000:.2f} ms')
    if timing:
        t0 = time.time()
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    if timing:
        t1 = time.time()
        print(f'Scatter = {(t1-t0)*1000:.2f} ms')
    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt[:len(S_calc)] - S_calc) / (0.003 * S_calc_complex_gt[:len(S_calc)]))**2)))
    cs_temp = chi_squared**2
    if xyzfh is not None:
        if timing:
            t0 = time.time()
        xyzfh.write(f'{len(lig.elements)}\n\n')
        for ele, xyz in zip(lig.elements, new_ligand_coords):
            # Write xyz to the file handle
            xyzfh.write(f'{ele} {xyz[0]} {xyz[1]} {xyz[2]}\n')
        if timing:
            t1 = time.time()
            print(f'Writeout = {(t1-t0)*1000:.2f} ms')
    #op, ei = overlap(protein=pro, ligand=lig, ligand_coords=new_ligand_coords)
    #eia = ei * 332.07 / 10 # electronic interaction adjusted
    #ce = lig.calculate_energy(new_ligand_coords)
    #print(chi_squared)
    #chi_squared = 200 / (1 + np.exp(-chi_squared / 20 + 5))
    #chi_squared = np.log2(chi_squared)
    chi_squared = np.log2(cs_temp)
    #overlap_penalty = 10 / (1 + np.exp(-op/2 + 10))
    overlap_penalty = 0
    if printing:
        print(f'{cs_temp:10.3f}, {chi_squared:10.3f}, {overlap_penalty:10.3f}, {symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, isomorphism)[0]:10.3f}')
    return chi_squared + overlap_penalty #+ eia + ce

def reporting_sa(x, f, ctx):
    new_ligand_coords = lig.transform(conf_id, x_to_dict(x, lig))
    rmsd = symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, iso)[0]
    print(f'New rmsd is {rmsd:10.3f} A. ctx = {ctx}, Chi^2 = {2**f:.3f}') 

def reporting_ga(xk, convergence):
    new_ligand_coords = lig.transform(conf_id, x_to_dict(xk, lig))
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt[:len(S_calc)] - S_calc) / (0.003 * S_calc_complex_gt[:len(S_calc)]))**2)))
    rmsd = symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, iso)[0]
    print(f'New rmsd is {rmsd:10.3f} A. Chi^2 = {chi_squared**2:.3f}') 

def reporting_shgo(xk):
    new_ligand_coords = lig.transform(conf_id, x_to_dict(xk, lig))
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt[:len(S_calc)] - S_calc) / (0.003 * S_calc_complex_gt[:len(S_calc)]))**2)))
    rmsd = symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, iso)[0]
    print(f'New rmsd is {rmsd:10.3f} A. Chi^2 = {chi_squared**2:.3f}') 

def reporting_direct(xk):
    #print(x_to_dict(np.array(xk), lig))
    new_ligand_coords = lig.transform(conf_id, x_to_dict(np.array(xk), lig))
    S_calc = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=new_ligand_coords))
    chi_squared = np.sqrt((np.sum(((S_calc_complex_gt[:len(S_calc)] - S_calc) / (0.003 * S_calc_complex_gt[:len(S_calc)]))**2)))
    rmsd = symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, iso)[0]
    res_loc = get_direct_coord(xk, spb)
    print(f'Minimum found at coord {res_loc}')
    dist = np.sqrt(np.sum((res_rmsd_loc - res_loc) ** 2))
    print(f'New rmsd is {rmsd:10.3f} A. Chi^2 = {chi_squared**2:.3f}. Dist = {dist:.3f}') 

def reporting_direct_rmsd(xk):
    new_ligand_coords = lig.transform(conf_id, x_to_dict(np.array(xk), lig))
    rmsd = symmRMSD(new_ligand_coords, lig_coord_gt, lig.A, lig.G, iso)[0]
    print(f'New rmsd is {rmsd:10.3f} A. ') 

sp0 = np.concatenate([lig_coord_gt.mean(0), np.zeros(2 + lig.num_torsion + lig.num_angle)])
#sp0 = np.concatenate([lig_coord_gt.mean(0), 
#                      np.random.random(3) * 20 - 10, 
#                      np.random.random(lig_gt.num_angle) * 10 - 5,
#                      np.random.random(lig_gt.num_torsion) * 20 - 10])
#print('Initial Guess')
#print(sp0)
#print()
#print('Bounds')
#print(spb)
 
def simple_brute(func, ranges, args=(), Ns=20, finish=None):
    from itertools import product
    # simplified brute force optimization
    best_val = np.inf
    best_x = np.zeros(len(ranges))
    xranges = []
    num_ev = Ns**len(ranges)
    print(f'Attempting to run {num_ev:2e} evaluations')
    for r in ranges:
        xranges.append(np.linspace(r[0], r[1], num=Ns))
    t0 = time.time()
    counter = 0
    for trial_x in product(*xranges):
        val = func(np.array(trial_x), *args)
        counter += 1
        if val < best_val:
            best_val = val
            best_x = trial_x
        if counter % 1000 == 0:
            t1 = time.time()
            print(f'nev = {counter:7d}, time = {t1-t0:.2f} s')
            remain_time_s = num_ev / counter * (t1-t0)
            if remain_time_s > 31536000:
                print(f'Remaining time: {remain_time_s / 31536000:.1f} years')
            elif remain_time_s > 86400:
                print(f'Remaining time: {remain_time_s / 86400:.1f} days')
            elif remain_time_s > 3600: 
                print(f'Remaining time: {remain_time_s / 3600:.1f} hours')
            elif remain_time_s > 60: 
                print(f'Remaining time: {remain_time_s / 60:.1f} minutes')
            else: 
                print(f'Remaining time: {remain_time_s:.1f} seconds')

def get_direct_coord(x, spb):
    coord = []
    for x, b in zip(x, spb):
        coord.append((x - b[0]) / (b[1] - b[0]))
    return np.array(coord)

fh = None
#fh = open('test_traj.xyz','w')

segs = 1
method = 'differential_evolution'
method = 'dual_annealing'
method = 'direct'
#method = 'brute'
rmsd_method = 'direct'
rmsd_method = 'bfgs'
for conf_id in range(num_conf):
    Chem.MolToPDBFile(lig.molecule, 'initial_ligand.pdb')
    print(f'\nUsing conf_id = {conf_id} to do modeling\n') 
    spb = [(None, None) for x in sp0[:3]] + [(None, None)] * 2 + [(None, None)] * lig.num_angle + [(None, None)] * lig.num_torsion
    spb = [(x-5, x+5) for x in sp0[:3]] + [(None, None)] * 2 + [(None, None)] * lig.num_angle + [(None, None)] * lig.num_torsion
    spb = [(x-50, x+50) for x in sp0[:3]] + [(-1000, 1000)] * 2 + [(-500, 500)] * lig.num_angle + [(-1000, 1000)] * lig.num_torsion
    spb = [(x-5, x+5) for x in sp0[:3]] + [(-20, 20)] * 3 + [(-50, 50)] * lig.num_angle + [(-20, 20)] * lig.num_torsion
    if rmsd_method == 'direct':
        #spb = [(x-5, x+5) for x in sp0[:3]] + [(-20, 20)] * 3
        res_rmsd = direct(myminfunc, 
                            bounds=spb, 
                            args=(lig, lig_coord_gt, conf_id, iso),
                            callback=reporting_direct_rmsd,
                            vol_tol = 1e-24, 
                            )
    elif rmsd_method == 'bfgs':
        #spb = [(x-5, x+5) for x in sp0[:3]] + [(-20, 20)] * 3
        res_rmsd = minimize(myminfunc_RMSD, 
                            np.concatenate([lig_coord_gt.mean(0), np.zeros(3+lig.num_torsion+lig.num_angle)]), 
                            #np.concatenate([lig_coord_gt.mean(0), np.zeros(3)]), 
                            bounds=spb, 
                            args=(lig, lig_coord_gt, conf_id, iso),
                            method='L-BFGS-B', 
                            options={'iprint':50, 'maxfun': 100000}
                            )
    print(res_rmsd)
    xdict = x_to_dict(res_rmsd.x, lig)
    for rg, val in zip(lig.rgroup, xdict['d']):
        print(rg, val)
    for ag, val in zip(lig.agroup, xdict['a']):
        print(ag, val)
    lig.set_coordinates(lig.transform(0, x_to_dict(res_rmsd.x, lig)))
    Chem.MolToPDBFile(lig.molecule, 'refined_ligand.pdb')
    exit()
    print(f'Kabsch before {Kabsch_RMSD(lig_coord_gt, lig.get_coordinates(0))[0]:.3f} A')
    print(f'Kabsch after {Kabsch_RMSD(lig_coord_gt, lig.transform(0, x_to_dict(res_rmsd.x, lig)))[0]:.3f} A')
    spb = [(x-5, x+5) for x in sp0[:3]] + [(-5, 5)] + [(-10, 10)] + [(-5, 5)] * lig.num_angle + [(-10, 10)] * lig.num_torsion
    spb = [(x-5, x+5) for x in sp0[:3]] + [(-100, 100)] * 2 + [(-50, 50)] * lig.num_angle + [(-100, 100)] * lig.num_torsion
    for ii in np.linspace(len(S_calc_complex_gt)/segs, len(S_calc_complex_gt), num=segs, dtype=int):
        #print(f'\n\nUsing q range {scat.q[0]:.3f} to {scat.q[ii-1]:.3f} to fit conformations ...\n\n')
        scat_trial = Scatter(q = scat.q[:ii], use_oa=1)
        if method == 'differential_evolution':
            print(f'\n\nThere are {len(sp0)} variables, so GA will have a population of {len(sp0)*15}\n\n')
            res = differential_evolution(myminfunc_scatter, spb, 
                   args=(lig, pro, scat_trial, S_calc_complex_gt, lig_coord_gt, conf_id, iso, fh),
                   x0=sp0, callback=reporting_ga,)
        elif method == 'dual_annealing':
            res = dual_annealing(myminfunc_scatter, spb, 
                   args=(lig, pro, scat_trial, S_calc_complex_gt, lig_coord_gt, conf_id, iso, fh),
                   #no_local_search=False, 
                   x0=sp0, callback=reporting_sa,
                   minimizer_kwargs={'method': 'CG'})
        elif method == 'direct':
            res_rmsd_loc = get_direct_coord(res_rmsd.x, spb)
            print(res_rmsd.x)
            print(f'Optimal transformation is at {res_rmsd_loc}')
            res = direct(myminfunc_scatter, spb, 
                   args=(lig, pro, scat_trial, S_calc_complex_gt, lig_coord_gt, conf_id, iso, fh),
                   eps=1e-4, vol_tol=1e-32,
                   callback=reporting_direct,)
        elif method == 'brute':
            Ns = 5
            print(f'Brute force eval: There will be {Ns**len(spb):.2e} evaluations!!')
            res = simple_brute(myminfunc_scatter, spb, Ns=Ns, 
                    args=(lig, pro, scat_trial, S_calc_complex_gt, lig_coord_gt, conf_id, iso),)
        sp0 = res.x 
    S_calc_min_chi = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(conf_id, x_to_dict(res.x, lig))))
    print('chi2 at low chi2:', end=' ')
    print(f'{np.sum(((S_calc_complex_gt - S_calc_min_chi) / (0.003 * S_calc_complex_gt))**2):.3f}', end=' ')
    print(f'at {symmRMSD(lig.transform(conf_id, x_to_dict(res.x, lig)), lig_coord_gt, lig.A, lig.G, iso)[0]:.3f} A')

#fh.close()


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

exit()

S_calc_min_chi = np.array(scat.scatter(protein=pro, ligand=lig, ligand_coords=lig.transform(0, x_to_dict(res.x, lig))))

print("chi2 minimization")
print(res)
print(sp_np)
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
