
import numpy as np
from saxstats import saxstats
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from Geometry import *
from rdkit.Geometry import Point3D

class Ligand:
    def __init__(self, mol, addHs=False, removeHs=False):
        # mol can be filename of pdb, sdf, or a SMILES string
        if '.pdb' in mol:
            self.molecule = Chem.MolFromPDBFile(mol)
            self.molecule2 = Chem.MolFromPDBFile(mol) # This is for scipy.optimization, where we set coordinates to compute energy
        elif '.sdf' in mol:
            self.molecule = Chem.MolFromMolFile(mol, removeHs=removeHs)
            self.molecule2 = Chem.MolFromMolFile(mol, removeHs=removeHs)
        else:
            self.molecule = Chem.MolFromSmiles(mol)
            self.molecule2 = Chem.MolFromSmiles(mol)

        if addHs:
            self.molecule = Chem.AddHs(self.molecule, addCoords=True)
            self.molecule2 = Chem.AddHs(self.molecule2, addCoords=True)

        self.elements = np.array([self.molecule.GetAtoms()[x].GetSymbol() for x in range(self.molecule.GetNumAtoms())])
        self.heavy_atom_idx = np.array([x for x in range(self.molecule.GetNumAtoms()) if self.molecule.GetAtoms()[x].GetSymbol() != 'H'])
        self.hasManyH = np.sum([x == 'H' for x in self.elements]) / np.sum([x != 'H' for x in self.elements]) > 0.1
        if not self.hasManyH:
            print("This ligand does not seem to have hydrogens modeled into them")
            print("Consider doing so for best X-ray scattering signal accuracy")
        else:
            print("This ligand seems to have hydrogens modeled")
        self.electrons = np.array([saxstats.electrons.get(x, 6) for x in self.elements])
        self.num_conformers = self.molecule.GetNumConformers()
        self.num_atoms = len(self.elements)
        

        self.MMFF_pro = AllChem.MMFFGetMoleculeProperties(self.molecule) # pro stands for property
        self.MMFF_lig = AllChem.MMFFGetMoleculeForceField(self.molecule, self.MMFF_pro)
        self.partial_charge = np.array([self.MMFF_pro.GetMMFFPartialCharge(x) for x in range(self.num_atoms)])
        self.process_bonds()
        self.process_angles()
        self.process_graph()

    def generate_conformers(self, num_conformers=10):
        AllChem.EmbedMultipleConfs(self.molecule, numConfs=num_conformers)
        print(f'Generated {self.molecule.GetNumConformers()} conformers')

        self.num_conformers = self.molecule.GetNumConformers()
       
    def get_coordinates(self, conformerID=0):
        if conformerID >= self.num_conformers:
            raise ValueError(f'Conformer ID requested ({conformerID}) is larger than number of conformers available ({self.num_conformers})')
        return self.molecule.GetConformer(conformerID).GetPositions()

    def set_coordinates(self, coord, conformerID=0):
        if conformerID >= self.num_conformers:
            raise ValueError(f'Conformer ID requested ({conformerID}) is larger than number of conformers available ({self.num_conformers})')
        conf = self.molecule.GetConformer(conformerID)
        for i in range(self.num_atoms):
            conf.SetAtomPosition(i,Point3D(*coord[i]))

    def optimize_conformer(self, conformerID=0):
        AllChem.MMFFOptimizeMolecule(self.molecule, confID=conformerID)

    def optimize_all_conformers(self):
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)

    def process_bonds(self):
        RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        self.rbond = self.molecule.GetSubstructMatches(RotatableBond)
        self.bond = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in self.molecule.GetBonds()] 
        self.rgroup = rot_group(self.bond, self.rbond)
        self.num_torsion = len(self.rgroup)
        
    def process_graph(self):
        self.G = nx.Graph()
        self.G.add_edges_from(self.bond)
        self.A = self.electrons
        
    def process_angles(self):
        agroup_candidates = []
        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                for k in range(i+1, self.num_atoms):
                    if self.MMFF_pro.GetMMFFAngleBendParams(self.molecule, i,j,k) is not None:
                        # do something similar to rot_group with following criteria:
                        # 1. The molecule has to break into three parts
                        # 2. All parts should be > 1 atom
                        agroup_candidates.append(np.array([i,j,k]))
#                         print(i,j,k,self.MMFF_pro.GetMMFFAngleBendParams(self.molecule, i,j,k))
        self.agroup = angle_group(self.bond, agroup_candidates)
        self.num_angle = len(self.agroup)
        
    def transform(self, conformerID=0, structure_parameters=None, debug=False):
        # structure parameters should be a dict of np arrays:
        # t = [x, y, z] (translation)
        # r = [phi, theta], not [rotx, roty, rotz] (rotation)
        # a = [ang1, ang2, ...] (bond angles)
        # d = [dihe1, dihe2 ...] (dihedral angles / torsions)
        sp = structure_parameters.copy()
        for t, n in zip(['t', 'r', 'a', 'd'], [[3], [2, 3], [len(self.agroup)], [len(self.rgroup)]]):
            if t in sp:
                #print(t, n)
                assert len(sp[t]) in n
                sp[t] = np.array(sp[t])
            else:
                if t in ['t', 'r']:
                    sp[t] = np.zeros(3)
        if debug:
            print(structure_parameters)

        coord = self.get_coordinates(conformerID=conformerID).copy()
        
        # Transform by torsion
        if 'd' in sp:
            for idx, r in enumerate(sp['d']):
                if debug:
                    print(f'{idx}: ', end=' ')
                    print(f'Rotating atom group {self.rgroup[idx][1]} by {r} degrees using {self.rgroup[idx][0]} as axis')
                coord[self.rgroup[idx][1]] = rotate_by_axis(coord[self.rgroup[idx][1]], coord[self.rgroup[idx][0][0]], coord[self.rgroup[idx][0][1]], r)
        
        if 'a' in sp:
            # Transform by angle
            for idx, r in enumerate(sp['a']):
                if debug:
                    print(f'{idx}: ', end=' ')
                    print(f'Rotating atom group {self.agroup[idx][1]} and {self.agroup[idx][2]} by {r} degrees using {self.agroup[idx][0]} as axis')
                #coord[self.agroup[idx][1]], coord[self.agroup[idx][2]] = rotate_by_3pt(coord[self.agroup[idx][1]], coord[self.agroup[idx][2]], coord[self.agroup[idx][0]], r)
                coord[self.agroup[idx][1]] = rotate_by_3pt(coord[self.agroup[idx][0]], r, coord[self.agroup[idx][1]])
        if len(sp['r']) == 2:
            coord = rotate_then_center(coord, make_rot(*sp['r']), np.array(sp['t']))
        elif len(sp['r']) == 3:
            coord = rotate_then_center(coord, make_rot_xyz(*sp['r']), np.array(sp['t']))
        return coord  

    def calculate_energy(self, ligand_coords=None):
        if ligand_coords is not None:
            conf = self.molecule2.GetConformer(0)
            for i in range(self.num_atoms):
                conf.SetAtomPosition(i,Point3D(*ligand_coords[i]))
            
        return self.MMFF_lig.CalcEnergy()
    
    def save(self, fname):
        assert fname is not None, "Must provide filename in .pdb or .sdf format"
        if fname.split('.')[-1] == 'pdb':
            Chem.MolToPDBFile(self.molecule, fname)
        elif fname.split('.')[-1] == 'sdf':
            Chem.MolToMolFile(self.molecule, fname)
        else:
            print("File format not supported, use either pdb or sdf")

def rot_group(bond, rbond):
    rgroup = []
    G = nx.Graph()
    G.add_edges_from(bond)
    for r in rbond:
        G.remove_edge(r[0], r[1])
        subG = list(G.subgraph(c) for c in nx.connected_components(G))
        if len(subG) == 2:
            if len(list(nx.descendants(G, r[0]))) < len(list(nx.descendants(G, r[1]))):
                rgroup.append([r, np.array(list(nx.descendants(G, r[0]))).astype(int)])
            else:
                rgroup.append([r, np.array(list(nx.descendants(G, r[1]))).astype(int)])
            G.add_edge(r[0], r[1])
    return rgroup

def angle_group(bond, abond_candidate):
    agroup = []
    G = nx.Graph()
    G.add_edges_from(bond)
    for r in abond_candidate:
#         print(f'Processing candidate {r}')
        G.remove_edge(r[0], r[1])
        G.remove_edge(r[1], r[2])
        subG = list(G.subgraph(c) for c in nx.connected_components(G))
#         print(len(subG))
#         print(f'Graph is split to {len(subG)} subgraphs')
        if len(subG) == 3:
            for s in subG:
                if (r[1] in s):
                    continue
                if (len(s) == 1):
                    break
            else:
                # This is a valid bond angle
                #agroup.append([r, np.array(list(nx.descendants(G, r[0]) | {r[0]})).astype(int), np.array(list(nx.descendants(G, r[2]) | {r[2]})).astype(int)])
                if len(list(nx.descendants(G, r[0]))) < len(list(nx.descendants(G, r[2]))):
                    agroup.append([r, np.array(list(nx.descendants(G, r[0]) | {r[0]})).astype(int)])
                else:
                    agroup.append([r, np.array(list(nx.descendants(G, r[2]) | {r[2]})).astype(int)])
        G.add_edge(r[0], r[1])
        G.add_edge(r[1], r[2])
#     print(agroup)
    return agroup
                        
                    
          
            
            

