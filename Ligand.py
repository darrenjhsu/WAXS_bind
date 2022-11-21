
import numpy as np
from saxstats import saxstats
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from Geometry import *
from rdkit.Geometry import Point3D

class Ligand:
    def __init__(self, mol, addHs=False):
        # mol can be filename of pdb, sdf, or a SMILES string
        if '.pdb' in mol:
            self.molecule = Chem.MolFromPDBFile(mol)
            self.molecule2 = Chem.MolFromPDBFile(mol) # This is for scipy.optimization, where we set coordinates to compute energy
        elif '.sdf' in mol:
            self.molecule = Chem.MolFromMolFile(mol)
            self.molecule2 = Chem.MolFromMolFile(mol)
        else:
            self.molecule = Chem.MolFromSmiles(mol)
            self.molecule2 = Chem.MolFromMolFile(mol)

        if addHs:
            self.molecule = Chem.AddHs(self.molecule, addCoords=True)
            self.molecule2 = Chem.AddHs(self.molecule2, addCoords=True)

        self.elements = np.array([self.molecule.GetAtoms()[x].GetSymbol() for x in range(self.molecule.GetNumAtoms())])
        self.hasManyH = np.sum([x == 'H' for x in self.elements]) / np.sum([x != 'H' for x in self.elements]) > 0.1
        if not self.hasManyH:
            print("This ligand does not seem to have hydrogens modeled into them")
            print("Consider doing so for best X-ray scattering signal accuracy")
        else:
            print("This ligand seems to have hydrogens modeled")
        self.electrons = np.array([saxstats.electrons.get(x, 6) for x in self.elements])
        self.num_conformers = self.molecule.GetNumConformers()
        self.num_atoms = len(self.elements)

        self.process_bonds()
        MMFF_pro = AllChem.MMFFGetMoleculeProperties(self.molecule2) # pro stands for property
        self.MMFF_lig = AllChem.MMFFGetMoleculeForceField(self.molecule2, MMFF_pro)
        self.partial_charge = np.array([MMFF_pro.GetMMFFPartialCharge(x) for x in range(self.num_atoms)])

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

    def transform(self, conformerID=0, structure_parameters=None, debug=False):
        # structure parameters should either be 5 or 5 + len(rgroup) values
        # x, y, z, theta [0, pi], phi [0, 2*pi], torsions [0, 2*pi]
        assert len(structure_parameters) == 6 or len(structure_parameters) == 6 + self.num_torsion
        if debug:
            print(structure_parameters)
        sp = structure_parameters.copy()
        sp[3:] *= 18
        coord = self.get_coordinates(conformerID=conformerID).copy()
        if len(sp) > 6:
            for idx, r in enumerate(sp[6:]):
                if debug:
                    print(f'Rotating atom group {self.rgroup[idx][1]} by {r} degrees')
                # transform by torsion
                coord[self.rgroup[idx][1]] = rotate_by_axis(coord[self.rgroup[idx][1]], coord[self.rgroup[idx][0][0]], coord[self.rgroup[idx][0][1]], r)
        coord = rotate_then_center(coord, make_rot_xyz(*sp[3:6]), np.array(sp[:3]))
        return coord  

    def calculate_energy(self, ligand_coords=None):
        if ligand_coords is not None:
            conf = self.molecule2.GetConformer(0)
            for i in range(self.num_atoms):
                conf.SetAtomPosition(i,Point3D(*ligand_coords[i]))
            
        return self.MMFF_lig.CalcEnergy()

def rot_group(bond, rbond):
    rgroup = []
    G = nx.Graph()
    G.add_edges_from(bond)
    for r in rbond:
        G.remove_edge(r[0], r[1])
        rgroup.append([r, np.array(list(nx.descendants(G, r[1]))).astype(int)])
        G.add_edge(r[0], r[1])
    return rgroup



