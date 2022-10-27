
import numpy as np
from saxstats import saxstats
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from Geometry import *

class Ligand:
    def __init__(self, mol):
        # mol can be filename of pdb, sdf, or a SMILES string
        if '.pdb' in mol:
            self.molecule = Chem.MolFromPDBFile(mol)
        elif '.sdf' in mol:
            self.molecule = Chem.MolFromMolFile(mol)
        else:
            self.molecule = Chem.MolFromSmiles(mol)
            self.molecule = Chem.AddHs(self.molecule, addCoords=True)


        self.elements = np.array([self.molecule.GetAtoms()[x].GetSymbol() for x in range(self.molecule.GetNumAtoms())])
        self.hasManyH = np.sum([x == 'H' for x in self.elements]) / np.sum([x != 'H' for x in self.elements]) > 0.1
        if not self.hasManyH:
            print("This ligand does not seem to have hydrogens modeled into them")
            print("Consider doing so for best X-ray scattering signal accuracy")
        self.electrons = np.array([saxstats.electrons.get(x, 6) for x in self.elements])
        self.num_conformers = self.molecule.GetNumConformers()
        self.num_atoms = len(self.elements)

        self.process_bonds()

    def generate_conformers(self, num_conformers=10):
        AllChem.EmbedMultipleConfs(self.molecule, numConfs=num_conformers)
        print(f'Generated {self.molecule.GetNumConformers()} conformers')

        self.num_conformers = self.molecule.GetNumConformers()
       
    def get_coordinates(self, conformerID=0):
        if conformerID >= self.num_conformers:
            raise ValueError(f'Conformer ID requested ({conformerID}) is larger than number of conformers available ({self.num_conformers})')
        return self.molecule.GetConformer(conformerID).GetPositions()

    def process_bonds(self):
        RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        self.rbond = self.molecule.GetSubstructMatches(RotatableBond)
        self.bond = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in self.molecule.GetBonds()]   
        self.rgroup = rot_group(self.bond, self.rbond)
        self.num_torsion = len(self.rgroup)

    def transform(self, conformerID=0, structure_parameters=None, sp=None, debug=False):
        # structure parameters should either be 5 or 5 + len(rgroup) values
        # x, y, z, theta [0, pi], phi [0, 2*pi], torsions [0, 2*pi]
        if structure_parameters is not None and sp is None:
            sp = structure_parameters # shorthand 
        assert len(sp) == 5 or len(sp) == 5 + self.num_torsion
        coord = self.get_coordinates(conformerID=conformerID).copy()
        coord = rotate_then_center(coord, make_rot(*sp[3:5]), np.array(sp[:3]))
        if len(sp) > 5:
            for idx, r in enumerate(sp[5:]):
                if debug:
                    print(f'Rotating atom group {self.rgroup[idx][1]} by {r} degrees')
                # transform by torsion
                coord[self.rgroup[idx][1]] = rotate_by_axis(coord[self.rgroup[idx][1]], coord[self.rgroup[idx][0][0]], coord[self.rgroup[idx][0][1]], r)
        return coord  

def rot_group(bond, rbond):
    rgroup = []
    G = nx.Graph()
    G.add_edges_from(bond)
    for r in rbond:
        G.remove_edge(r[0], r[1])
        rgroup.append([r, np.array(list(nx.descendants(G, r[1]))).astype(int)])
        G.add_edge(r[0], r[1])
    return rgroup



