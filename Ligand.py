
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

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
        self.num_conformers = self.molecule.GetNumConformers()
        self.num_atoms = len(self.elements)

    def generate_conformers(self, num_conformers=10):
        AllChem.EmbedMultipleConfs(self.molecule, numConfs=num_conformers)
        print(f'Generated {self.molecule.GetNumConformers()} conformers')

        self.num_conformers = self.molecule.GetNumConformers()
       
    def get_coordinates(self, conformerID=0):
        if conformerID >= self.num_conformers:
            raise ValueError(f'Conformer ID requested ({conformerID}) is larger than number of conformers available ({self.num_conformers})')
        return self.molecule.GetConformer(conformerID).GetPositions()





