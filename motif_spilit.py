import pickle
import os
import torch

import rdkit.Chem as Chem
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import HeteroData, InMemoryDataset
from motif_extract import mol_motif, motif_graph

import importlib

importlib.reload(mol_motif)


class MoleculeMotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        # 用于计算向量用的
        self.ATOM_LIST = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self.atom_to_index = {atom: i for i, atom in enumerate(self.ATOM_LIST)}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_file_names(self):
        # 如果没有这个文件，那么就会运行process函数
        return [f'{self.name}_hetero.pt']

    def download(self):
        _ = MoleculeNet(root=self.raw_dir, name=self.name)


    def get_atom_vector(self,mol, atom_indices):
        """统计原子种类向量"""
        vec = torch.zeros(len(self.ATOM_LIST))
        for idx in atom_indices:
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            if symbol in self.atom_to_index:
                vec[self.atom_to_index[symbol]] += 1
        return vec


    def process(self):
        dataset = MoleculeNet(root=self.raw_dir, name=self.name)

        hetero_data_list = []
        for i, data in enumerate(dataset):
            try:

                smiles = data.smiles
                mol = Chem.MolFromSmiles(smiles)

                # mol is invalid
                if mol is None:
                    continue

                # Extract motifs:multiple intersection between motifs
                motifs_type, motifs_node = mol_motif.mol_get_motif(mol)
                motif_smiles = mol_motif.get_motif_smiles(mol, motifs_node)

                if motif_graph.get_motif_edge(data, motifs_node) is None:
                    continue
                motif_edge_index, motif_edge_attr = motif_graph.get_motif_edge(data, motifs_node)

                # Skip if no valid motifs found:no motif_edge
                if len(motifs_node) == 1 or len(motifs_node) == 0 or motif_edge_index.numel() == 0:
                    continue

                hetero_data = HeteroData()

                # Add molecule level information
                hetero_data['mol'].y = data.y
                hetero_data['mol'].smiles = data.smiles

                # Add atom level information
                hetero_data['atom'].x = data.x
                hetero_data['atom', 'bond', 'atom'].edge_index = data.edge_index
                hetero_data['atom', 'bond', 'atom'].edge_attr = data.edge_attr

                hetero_data['motif'].type = torch.tensor(motifs_type,dtype=torch.long)
                hetero_data['motif'].smiles = motif_smiles
                hetero_data['motif'].x = torch.zeros(size=(len(motifs_type), 5))
                hetero_data['motif'].vector = torch.stack([self.get_atom_vector(mol,motif_node) for motif_node in motifs_node])

                # Create mapping from motif to atoms and build internal atom connections
                motif_internal_edge_src = []
                motif_internal_edge_dst = []
                motif_internal_edge_attr = []

                # Extract original molecular bond information
                mol_bonds = {}
                for bond_idx in range(data.edge_index.size(1)):
                    src, dst = data.edge_index[:, bond_idx].tolist()
                    attr = data.edge_attr[bond_idx]
                    mol_bonds[(src, dst)] = attr
                    mol_bonds[(dst, src)] = attr  # Add reverse direction too

                for motif_idx, atom_indices in enumerate(motifs_node):
                    if not atom_indices:  # Skip empty motifs
                        continue

                    # Add edges from motif to its atoms
                    src = torch.full((len(atom_indices),), motif_idx, dtype=torch.long)
                    dst = torch.tensor(list(atom_indices), dtype=torch.long)

                    # contain all atom and motif
                    if ('motif', 'contains', 'atom') not in hetero_data.edge_index_dict:
                        hetero_data['motif', 'contains', 'atom'].edge_index = torch.stack([src, dst])
                    else:
                        existing = hetero_data['motif', 'contains', 'atom'].edge_index
                        hetero_data['motif', 'contains', 'atom'].edge_index = torch.cat(
                            [existing, torch.stack([src, dst])], dim=1)

                    # Add edges from atom to motif (inverse relation)
                    if ('atom', 'in', 'motif') not in hetero_data.edge_index_dict:
                        hetero_data['atom', 'in', 'motif'].edge_index = torch.stack([dst, src])
                    else:
                        existing = hetero_data['atom', 'in', 'motif'].edge_index
                        hetero_data['atom', 'in', 'motif'].edge_index = torch.cat(
                            [existing, torch.stack([dst, src])], dim=1)

                    # Add internal motif connections (atom-to-atom within the same motif)
                    atom_list = list(atom_indices)
                    for idx1, atom1 in enumerate(atom_list):
                        for atom2 in atom_list[idx1 + 1:]:
                            # Check if there's a bond between these atoms in the original molecule
                            if (atom1, atom2) in mol_bonds:
                                motif_internal_edge_src.append(atom1)
                                motif_internal_edge_dst.append(atom2)
                                motif_internal_edge_attr.append(mol_bonds[(atom1, atom2)])

                                # Add reverse direction
                                motif_internal_edge_src.append(atom2)
                                motif_internal_edge_dst.append(atom1)
                                motif_internal_edge_attr.append(mol_bonds[(atom2, atom1)])

                # Add motif internal atom-atom edges if any were found
                if motif_internal_edge_src:
                    internal_edge_index = torch.stack([
                        torch.tensor(motif_internal_edge_src, dtype=torch.long),
                        torch.tensor(motif_internal_edge_dst, dtype=torch.long)
                    ])
                    internal_edge_attr = torch.stack(motif_internal_edge_attr)

                    hetero_data['atom', 'motif_internal', 'atom'].edge_index = internal_edge_index
                    hetero_data['atom', 'motif_internal', 'atom'].edge_attr = internal_edge_attr


                # Add motif-to-motif connections：

                hetero_data['motif', 'connects', 'motif'].edge_index = motif_edge_index
                src = hetero_data['motif'].type[motif_edge_index[0]].unsqueeze(1)   #
                dis = hetero_data['motif'].type[motif_edge_index[1]].unsqueeze(1)
                hetero_data['motif', 'connects', 'motif'].edge_attr = torch.cat([src,dis,motif_edge_attr],dim = -1)

                hetero_data_list.append(hetero_data)

                if i % 100 == 0:
                    print(f"Processed {i} molecules")

            except Exception as e:
                pass
                print(e)

        if self.pre_filter is not None:
            hetero_data_list = [data for data in hetero_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            hetero_data_list = [self.pre_transform(data) for data in hetero_data_list]

        # 合并
        data, slices = self.collate(hetero_data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    print("Starting dataset creation...")

    os.makedirs("./dataset", exist_ok=True)

    dataset = MoleculeMotifDataset(root="./dataset/", name="MUV")

    print(f"Dataset contains {len(dataset)} heterogeneous graphs")

    print("Sample heterogeneous graph structure:")
    sample_data = dataset[0]
    print(sample_data["motif"].type)
    print(sample_data["motif"].smiles)
    print(sample_data)

    n = len(dataset)
    indices = list(range(n))
    # random.shuffle(indices)

    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    split_dict = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }

    with open('./dataset/esol_split.pkl', 'wb') as f:
        pickle.dump(split_dict, f)

    print("Dataset creation and split complete!")
