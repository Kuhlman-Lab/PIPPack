import os
import glob
import json
import copy
import random
import logging
from typing import *
from functools import partial
from Bio import SeqIO

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch_geometric.data import Data
from lightning import LightningDataModule

from data.featurizer import Featurizer
import data.residue_constants as rc
from utils.utils import download_file, extract, dir_size
from data.protein import from_pdb_file


log = logging.getLogger(__name__)


class Top2018Dataset(Dataset):
    """
    High-quality protein residues from high-quality, low redundancy protein chains in the PDB.
    Curated by the Richardson Lab at Duke University.
    See: https://doi.org/10.5281/zenodo.4626149 and https://doi.org/10.5281/zenodo.5115232
    
    Parameters:
        path (str): path to store the dataset
        filter_level (str): the filter level used to prune the dataset. Either "mc" for main-chain
            filtered residues or "full" for full-atom filtered residues.
        version (str): which version of the datasets to use. Available versions for "mc" are ["2.01"] and
            for "full" are ["2.0"].
        transform (Callable, optional): how to transform the protein dictionary. Usually constructs a ProteinGraph.
    """

    # Only includes latest repo versions (as of 04/2023)
    # Previous versions may be structured slightly differently
    top2018_versions = {
        "full": {
            '2.0': {
                'metadata_url': 'https://zenodo.org/record/5773255/files/top2018_metadata_full_filtered.csv',
                'metadata_md5': '385a2968247083a71e65e9a9e573d059',
                'pdb_tarball_url': 'https://zenodo.org/record/5773255/files/top2018_pdbs_full_filtered_hom90.tar.gz',
                'pdb_tarball_md5': 'af7f00ddad4d96a3d3436f669985a53e',
                'pdb_dir_size': 4998354114,
            }
        },
        "mc": {
            '2.01': {
                'metadata_url': 'https://zenodo.org/record/5777651/files/top2018_metadata_mc_filtered.csv',
                'metadata_md5': 'eecc82da39bed07f3d26bdb6d5272c2d',
                'pdb_tarball_url': 'https://zenodo.org/record/5777651/files/top2018_pdbs_mc_filtered_hom90.tar.gz',
                'pdb_tarball_md5': 'fd1671aa7e88e49335ff44e87d96883c',
                'pdb_dir_size': 6164619152,
            }
        }
    }
    
    def __init__(self,
        path: str,
        filter_level: str,
        version: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        self.path = os.path.expanduser(path)
        self.filter_level = filter_level
        self.version = version
        self.transform = transform
        
        # Download, verify, and preprocess data
        self.download()
        
        # Construct protein_info from metadata
        self.protein_info = self.metadata_to_protein_info()
        self.protein_id_to_index = self.get_id_to_index_mapping()
        self.clusters_to_protein_id = self.get_cluster_to_id_mapping()
        
    def download(self) -> None:
        # Create path for data
        os.makedirs(self.path, exist_ok=True)
        
        # Download and verify data
        log.info(f"Downloading and verifying Top2018 {self.filter_level} v{self.version} data!")
        version_info = self.top2018_versions[self.filter_level][self.version]
        self.metadata_file = download_file(version_info["metadata_url"], self.path, md5=version_info["metadata_md5"])
        pdb_tarball = download_file(version_info["pdb_tarball_url"], self.path, md5=version_info["pdb_tarball_md5"])
        pdb_dir = os.path.join(self.path, os.path.basename(pdb_tarball)[:-7])
        if not os.path.isdir(pdb_dir) or dir_size(pdb_dir) != version_info["pdb_dir_size"]:
            _ = extract(pdb_tarball)
        self.pdb_files = glob.glob(os.path.join(pdb_dir, "**", "*.pdb"), recursive=True)

    def metadata_to_protein_info(self) -> Sequence[Dict[str, Union[str, int, float]]]:
        # Read metadata csv
        with open(self.metadata_file, 'r') as f:
            lines = [line.split(",") for line in f.readlines()]

        # Extract information for each chain
        protein_info = {}
        for chain_info in lines[1:]:
            chain_ex = {
                "pdb_id": chain_info[0],
                "chain_id": chain_info[1],
                "exp_method": chain_info[4],
                "release_data": chain_info[5],
                "resolution": float(chain_info[6]),
                "cluster_95pc": int(chain_info[19]),
                "cluster_90pc": int(chain_info[20]),
                "cluster_70pc": int(chain_info[21]),
                "cluster_50pc": int(chain_info[22]),
                "cluster_40pc": int(chain_info[23]),
                "cluster_30pc": int(chain_info[24]),
            }
            protein_info[f"{chain_info[0].upper()}_{chain_info[1]}"] = chain_ex
        
        return protein_info

    def get_cluster_to_id_mapping(self) -> Dict[str, Dict[int, Sequence[int]]]:
        # Map each cluster id to each protein for each sequence identity level
        clusters = {
            "95pc": {},
            "90pc": {},
            "70pc": {},
            "50pc": {},
            "40pc": {},
            "30pc": {},
        }
        for protein in self.protein_info:
            # Skip the protein if we don't have a pdb file for it.
            if protein not in self.protein_id_to_index:
                continue
            
            for seq_id in clusters:
                cluster_id = self.protein_info[protein][f"cluster_{seq_id}"]
                if cluster_id not in clusters[seq_id]:
                    clusters[seq_id][cluster_id] = [protein]
                else:
                    clusters[seq_id][cluster_id].append(protein)
        
        return clusters
    
    def get_id_to_index_mapping(self) -> Dict[str, int]:
        # Map each protein id to its corresponding data idx
        protein_id_to_index = {}
        for idx, pdb in enumerate(self.pdb_files):
            pdb_id = os.path.basename(pdb)[:6]
            pdb_id = pdb_id[:4].upper() + pdb_id[4:]
            protein_id_to_index[pdb_id] = idx
        
        return protein_id_to_index

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, index: int) -> Dict[str, Union[str, int, float, torch.Tensor]]:
        # Grab the appropriate protein and load its info
        pdb_file = self.pdb_files[index]
        pdb_id = os.path.basename(pdb_file)[:6]
        pdb_id = pdb_id[:4].upper() + pdb_id[4:]
        protein = copy.deepcopy(self.protein_info[pdb_id])
        
        # Load the PDB structure.
        protein.update(vars(from_pdb_file(pdb_file, chain_id=protein["chain_id"], mse_to_met=True)))
        
        if self.transform:
            protein = self.transform(protein)
        
        return protein
    
    def idx_from_cluster(self, seq_id: str, cluster_id: Optional[int] = None, subcluster_idx: Optional[int] = None) -> int:
        # Get cluster info for specified seq_id
        cluster_dict = self.clusters_to_protein_id[seq_id]
        
        # Randomly select a cluster, if not provided.
        if cluster_id is None:
            cluster_id = random.choice(list(cluster_dict))
            
        # Randomly select a protein index within the cluster, if not provided.
        if subcluster_idx is None:
            subcluster_idx = random.choice(range(len(cluster_dict[cluster_id])))
        
        # Grab appropriate protein and its info
        protein_id = cluster_dict[cluster_id][subcluster_idx]
        protein_index = self.protein_id_to_index[protein_id]
        
        return protein_index
    
    def _prune_from_cluster_split(self, seq_id: str, cluster_subset: Sequence[int]) -> None:
        # Determine which proteins should be extracted based on specified clusters
        subset_ids = []
        for cluster in cluster_subset:
            subset_ids.extend(self.clusters_to_protein_id[seq_id].get(cluster, []))
        
        # Extract subset of proteins
        subset_pdb_files = [self.pdb_files[self.protein_id_to_index[protein_id]] for protein_id in subset_ids
                            if protein_id in self.protein_id_to_index]
        
        # Reassign proteins and indexes
        self.pdb_files = subset_pdb_files
        self.protein_id_to_index = self.get_id_to_index_mapping()
        
        # Update clusters
        self.clusters_to_protein_id[seq_id] = {cluster: self.clusters_to_protein_id[seq_id][cluster] for cluster in cluster_subset if cluster in self.clusters_to_protein_id[seq_id]}

    def _prune_from_removal_list(self, removal_list: Sequence[str]) -> None:
        # Extract subset of protein ids
        subset_protein_ids = [protein_id for protein_id in self.protein_id_to_index if protein_id not in removal_list]
        
        # Extract subset of pdb files
        subset_pdb_files = [self.pdb_files[self.protein_id_to_index[protein_id]] for protein_id in subset_protein_ids
                            if protein_id in self.protein_id_to_index]
        
        # Reassign proteins and indexes
        self.pdb_files = subset_pdb_files
        self.protein_id_to_index = self.get_id_to_index_mapping()
        
        # Update clusters
        to_pop = []
        for seq_id in self.clusters_to_protein_id:
            for cluster_id in self.clusters_to_protein_id[seq_id]:
                old_cluster = self.clusters_to_protein_id[seq_id][cluster_id]
                new_cluster = [protein for protein in old_cluster if protein not in removal_list]
                if len(new_cluster) == 0:
                    to_pop.append((seq_id, cluster_id))
                else:
                    self.clusters_to_protein_id[seq_id][cluster_id] = new_cluster
        for pop_item in to_pop:
            self.clusters_to_protein_id[pop_item[0]].pop(pop_item[1])

    def _to_fasta(self, out_file: str) -> None:
        
        # Loop over all pdb files in dataset and write to fasta file.
        with open(out_file, 'w') as f:
            for pdb_file in self.pdb_files:
                
                # Write the header
                pdb_basename = os.path.basename(pdb_file)
                f.write(f">{pdb_basename[:4].upper()}_{pdb_basename[5]}\n")
                
                # Write the sequence
                for record in SeqIO.parse(pdb_file, "pdb-seqres"):
                    if record.annotations["chain"] == pdb_basename[5]:
                        f.write(str(record.seq) + "\n")
                        break
            
            
class Top2018Sampler(Sampler):
    def __init__(self, dataset: Top2018Dataset, seq_id: str = "40pc", shuffle: bool = True) -> None:
        super().__init__(None)
        self.dataset = dataset
        self.seq_id = seq_id
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.dataset.clusters_to_protein_id[self.seq_id])

    def __iter__(self):
        # Create full batch by selecting one chain from each cluster
        batch = [
            self.dataset.idx_from_cluster(self.seq_id, cluster_id)
            for cluster_id in self.dataset.clusters_to_protein_id[self.seq_id]
        ]

        # Shuffle, if necessary
        if self.shuffle:
            random.shuffle(batch)
            
        return iter(batch)


def transform_structure(protein, n_chi_bin=36, crop_size=None, random_truncate=True, sc_d_mask_from_seq=False):
    if crop_size is not None:
        if protein['aatype'].shape[0] > crop_size:
            # Determine starting index for truncation.
            start_idx = 0
            if random_truncate:
                start_idx = random.choice(list(range(protein['aatype'].shape[0] - crop_size + 1)))
            
            # Truncate appropriate features.
            for feat in ['aatype', 'atom_positions', 'atom_mask', 'residue_index', 'chain_index', 'b_factors']:
                protein[feat] = protein[feat][start_idx:(start_idx + crop_size)]
                
    # Update residue_index based on chain_index (adding 100-residue gap)
    if len(np.unique(protein['chain_index'])) > 1:
        index_offset = 0
        for chain_idx in np.unique(protein['chain_index'])[:-1]:
            index_offset += max(protein['residue_index'][protein['chain_index'] == chain_idx])
            index_offset += 100
            protein['residue_index'][protein['chain_index'] == chain_idx + 1] += index_offset
    
    # Create all necessary residue features.
    X = protein["atom_positions"]
    S = protein["aatype"]
    L = protein["aatype"].shape[0]
    X_mask = protein["atom_mask"]
    residue_index = protein["residue_index"]
    chain_index = protein["chain_index"]
    BB_D, BB_D_mask = Featurizer.calc_bb_dihedrals(protein['atom_positions'], protein['residue_index'])
    SC_D, SC_D_mask = Featurizer.calc_sc_dihedrals(protein['atom_positions'], protein['aatype'])
    if sc_d_mask_from_seq:
        SC_D_mask = np.array(rc.chi_mask_atom14, dtype=np.float32)[protein["aatype"]]
    SC_D_BF_mask = Featurizer.chi_mask_from_b_factors(protein['aatype'], protein['b_factors'])
        
    # Create residue mask based on existance of backbone and apply it
    residue_mask = np.isfinite(np.sum(X[:, :4], axis=(-1, -2)))
    S = S * residue_mask
    X = X * residue_mask[..., None, None]
    X_mask = X_mask * residue_mask[..., None]
    residue_index = residue_index * residue_mask
    chain_index = chain_index * residue_mask
    BB_D = BB_D * residue_mask[..., None]
    BB_D_mask = BB_D_mask * residue_mask[..., None]
    SC_D = SC_D * residue_mask[..., None]
    SC_D_mask = SC_D_mask * residue_mask[..., None]
    SC_D_BF_mask = SC_D_BF_mask * residue_mask[..., None]
        
    # Create binned representation of sidechain dihedrals
    SC_D_bin = (SC_D + np.pi) // (2 * np.pi / n_chi_bin)
    SC_D_bin = np.nan_to_num(SC_D_bin) # Convert nans to 0 before masking and converting to int64
    SC_D_bin_offset = (SC_D + np.pi) % (2 * np.pi / n_chi_bin)
    SC_D_bin = (SC_D_bin * SC_D_mask + n_chi_bin * (1. - SC_D_mask))
    SC_D_bin_offset = SC_D_bin_offset * SC_D_mask
    
    # Lift dihedrals to unit circle with sin and cos
    BB_D_sincos = np.stack((np.sin(BB_D), np.cos(BB_D)), axis=-1)
    BB_D_sincos = BB_D_sincos * BB_D_mask[..., None]
    SC_D_sincos = np.stack((np.sin(SC_D), np.cos(SC_D)), axis=-1)
    SC_D_sincos = SC_D_sincos * SC_D_mask[..., None]
    
    # Create new batch dictionary
    protein_data = Data(
        num_nodes = L,
        S = torch.from_numpy(S).to(torch.int64), # [L]
        X = torch.from_numpy(X).to(torch.float32), # [L, 14, 3]
        X_mask = torch.from_numpy(X_mask).to(torch.float32), # [L, 14]
        residue_index = torch.from_numpy(residue_index).to(torch.int32), # [L]
        chain_index = torch.from_numpy(chain_index).to(torch.int32), # [L]
        residue_mask = torch.from_numpy(residue_mask).to(torch.float32), # [L]
        BB_D = torch.from_numpy(BB_D).to(torch.float32), # [L, 3]
        BB_D_sincos = torch.from_numpy(BB_D_sincos).to(torch.float32), # [L, 3, 2]
        BB_D_mask = torch.from_numpy(BB_D_mask).to(torch.float32), # [L, 3]
        SC_D = torch.from_numpy(SC_D).to(torch.float32), # [L, 4]
        SC_D_sincos = torch.from_numpy(SC_D_sincos).to(torch.float32), # [L, 4, 2]
        SC_D_bin = torch.from_numpy(SC_D_bin).to(torch.int64), # [L, 4]
        SC_D_bin_offset = torch.from_numpy(SC_D_bin_offset).to(torch.float32), # [L, 4]
        SC_D_mask = torch.from_numpy(SC_D_mask).to(torch.float32), # [L, 4]
        SC_D_BF_mask = torch.from_numpy(SC_D_BF_mask).to(torch.float32), # [L, 4]
    )
    
    # Remove any potential nans
    remove_nans = lambda x: torch.nan_to_num(x) if isinstance(x, torch.Tensor) else x
    protein_data = protein_data.apply(remove_nans)
    
    return protein_data


def collate_fn(protein_batch):

    # Padding function
    max_size = max([protein.num_nodes for protein in protein_batch])
    def _maybe_pad(protein, attr):
        attr_tensor = getattr(protein, attr)
        return F.pad(attr_tensor, [0, 0] * (len(attr_tensor.shape) - 1) + [0, max_size - protein.num_nodes])
    
    # Create batch by stacking all features
    batch = Data(
        num_proteins = len(protein_batch),
        num_res = sum(protein.num_nodes for protein in protein_batch),
        S = torch.stack([_maybe_pad(protein, "S") for protein in protein_batch]), # [B, L]
        X = torch.stack([_maybe_pad(protein, "X") for protein in protein_batch]), # [B, L, 14, 3]
        X_mask = torch.stack([_maybe_pad(protein, "X_mask") for protein in protein_batch]), # [B, L, 14]
        residue_index = torch.stack([_maybe_pad(protein, "residue_index") for protein in protein_batch]), # [B, L]
        chain_index = torch.stack([_maybe_pad(protein, "chain_index") for protein in protein_batch]), # [B, L]
        residue_mask = torch.stack([_maybe_pad(protein, "residue_mask") for protein in protein_batch]), # [B, L]
        BB_D = torch.stack([_maybe_pad(protein, "BB_D") for protein in protein_batch]), # [B, L, 3]
        BB_D_sincos = torch.stack([_maybe_pad(protein, "BB_D_sincos") for protein in protein_batch]), # [B, L, 3, 2]
        BB_D_mask = torch.stack([_maybe_pad(protein, "BB_D_mask") for protein in protein_batch]), # [B, L, 3]
        SC_D = torch.stack([_maybe_pad(protein, "SC_D") for protein in protein_batch]), # [B, L, 4]
        SC_D_sincos = torch.stack([_maybe_pad(protein, "SC_D_sincos") for protein in protein_batch]), # [B, L, 4, 2]
        SC_D_bin = torch.stack([_maybe_pad(protein, "SC_D_bin") for protein in protein_batch]), # [B, L, 4]
        SC_D_bin_offset = torch.stack([_maybe_pad(protein, "SC_D_bin_offset") for protein in protein_batch]), # [B, L, 4]
        SC_D_mask = torch.stack([_maybe_pad(protein, "SC_D_mask") for protein in protein_batch]), # [B, L, 4]
        SC_D_BF_mask = torch.stack([_maybe_pad(protein, "SC_D_BF_mask") for protein in protein_batch]), # [B, L, 4]
    )
    
    return batch


class Top2018DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        filter_level: str = "mc",
        version: str = "2.01",
        seq_id: str = "40pc",
        transform: Optional[Callable] = None,
        n_chi_bins: int = 72, 
        crop_size: Optional[int] = None,
        random_truncate: bool = True,
        removal_list_file: str = "",
        split_prefix: str = "chain_splits",
        data_split: Sequence[float] = [0.8, 0.1, 0.1],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_chains: int = -1,
    ) -> None:
        super().__init__()

        # Save init params
        self.save_hyperparameters(logger=False)

        # Datasets and transforms
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        if self.hparams.transform is None:
            self.transform = partial(transform_structure, n_chi_bin=self.hparams.n_chi_bins, crop_size=self.hparams.crop_size, random_truncate=self.hparams.random_truncate)
        
    @property
    def data_dir(self) -> str:
        return os.path.expanduser(
            os.path.join(
                self.hparams.data_dir,
                f"top2018_{self.hparams.filter_level}_v{self.hparams.version}"
            )
        )

    @property
    def split_file(self) -> str:
        return os.path.join(self.data_dir, f"{self.hparams.split_prefix}_{self.hparams.seq_id}.json")

    def prepare_data(self) -> None:
        log.info(f"Preparing Top2018 {self.hparams.filter_level} v{self.hparams.version} data!")
        Top2018Dataset(
            path=self.data_dir,
            filter_level=self.hparams.filter_level,
            version=self.hparams.version,
        )
        
    def setup(self, stage: str) -> None:
        # Load full dataset
        log.info(f"Setting up Top2018 {self.hparams.filter_level} v{self.hparams.version} data!")
        top2018 = Top2018Dataset(
            path=self.data_dir,
            filter_level=self.hparams.filter_level,
            version=self.hparams.version,
            transform=self.transform,
        )
        
        # Remove any chains specified in removal list
        if os.path.exists(os.path.expanduser(self.hparams.removal_list_file)):
            with open(os.path.expanduser(self.hparams.removal_list_file), 'r') as f:
                lines = f.readlines()
            removal_list = [line.strip() for line in lines if line.strip()]
            top2018._prune_from_removal_list(removal_list)
        
        # Create split file (if needed) and load it
        if not os.path.exists(self.split_file):
            _ = self._create_cluster_split(top2018)
        with open(self.split_file, 'r') as f:
            data_split = json.load(f)
        
        # Create datasets depending on stage
        if stage in ["fit", "all"]:
            # Training data
            self.data_train = copy.deepcopy(top2018)
            self.data_train._prune_from_cluster_split(self.hparams.seq_id, data_split["train"])
            if self.hparams.num_chains > 0 and self.hparams.num_chains < len(self.data_train.pdb_files):
                self.data_train.pdb_files = self.data_train.pdb_files[:self.hparams.num_chains]
                self.data_train.protein_id_to_index = self.data_train.get_id_to_index_mapping()
                self.data_train.clusters_to_protein_id = self.data_train.get_cluster_to_id_mapping()
        if stage in ["fit", "validate", "all"]:
            # Validation data
            self.data_val = copy.deepcopy(top2018)
            self.data_val._prune_from_cluster_split(self.hparams.seq_id, data_split["valid"])
            if self.hparams.num_chains > 0 and self.hparams.num_chains < len(self.data_val.pdb_files):
                self.data_val.pdb_files = self.data_val.pdb_files[:self.hparams.num_chains]
                self.data_val.protein_id_to_index = self.data_val.get_id_to_index_mapping()
                self.data_val.clusters_to_protein_id = self.data_val.get_cluster_to_id_mapping()
        if stage in ["test", "all"]:
            # Testing data
            self.data_test = copy.deepcopy(top2018)
            self.data_test._prune_from_cluster_split(self.hparams.seq_id, data_split["test"])
            if self.hparams.num_chains > 0 and self.hparams.num_chains < len(self.data_test.pdb_files):
                self.data_test.pdb_files = self.data_test.pdb_files[:self.hparams.num_chains]
                self.data_test.protein_id_to_index = self.data_test.get_id_to_index_mapping()
                self.data_test.clusters_to_protein_id = self.data_test.get_cluster_to_id_mapping()
    
    def _create_cluster_split(self, top2018_ds: Top2018Dataset) -> str:
        # Get all cluster ids
        cluster_dict = top2018_ds.clusters_to_protein_id[self.hparams.seq_id]
        cluster_ids = list(cluster_dict)
        
        # Randomly shuffle cluster_ids and assign to datasets.
        random.shuffle(cluster_ids)
        test_count = int(self.hparams.data_split[2] * len(cluster_ids))
        test_clusters = cluster_ids[:test_count]
        valid_count = int(self.hparams.data_split[1] * len(cluster_ids))
        valid_clusters = cluster_ids[test_count:(test_count + valid_count)]
        train_clusters = cluster_ids[(test_count + valid_count):]

        # Create data_splits which maps mode to list of cluster ids
        data_splits = {
            'train': train_clusters,
            'valid': valid_clusters,
            'test': test_clusters 
        }

        # Save chain_splits
        with open(self.split_file, 'w') as f:
            json.dump(data_splits, f)
            
        log.info(f"Created new Top2018 split: {self.split_file}")
        return self.split_file

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            sampler=Top2018Sampler(
                self.data_train, self.hparams.seq_id,
                shuffle=True),
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0)
