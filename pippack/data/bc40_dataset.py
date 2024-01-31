import os
import gzip
import io
import glob
import json
import copy
import random
import logging
from typing import *
from functools import partial
from Bio import SeqIO

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from pippack.utils.utils import download_file, extract, dir_size
from pippack.data.protein import from_pdb_file
from pippack.data.top2018_dataset import transform_structure, collate_fn


log = logging.getLogger(__name__)


class BC40Dataset(Dataset):

    ss3_file_info = {
        "url": "https://drug.ai.tencent.com/protein/bc40/bc40_ss3.tar.gz",
        "md5": "ecdd11d6563c017aa94ff72773fd265d",
        "dir_size": 10034170
    }
    pdb_dir_info = {
        "dir_size": 8262294903,
        "bad_pdbs": ["5SZS", "5EXC", "3K1Q", "6TKZ"],
        "bad_chains": ["3VKH_C"]
    }
    
    def __init__(self,
        path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        self.path = os.path.expanduser(path)
        self.transform = transform
        
        # Download, verify, and preprocess data
        self.download()
        
    def download(self) -> None:
        # Create path for data
        os.makedirs(self.path, exist_ok=True)
        
        log.info(f"Downloading and verifying BC40 data!")
        # Download and verify ss3 data
        ss3_file = download_file(self.ss3_file_info["url"], self.path, md5=self.ss3_file_info["md5"])
        ss3_dir = os.path.join(self.path, os.path.basename(ss3_file)[:-7])
        if not os.path.isdir(ss3_dir) or dir_size(ss3_dir) != self.ss3_file_info["dir_size"]:
            _ = extract(ss3_file)
        self.ss3_files = glob.glob(os.path.join(ss3_dir, "**", "*.ss3"), recursive=True)
        
        # Construct protein_ids from ss3_files
        protein_ids = [os.path.basename(ss3_file)[:5] for ss3_file in self.ss3_files]
        self.protein_ids = [protein[:4].upper() + "_" + protein[4] for protein in protein_ids]
        self.protein_ids = [protein_id for protein_id in self.protein_ids if protein_id[:4] not in self.pdb_dir_info["bad_pdbs"]]
        self.protein_ids = [protein_id for protein_id in self.protein_ids if protein_id not in self.pdb_dir_info["bad_chains"]]
        
        # Download and verify pdb data
        # NOTE: THIS TAKES A LONG TIME TO DOWNLOAD.
        pdb_dir = os.path.join(self.path, "bc40_pdb")
        self.protein_id_to_pdb_file = {}
        if not os.path.isdir(pdb_dir) or dir_size(pdb_dir) != self.pdb_dir_info["dir_size"]:
            os.makedirs(pdb_dir, exist_ok=True)
            for protein_id in self.protein_ids:
                pdb_id = protein_id[:4]
                _ = download_file(f"https://files.rcsb.org/download/{pdb_id}.pdb.gz", pdb_dir)
                self.protein_id_to_pdb_file[protein_id] = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
        else:
            for protein_id in self.protein_ids:
                pdb_id = protein_id[:4]
                self.protein_id_to_pdb_file[protein_id] = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> Dict[str, Union[str, int, float, torch.Tensor]]:
        # Grab the appropriate protein and load its info
        protein_id = self.protein_ids[index]
        pdb_file = self.protein_id_to_pdb_file[protein_id]
        protein = {
            "name": protein_id,
        }
        
        # Load the PDB structure.
        protein.update(vars(from_pdb_file(pdb_file, chain_id=protein_id[-1], mse_to_met=True)))
        
        if self.transform:
            protein = self.transform(protein)
        
        return protein
    
    def _prune_from_id_split(self, id_subset: Sequence[str]) -> None:
        # Reassign protein_ids and protein_id_to_pdb_file
        self.protein_ids = [protein_id for protein_id in id_subset if protein_id in self.protein_ids]
        self.protein_id_to_pdb_file = {protein_id: self.protein_id_to_pdb_file[protein_id]
                                       for protein_id in id_subset
                                       if protein_id in self.protein_id_to_pdb_file}

    def _prune_from_removal_list(self, removal_list: Sequence[str]) -> None:
        # Extract and prune subset of protein ids
        subset_protein_ids = [protein_id for protein_id in self.protein_ids if protein_id not in removal_list]
        self._prune_from_id_split(id_subset=subset_protein_ids)

    def _to_fasta(self, out_file: str) -> None:
        # Loop over all protein ids in dataset and write to fasta file.
        with open(out_file, 'w') as f:
            for protein_id in self.protein_ids:
                # Write the header
                f.write(f">{protein_id}\n")
                
                # Write the sequence
                with gzip.open(self.protein_id_to_pdb_file[protein_id], "rb") as pdb:
                    pdb_str = pdb.read().decode()
                pdb_str = io.StringIO(pdb_str)
                for record in SeqIO.parse(pdb_str, "pdb-seqres"):
                    if record.annotations["chain"] == protein_id[-1]:
                        f.write(str(record.seq) + "\n")
                        break


class BC40DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
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
            os.path.join(self.hparams.data_dir, "bc40")
        )

    @property
    def split_file(self) -> str:
        return os.path.join(self.data_dir, f"{self.hparams.split_prefix}.json")

    def prepare_data(self) -> None:
        log.info(f"Preparing BC40 data!")
        BC40Dataset(
            path=self.data_dir,
        )
        
    def setup(self, stage: str) -> None:
        # Load full dataset
        log.info(f"Setting up BC40 data!")
        bc40 = BC40Dataset(
            path=self.data_dir,
            transform=self.transform,
        )
        
        # Remove any chains specified in removal list
        if os.path.exists(os.path.expanduser(self.hparams.removal_list_file)):
            with open(os.path.expanduser(self.hparams.removal_list_file), 'r') as f:
                lines = f.readlines()
            removal_list = [line.strip() for line in lines if line.strip()]
            bc40._prune_from_removal_list(removal_list)
        
        # Create split file (if needed) and load it
        if not os.path.exists(self.split_file):
            _ = self._create_id_split(bc40)
        with open(self.split_file, 'r') as f:
            data_split = json.load(f)
        
        # Create datasets depending on stage
        if stage in ["fit", "all"]:
            # Training data
            self.data_train = copy.deepcopy(bc40)
            self.data_train._prune_from_id_split(data_split["train"])
        if stage in ["fit", "validate", "all"]:
            # Validation data
            self.data_val = copy.deepcopy(bc40)
            self.data_val._prune_from_id_split(data_split["valid"])
        if stage in ["test", "all"]:
            # Testing data
            self.data_test = copy.deepcopy(bc40)
            self.data_test._prune_from_id_split(data_split["test"])
    
    def _create_id_split(self, bc40_ds: BC40Dataset) -> str:
        # Get all protein ids
        protein_ids = bc40_ds.protein_ids
        
        # Randomly shuffle protein_ids and assign to datasets.
        random.shuffle(protein_ids)
        test_count = int(self.hparams.data_split[2] * len(protein_ids))
        test_proteins = protein_ids[:test_count]
        valid_count = int(self.hparams.data_split[1] * len(protein_ids))
        valid_proteins = protein_ids[test_count:(test_count + valid_count)]
        train_proteins = protein_ids[(test_count + valid_count):]

        # Create data_splits which maps mode to list of cluster ids
        data_splits = {
            'train': train_proteins,
            'valid': valid_proteins,
            'test': test_proteins 
        }

        # Save chain_splits
        with open(self.split_file, 'w') as f:
            json.dump(data_splits, f)
            
        log.info(f"Created new BC40 split: {self.split_file}")
        return self.split_file

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
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
