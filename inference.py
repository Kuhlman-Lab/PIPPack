import time, os
import logging
import glob, copy
import pickle
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
torch.cuda.empty_cache()
from typing import *
import lightning

# Library code
from utils.train_utils import load_checkpoint
from data.protein import to_pdb, Protein, from_pdb_file
from data.features import make_atom14_masks, atom14_to_atom37
from data.top2018_dataset import transform_structure, collate_fn
import data.residue_constants as rc


logger = logging.getLogger(__name__)


def replace_protein_sequence(protein, protein_name, new_seqs):
    proteins = []
    
    for i, seq in enumerate(new_seqs):
        # Replace protein sequence info
        new_protein = copy.deepcopy(protein)
        aatype = [rc.restype_order[res] for res in seq]
        new_protein["aatype"] = np.array(aatype).astype(np.int64)
        
        # Rebuild atom mask
        atom_mask = []
        for res in seq:
            res_atoms = rc.restype_name_to_atom14_names[rc.restype_1to3[res]]
            res_mask = [1 if atom != "" else 0 for atom in res_atoms]
            atom_mask.append(res_mask)
        new_protein["atom_mask"] = np.array(atom_mask).astype(np.float32)
        
        proteins.append((protein_name + f"_{i}", new_protein))
        
    return proteins


def sample_epoch(model, batch, temperature, device, n_recycle=0):    
    # Sampling epoch
    model.eval()
    with torch.no_grad():
        # Move to device
        batch = batch.to(device)
        
        # Sample the model
        results = model(batch, n_recycle=n_recycle)
        #results = model.sample(batch, temperature=temperature, n_recycle=n_recycle)

        # Compute metrics and add batch info
        #results.update(model.compute_loss(results, batch, True)[1])
        results.update(batch.to_dict())
        
    return results


def pdbs_from_prediction(sample_results) -> Sequence[str]:

    # Get the protein components.
    S = sample_results["S"]
    residue_index = sample_results["residue_index"]
    pred_xyz = sample_results["final_X"]
    
    # Convert atom14 coordinates to atom37 coordinates
    residx_atom37_to_atom14, atom37_atom_exists, _, _ = make_atom14_masks(S)
    pred_xyz = atom14_to_atom37(pred_xyz, residx_atom37_to_atom14, atom37_atom_exists)

    # Construct the components needed for the protein object
    proteins = []
    for i in range(S.shape[0]):
        aatype = S[i].cpu().numpy()
        atom_positions = pred_xyz[i].cpu().numpy()
        atom_mask = (np.sum(atom_positions, axis=-1) != 0.0).astype(np.int32)
        chain_index = np.zeros(aatype.shape)
        residue_idx = residue_index[i].cpu().numpy()
        b_factors = np.zeros(atom_mask.shape)

        protein = Protein(aatype=aatype, atom_positions=atom_positions, atom_mask=atom_mask, residue_index=residue_idx, 
                        chain_index=chain_index, b_factors=b_factors)

        protein_string = to_pdb(protein)
        proteins.append(protein_string)
    
    return proteins

@hydra.main(version_base=None, config_path="./config", config_name="inference")
def main(cfg: DictConfig) -> None:
    
    # Get the config used when running experiment
    with open(os.path.join(cfg.inference.exp_dir, '.hydra', 'config.pickle'), 'rb') as f:
        exp_cfg = pickle.load(f)
        
    # Set up RNG and device
    seed = lightning.seed_everything(cfg.inference.seed)
    logger.info(f"Using seed={seed} for RNG.")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not cfg.inference.force_cpu) else "cpu")
    
    # Load model with same config  
    model: torch.nn.Module = hydra.utils.instantiate(exp_cfg.model).to(device)
    
    # Find the best checkpoint to load into model
    best_checkpoint = glob.glob(os.path.join(cfg.inference.exp_dir, 'results', 'best_ckpt_*'))[0]

    # Load the best checkpoint
    load_checkpoint(best_checkpoint, model)

    # Get the dataset
    pdb_files = glob.glob(os.path.join(cfg.inference.pdb_path, '*.pdb'))
    if cfg.inference.get("replace_seqs", False):
        assert len(pdb_files) == 1
        
        fasta_files = glob.glob(os.path.join(cfg.inference.pdb_path, '*.fasta'))
        assert len(fasta_files) == 1
        
        with open(fasta_files[0], 'r') as f:
            lines = f.readlines()
        new_seqs = [line.strip() for line in lines if line[0] != ">" and line]
        
        proteins = replace_protein_sequence(vars(from_pdb_file(pdb_files[0], mse_to_met=True)), os.path.basename(pdb_files[0])[:-4], new_seqs)
    else: 
        proteins = [(os.path.basename(pdb_file)[:-4], vars(from_pdb_file(pdb_file, mse_to_met=True))) for pdb_file in pdb_files]
    
    # Transform proteins
    proteins = [(protein[0], transform_structure(protein[1], exp_cfg.model.n_chi_bins, sc_d_mask_from_seq=True)) for protein in proteins]
    
    # Form batches
    sorted_proteins = sorted(proteins, key=lambda x: x[1].S.shape[0])

    # Cluster into minibatches of similar sizes
    batches, minibatch = [], []
    for protein in sorted_proteins:
        if protein[1].S.shape[0] * (len(minibatch) + 1) <= cfg.inference.batch_size:
            minibatch.append(protein)
        else:
            batches.append(minibatch)
            if protein[1].S.shape[0] <= cfg.inference.batch_size:
                minibatch = [protein]
    if len(minibatch) > 0:
        batches.append(minibatch)

    # Make output dir
    os.makedirs(cfg.inference.output_dir, exist_ok=True)

    # Loop over all desired proteins
    start = time.time()
    for batch in batches:
        
        # Unpack batch
        pdb_names = [protein[0] for protein in batch]
        proteins = [protein[1] for protein in batch]
        
        # Collate the batch
        batch = collate_fn(proteins)
        
        # Run sample
        sample_results = sample_epoch(model, batch, cfg.inference.temperature, device, n_recycle=cfg.inference.n_recycle)

        # Get full atom proteins
        protein_strings = pdbs_from_prediction(sample_results)
        
        for idx, protein_string in enumerate(protein_strings):
            protein_name = pdb_names[idx]
            # Write sampled pdb
            with open(os.path.join(cfg.inference.output_dir, protein_name + '.pdb'), 'w') as f:
                f.write(protein_string)
            
    elapsed = time.time() - start
        

if __name__ == "__main__":
    main()
