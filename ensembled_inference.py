import os
import logging
import glob
import pickle
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
torch.cuda.empty_cache()
from typing import *
import lightning

# Library code
from utils.train_utils import load_checkpoint
from model.modules import get_atom14_coords
from data.protein import from_pdb_file
from data.top2018_dataset import transform_structure, collate_fn
import data.residue_constants as rc
from inference import replace_protein_sequence, pdbs_from_prediction


logger = logging.getLogger(__name__)


def sample_epoch(ensemble, batch, temperature, device, n_recycle=0):    
    # Sampling epoch
    model_logits = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            # Move to device
            batch = batch.to(device)
        
            # Sample the model
            results = model.sample(batch, temperature=temperature, n_recycle=n_recycle)

            # Get the final logits
            model_logits.append(results['chi_logits'])
            
    # Perform ensemble averaging with temperature sampling
    logits = torch.stack(model_logits, dim=-1)
    logits = torch.mean(logits, dim=-1)
    if temperature > 0.0:
        logits = logits / temperature
        chi_probs = F.softmax(logits, -1)
        chi_bin = torch.multinomial(chi_probs.view(-1, logits.shape[-1]), 1).view(*logits.shape[:2], -1).squeeze(-1)
    else:
        chi_bin = torch.argmax(F.softmax(logits, -1), dim=-1)
    
    chi_bin_one_hot = torch.nn.functional.one_hot(chi_bin, num_classes=model.n_chi_bins + 1)

    # Determine actual chi value from bin
    chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / model.n_chi_bins, device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
    pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
    
    # Add bin offset
    chi_bin_offset = results.get('chi_bin_offset', None)
    if chi_bin_offset is not None:
        bin_sample_update = chi_bin_offset
    else:
        bin_sample_update = (2 * torch.pi / model.n_chi_bins) * torch.rand(chi_bin.shape, device=chi_bin.device)
    chi_pred = pred_chi_bin + bin_sample_update
    
    # Construct final atom14 coordinates
    aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]
    chi_pred = aatype_chi_mask * chi_pred
    atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

    results['final_X'] = atom14_xyz
    results.update(batch.to_dict())
    
    return results

@hydra.main(version_base=None, config_path="./config", config_name="inference_ensemble")
def main(cfg: DictConfig) -> None:
    # Set up RNG and device
    seed = lightning.seed_everything(cfg.inference.seed)
    logger.info(f"Using seed={seed} for RNG.")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not cfg.inference.force_cpu) else "cpu")

    ensemble = []
    for model_name in cfg.inference.model_names:
        # Get the config used when running experiment
        with open(os.path.join(cfg.inference.weights_path, f'{model_name}_config.pickle'), 'rb') as f:
            exp_cfg = pickle.load(f)
            
        # Load model with same config  
        model: torch.nn.Module = hydra.utils.instantiate(exp_cfg.model).to(device)
    
        # Find the best checkpoint to load into model
        checkpoint = os.path.join(cfg.inference.weights_path, f'{model_name}_ckpt.pt')

        # Load the best checkpoint
        load_checkpoint(checkpoint, model)

        # Add model to ensemble
        ensemble.append(model)

    # Get the dataset
    pdb_files = glob.glob(os.path.join(cfg.inference.pdb_path, '*.pdb'))
    if cfg.inference.get("replace_seqs", False):
        assert len(pdb_files) == 1
        
        fasta_files = glob.glob(os.path.join(cfg.inference.pdb_path, '*.fasta'))
        assert len(fasta_files) == 1
        
        with open(fasta_files[0], 'r') as f:
            lines = f.readlines()
        new_seqs = [line.strip().split('/') for line in lines if line[0] != ">" and line]
        
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
    for batch in batches:
        
        # Unpack batch
        pdb_names = [protein[0] for protein in batch]
        proteins = [protein[1] for protein in batch]
        
        # Collate the batch
        batch = collate_fn(proteins)
        
        # Run sample
        sample_results = sample_epoch(ensemble, batch, cfg.inference.temperature, device, n_recycle=cfg.inference.n_recycle)

        # Get full atom proteins
        protein_strings = pdbs_from_prediction(sample_results)
        
        for idx, protein_string in enumerate(protein_strings):
            protein_name = pdb_names[idx]
            pdb_out = os.path.join(cfg.inference.output_dir, protein_name + '.pdb')
            
            # Write sampled pdb
            print('Finished packing:', pdb_out)
            with open(os.path.join(cfg.inference.output_dir, protein_name + '.pdb'), 'w') as f:
                f.write(protein_string)
        

if __name__ == "__main__":
    main()
