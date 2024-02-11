import os
import glob, copy
import pickle
import numpy as np
import torch
import hydra
from omegaconf import DictConfig,OmegaConf

from typing import *
import lightning
import torch.nn.functional as F

# Library code
from pippack.utils.train_utils import load_checkpoint

from pippack.model.modules import get_atom14_coords
from pippack.data.protein import to_pdb, Protein, from_pdb_file
from pippack.data.features import make_atom14_masks, atom14_to_atom37
from pippack.data.top2018_dataset import transform_structure, collate_fn
import pippack.data.residue_constants as rc
from pippack.model.resampling import resample_loop




def replace_protein_sequence(protein, protein_name, new_seqs):
    proteins = []
    
    for i, seqs in enumerate(new_seqs):
        # Verify that the lengths match.
        if len(np.unique(protein['chain_index'])) != len(seqs):
            raise ValueError(f"Length of number of chains in the new sequence ({len(seqs)}) does not match the number of chains in the protein ({len(np.unique(protein['chain_index']))}).")
        else:
            for chain in np.unique(protein['chain_index']):
                if len(seqs[chain]) != len(np.where(protein['chain_index'] == chain)[0]):
                    raise ValueError(f"Length of sequence for chain {chain} ({len(seqs[chain])}) does not match the number of residues in chain {chain} ({len(np.where(protein['chain_index'] == chain)[0])}).")
        
        # Replace protein sequence info
        new_protein = copy.deepcopy(protein)
        for j, seq in enumerate(seqs):
            aatype = np.array([rc.restype_order[res] for res in seq]).astype(np.int64)
            new_protein['aatype'][np.where(new_protein['chain_index'] == j)[0]] = aatype
            
        # Rebuild atom mask
        atom_mask = []
        for res in ''.join(seqs):
            res_atoms = rc.restype_name_to_atom14_names[rc.restype_1to3[res]]
            res_mask = [1 if atom != "" else 0 for atom in res_atoms]
            atom_mask.append(res_mask)
        new_protein["atom_mask"] = np.array(atom_mask).astype(np.float32)
        
        proteins.append((protein_name + f"_{i}", new_protein))
        
    return proteins

@torch.no_grad()
def sample_epoch(model: Union[torch.nn.Module, list[torch.nn.Module]], batch, temperature, device, n_recycle=0, resample=False, resample_args={}):    
    if isinstance(model, list):
        # Sampling epoch
        model_logits = []
        for _model in model:
            _model.eval()
            
            # Move to device
            batch = batch.to(device)
        
            # Sample the model
            results = _model.sample(batch, temperature=temperature, n_recycle=n_recycle)

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
        
        chi_bin_one_hot = torch.nn.functional.one_hot(chi_bin, num_classes=_model.n_chi_bins + 1)

        # Determine actual chi value from bin
        chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / _model.n_chi_bins, device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
        pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
        
        # Add bin offset
        chi_bin_offset = results.get('chi_bin_offset', None)
        if chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / _model.n_chi_bins) * torch.rand(chi_bin.shape, device=chi_bin.device)
        chi_pred = pred_chi_bin + bin_sample_update
        
        # Construct final atom14 coordinates
        aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]
        chi_pred = aatype_chi_mask * chi_pred
        atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)

        results['final_X'] = atom14_xyz
        results.update(batch.to_dict())

    elif isinstance(model,torch.nn.Module):
        # Sampling epoch
        model.eval()
        
        # Move to device
        batch = batch.to(device)
        
        # Sample the model
        results = model.sample(batch, temperature=temperature, n_recycle=n_recycle)

        # Add batch information
        results.update(batch.to_dict())
    else:
        raise TypeError(f'model must be a list[torch.nn.Module] or torch.nn.Module!')
    
    
    if resample:
        for i in range(batch.S.shape[0]):
            # Get the protein components.
            protein = {
                "S": results["S"][i],
                "X": results["X"][i],
                "X_mask": results["X_mask"][i],
                "BB_D": results["BB_D"][i],
                "residue_index": results["residue_index"][i],
                "residue_mask": results["residue_mask"][i],
                "chi_logits": results["chi_logits"][i],
                "chi_bin_offset": results["chi_bin_offset"][i] if "chi_bin_offset" in results else None,
            }
            pred_xyz = results["final_X"][i]
            
            # Perform resampling
            resample_xyz, _ = resample_loop(protein, pred_xyz, **resample_args)
            
            # Update the coordinates
            results["final_X"][i] = resample_xyz
        
    return results


def pdbs_from_prediction(sample_results) -> Sequence[str]:

    # Get the protein components.
    S = sample_results["S"]
    residue_index = sample_results["residue_index"]
    chain_index = sample_results["chain_index"]
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
        chain_idx = chain_index[i].cpu().numpy()
        residue_idx = residue_index[i].cpu().numpy()
        b_factors = np.zeros(atom_mask.shape)
        
        # Update residue_index based on chain_index
        if len(np.unique(chain_idx)) > 1:
            adjustment = 0
            for idx in np.unique(chain_idx)[:-1]:
                adjustment += max(residue_idx[chain_idx == idx])
                adjustment += 100
                residue_idx[chain_idx == idx + 1] -= adjustment

        protein = Protein(aatype=aatype, atom_positions=atom_positions, atom_mask=atom_mask, residue_index=residue_idx, 
                        chain_index=chain_idx, b_factors=b_factors)

        protein_string = to_pdb(protein)
        proteins.append(protein_string)
    
    return proteins


class PIPPack:
    def __init__(self, model:str='pippack_model_1'):
        ALLOWED_MODEL_NAME = ['pippack_model_1', 'pippack_model_2', 'pippack_model_3']
        if model=='ensemble':
            model=ALLOWED_MODEL_NAME
        else:
            assert model in ALLOWED_MODEL_NAME
        self.model_names=model
        self.use_ensemble=isinstance(model,list)

        # loaded models
        self.loaded_models: Union[torch.nn.Module, list[torch.nn.Module]]=[]

        self.weights_path: str=None

        self.seed:Union[int,None]=42
        self.force_cpu:bool=True
        self.device: Union[str,torch.device]=torch.device('cpu')
        self.replace_seqs:bool=False
        self.batch_size=10000
        self.temperature:float=0.0
        self.n_recycle:int=3
        self.use_resample:bool=False
        self.resample_args:DictConfig=None
        
        self.exp_cfg=None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    
    def _download_weights(self):
        
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path,exist_ok=True)
        if not glob.glob(os.path.join(self.weights_path, '*.pt')):
            print('Fetching pretrained weights ...')
            from pippack.utils.utils import fetch_and_unzip_weight
            fetch_and_unzip_weight(self.weights_path)

    
    def _initialize_with_a_model(self,):
    
        self._download_weights()

        # Get the config used when running experiment
        with open(os.path.join(self.weights_path, f'{self.model_names}_config.pickle'), 'rb') as f:
            self.exp_cfg: DictConfig = pickle.load(f)
            #print(OmegaConf.to_yaml(self.exp_cfg))
            #print(type(self.exp_cfg))
            # patch the DictConf after pip installable changes
            self.exp_cfg.model._target_='pippack.model.modules.PIPPackFineTune'

            
        # Set up RNG and device
        seed = lightning.seed_everything(self.seed)
        print(f"Using seed={seed} for RNG.")

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.force_cpu) else "cpu")
        print(f'will run on {self.device}')
        
        # Load model with same config  
        self.loaded_models: torch.nn.Module = hydra.utils.instantiate(self.exp_cfg.model).to(self.device)
        
        # Find the checkpoint to load into model
        checkpoint = os.path.join(self.weights_path, f'{self.model_names}_ckpt.pt')

        # Load the best checkpoint
        load_checkpoint(checkpoint, self.loaded_models)
        print(f'PIPPack intialized with model {self.model_names}')
    
    def _initialize_with_ensemble(self):
        
        self._download_weights()
        
        # Set up RNG and device
        seed = lightning.seed_everything(self.seed)
        print(f"Using seed={seed} for RNG.")
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.force_cpu) else "cpu")

        self.loaded_models = []
        for model_name in self.model_names:
            # Get the config used when running experiment
            with open(os.path.join(self.weights_path, f'{model_name}_config.pickle'), 'rb') as f:
                self.exp_cfg = pickle.load(f)
                # print(OmegaConf.to_yaml(self.exp_cfg))
                # print(type(self.exp_cfg))

                # patch the DictConf after pip installable changes
                self.exp_cfg.model._target_='pippack.model.modules.PIPPackFineTune'

            # Load model with same config  
            model: torch.nn.Module = hydra.utils.instantiate(self.exp_cfg.model).to(self.device)
        
            # Find the best checkpoint to load into model
            checkpoint = os.path.join(self.weights_path, f'{model_name}_ckpt.pt')

            # Load the best checkpoint
            load_checkpoint(checkpoint, model)

            # Add model to ensemble
            self.loaded_models.append(model)
            print(f'Model loaded: {model_name}.')
        
        print(f'PIPPack intialized with ensemble model {self.model_names}')

    def _sorted_minibatches(self,proteins: list) -> list:
        # Form batches
        sorted_proteins = sorted(proteins, key=lambda x: x[1].S.shape[0])

        # Cluster into minibatches of similar sizes
        batches, minibatch = [], []
        for protein in sorted_proteins:
            if protein[1].S.shape[0] * (len(minibatch) + 1) <= self.batch_size:
                minibatch.append(protein)
            else:
                batches.append(minibatch)
                if protein[1].S.shape[0] <= self.batch_size:
                    minibatch = [protein]
        if len(minibatch) > 0:
            batches.append(minibatch)
        return batches
    
    def _process_a_batch(self,batch: list, output_dir: str):
        # Unpack batch
        pdb_names = [protein[0] for protein in batch]
        proteins = [protein[1] for protein in batch]
        
        # Collate the batch
        batch = collate_fn(proteins)
        
        # Run sample
        sample_results = sample_epoch(self.loaded_models, batch, self.temperature, self.device, n_recycle=self.n_recycle, resample=self.use_resample, resample_args=self.resample_args)

        # Get full atom proteins
        protein_strings = pdbs_from_prediction(sample_results)
        
        for idx, protein_string in enumerate(protein_strings):
            protein_name = pdb_names[idx]
            pdb_out = os.path.join(output_dir, protein_name + '.relaxed.pdb')
            
            # Write sampled pdb
            print('Finished packing:', pdb_out)
            with open(pdb_out, 'w') as f:
                f.write(protein_string)
    
    def _run_repack_batch(self,pdb_path,output_dir, mutant_sequence:Union[str, None,Sequence[str]]=None):
        pdb_files = glob.glob(os.path.join(pdb_path, '*.pdb'))
        # Get the dataset
        if mutant_sequence:
            assert len(pdb_files) == 1
            pdb_file=pdb_files[0]
            proteins = replace_protein_sequence(vars(from_pdb_file(pdb_file, mse_to_met=True)), os.path.basename(pdb_file)[:-4], mutant_sequence)
        else: 
            proteins = [(os.path.basename(pdb_file)[:-4], vars(from_pdb_file(pdb_file, mse_to_met=True)))  for pdb_file in pdb_files]
        
        # Transform proteins
        proteins = [(protein[0], transform_structure(protein[1], self.exp_cfg.model.n_chi_bins, sc_d_mask_from_seq=True)) for protein in proteins]
        
        print(f"Running repack with {len(proteins)} tasks.")
        batches=self._sorted_minibatches(proteins=proteins)
        print(f"Sorted into {len(batches)} batches.")

        # Make output dir
        os.makedirs(output_dir, exist_ok=True)

        # Loop over all desired proteins
        for batch in batches:
            self._process_a_batch(batch=batch,output_dir=output_dir)
            


    def _run_repack_single(self,pdb_file,output_file, mutant_sequence:Union[str, None,Sequence[str]]=None):
        if mutant_sequence:
            proteins = replace_protein_sequence(vars(from_pdb_file(pdb_file, mse_to_met=True)), os.path.basename(pdb_file)[:-4], mutant_sequence)
        else: 
            proteins = [(os.path.basename(pdb_file)[:-4], vars(from_pdb_file(pdb_file, mse_to_met=True)))]
        

        # Transform proteins
        proteins = [(protein[0], transform_structure(protein[1], self.exp_cfg.model.n_chi_bins, sc_d_mask_from_seq=True)) for protein in proteins]
        
        print(f"Running repack with {len(proteins)} tasks.")

        # Unpack batch
        pdb_names = [protein[0] for protein in proteins]
        proteins = [protein[1] for protein in proteins]
        
        # Collate the batch
        batch = collate_fn(proteins)
        
        # Run sample
        sample_results = sample_epoch(self.loaded_models, batch, self.temperature, self.device, n_recycle=self.n_recycle, resample=self.use_resample, resample_args=self.resample_args)

        # Get full atom proteins
        protein_strings = pdbs_from_prediction(sample_results)
        
        for idx, protein_string in enumerate(protein_strings):
            protein_name = pdb_names[idx]
            print(f'writing {protein_name}')
            
            # Write sampled pdb
            print('Finished packing:', output_file)
            with open(output_file, 'w') as f:
                f.write(protein_string)
        


        