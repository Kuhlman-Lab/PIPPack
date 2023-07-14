from typing import *
import numpy as np
import torch
import torch.nn.functional as F
import data.residue_constants as rc
from utils.utils import robust_normalize


FeatureDict = Mapping[str, Union[str, np.ndarray]]
Array = Union[np.ndarray, torch.Tensor]

# TODO: Make all operations function on Arrays
class Featurizer(object):
    def __init__(self) -> None:

        # TODO: Make this dynamic based on config
        # Note: Any feature not included in this dictionary will have a 
        # pad value of 0
        self.default_pad_values = {
            'name': '',
            'length': None,
            'sequence': "X",
            'aatype': rc.restype_num,
            'residue_index': -1,
            'sc_dihedrals_bin': 36,
        }

    def __call__(self, protein: FeatureDict) -> FeatureDict:
        # TODO: Make the computed features dynamic based on the config

        # Adjust protein based on residue_index
        protein = self._adjust_features_for_residue_index(protein)
        
        # Get the desired features
        bb_dihedrals, bb_dihedrals_mask = self.calc_bb_dihedrals(protein['atom_positions'])
        bb_dihedrals_sincos = self._sincos_encoding(bb_dihedrals)
        sc_dihedrals, sc_dihedrals_mask = self.calc_sc_dihedrals(protein['atom_positions'], protein['aatype'])
        sc_dihedrals_b_factor_mask = self._chi_mask_from_b_factors(protein['aatype'], protein['b_factors'])
        sc_dihedrals_sincos = self._sincos_encoding(sc_dihedrals)
        sc_dihedrals_bin, sc_dihedrals_bin_offset = self._dihedral_bin_encoding(sc_dihedrals, n_bins=36, return_offset=True)
        bb_exists = self._bb_exists(protein['atom_positions'])
        
        # Update protein with new features and remove nans
        protein.update({
            'length': np.array([protein['aatype'].shape[0]]),
            'bb_dihedrals': bb_dihedrals,
            'bb_dihedrals_mask': bb_dihedrals_mask,
            'bb_dihedrals_sincos': bb_dihedrals_sincos,
            'sc_dihedrals': sc_dihedrals,
            'sc_dihedrals_mask': sc_dihedrals_mask,
            'sc_dihedrals_b_factor_mask': sc_dihedrals_b_factor_mask,
            'sc_dihedrals_sincos': sc_dihedrals_sincos,
            'sc_dihedrals_bin': sc_dihedrals_bin,
            'sc_dihedrals_bin_offset': sc_dihedrals_bin_offset,
            'bb_exists': bb_exists
        })
        for feat in protein:
            if isinstance(protein[feat], np.ndarray):
                protein[feat] = np.nan_to_num(protein[feat])
        
        return protein
        
    def _adjust_features_for_residue_index(self, protein: FeatureDict) -> FeatureDict:
        
        # Adjust residue_index based on total chain length (including missing residues)
        adj_len = max(protein['residue_index']) - min(protein['residue_index']) + 1
        indexer = protein['residue_index'] - min(protein['residue_index'])
        residue_index = np.arange(adj_len, dtype=np.int32)
        
        # Recreate aatype and sequence with unknown (i.e. 20 or X) at nonexistent residues
        aatype = np.full((adj_len,), fill_value=rc.unk_restype_index, dtype=np.int32)
        aatype[indexer] = protein['aatype']
        sequence = np.full((adj_len,), fill_value="X")
        sequence[indexer] = list(protein['sequence'])
        sequence = ''.join(sequence.tolist())
        
        # Recreate atom_positions with np.nan at nonexistent residues
        atom_positions = np.full((adj_len, *protein['atom_positions'].shape[1:]), fill_value=np.nan, dtype=np.float32)
        atom_positions[indexer] = protein['atom_positions']
        
        # Recreate atom_mask with 0 at nonexistent residues
        atom_mask = np.zeros((adj_len, *protein['atom_mask'].shape[1:]), dtype=np.float32)
        atom_mask[indexer] = protein['atom_mask']
        
        # Update protein 
        protein.update({
            'residue_index': residue_index,
            'aatype': aatype,
            'sequence': sequence,
            'atom_positions': atom_positions,
            'atom_mask': atom_mask
        })
        
        return protein

    @staticmethod
    def _calc_dihedrals(atom_positions: Array, eps=1e-8) -> Array:

        # Unit vectors
        uvecs = robust_normalize(atom_positions[..., 1:, :] - atom_positions[..., :-1, :], eps=eps)
        uvec_2 = uvecs[..., :-2, :]
        uvec_1 = uvecs[..., 1:-1, :]
        uvec_0 = uvecs[..., 2:, :]
        
        # Normals
        if isinstance(atom_positions, np.ndarray):
            nvec_2 = robust_normalize(np.cross(uvec_2, uvec_1, axis=-1), eps=eps)
            nvec_1 = robust_normalize(np.cross(uvec_1, uvec_0, axis=-1), eps=eps)

            # Angle between normals
            cos_dihedral = np.sum(nvec_2 * nvec_1, axis=-1)
            cos_dihedral = np.clip(cos_dihedral, -1 + eps, 1 - eps)
            dihedral = np.sign(np.sum(uvec_2 * nvec_1, axis=-1)) * np.arccos(cos_dihedral)
        else:
            nvec_2 = robust_normalize(torch.cross(uvec_2, uvec_1, dim=-1), eps=eps)
            nvec_1 = robust_normalize(torch.cross(uvec_1, uvec_0, dim=-1), eps=eps)

            # Angle between normals
            cos_dihedral = torch.sum(nvec_2 * nvec_1, dim=-1)
            cos_dihedral = torch.clamp(cos_dihedral, -1 + eps, 1 - eps)
            dihedral = torch.sign(torch.sum(uvec_2 * nvec_1, dim=-1)) * torch.acos(cos_dihedral)
        
        return dihedral
    
    @staticmethod
    def calc_bb_dihedrals(atom_positions: Array, residue_index: Optional[Array] = None, use_pre_omega: bool = True, return_mask: bool = True) -> Union[Array, Tuple[Array, Array]]:
        
        # Get backbone coordinates (and reshape). First 3 coordinates are N, CA, C
        bb_atom_positions = atom_positions[:, :3].reshape((3 * atom_positions.shape[0], 3))

        # Get backbone dihedrals
        bb_dihedrals = Featurizer._calc_dihedrals(bb_atom_positions)
        if isinstance(atom_positions, np.ndarray):
            bb_dihedrals = np.pad(bb_dihedrals, [(1, 2)], constant_values=np.nan) # Add empty phi[0], psi[-1], and omega[-1]
            bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))
            
            # Get mask based on residue_index
            bb_dihedrals_mask = np.ones_like(bb_dihedrals)
            if residue_index is not None:
                assert type(atom_positions) == type(residue_index)
                pre_mask = np.concatenate(([0.0], (residue_index[1:] - 1 == residue_index[:-1]).astype(np.float32)), axis=-1)
                post_mask = np.concatenate(((residue_index[:-1] + 1 == residue_index[1:]).astype(np.float32), [0.0]), axis=-1)
                bb_dihedrals_mask = np.stack((pre_mask, post_mask, post_mask), axis=-1)
            
            if use_pre_omega:
                # Move omegas such that they're "pre-omegas" and reorder dihedrals
                bb_dihedrals[:, 2] = np.concatenate(([np.nan], bb_dihedrals[:-1, 2]), axis=-1)
                bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
                bb_dihedrals_mask[:, 1] = bb_dihedrals_mask[:, 0]
                
            # Update dihedral_mask
            bb_dihedrals_mask = bb_dihedrals_mask * np.isfinite(bb_dihedrals).astype(np.float32)
        else:
            bb_dihedrals = F.pad(bb_dihedrals, [1, 2], value=torch.nan) # Add empty phi[0], psi[-1], and omega[-1]
            bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))
            
            # Get mask based on residue_index
            bb_dihedrals_mask = torch.ones_like(bb_dihedrals)
            if residue_index is not None:
                assert type(atom_positions) == type(residue_index)
                pre_mask = torch.cat((torch.tensor([0.0]), (residue_index[1:] - 1 == residue_index[:-1]).to(torch.float32)), dim=-1)
                post_mask = torch.cat(((residue_index[:-1] + 1 == residue_index[1:]).to(torch.float32), torch.tensor([0.0])), dim=-1)
                bb_dihedrals_mask = torch.stack((pre_mask, post_mask, post_mask), axis=-1)
            
            if use_pre_omega:
                # Move omegas such that they're "pre-omegas" and reorder dihedrals
                bb_dihedrals[:, 2] = torch.cat((torch.tensor([torch.nan]), bb_dihedrals[:-1, 2]), dim=-1)
                bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
                bb_dihedrals_mask[:, 1] = bb_dihedrals_mask[:, 0]
                
            # Update dihedral_mask
            bb_dihedrals_mask = bb_dihedrals_mask * torch.isfinite(bb_dihedrals).to(torch.float32)
        if return_mask:
            return bb_dihedrals, bb_dihedrals_mask
        else:
            return bb_dihedrals
    
    @staticmethod
    def calc_sc_dihedrals(atom_positions: Array, aatype: Array, return_mask: bool = True) -> Union[Array, Tuple[Array, Array]]:

        # Make sure atom_positions and aatype are same class
        assert type(atom_positions) == type(aatype)

        # Get atom indicies for atoms that make up chi angles and chi mask
        if isinstance(atom_positions, np.ndarray):
            chi_atom_indices = np.array(rc.chi_atom_indices_atom14, dtype=np.int32)[aatype]
            chi_mask = np.array(rc.chi_mask_atom14, dtype=np.float32)[aatype]

            # Get coordinates for chi atoms
            chi_atom_positions = np.take_along_axis(atom_positions, chi_atom_indices[..., None].repeat(3, axis=-1), axis=-2)
        else:
            chi_atom_indices = torch.from_numpy(np.array(rc.chi_atom_indices_atom14, dtype=np.int32)).to(aatype.device)[aatype]
            chi_mask = torch.from_numpy(np.array(rc.chi_mask_atom14, dtype=np.float32)).to(aatype.device)[aatype]
            
            # Get coordinates for chi atoms
            chi_atom_positions = torch.gather(atom_positions, -2, chi_atom_indices[..., None].expand(*chi_atom_indices.shape, 3).long())
        sc_dihedrals = Featurizer._calc_dihedrals(chi_atom_positions)
        
        # Turn nonexistent chis into nan
        if isinstance(atom_positions, np.ndarray):
            chi_mask[chi_mask == 0.] = np.nan
            sc_dihedrals = sc_dihedrals * chi_mask
            sc_dihedrals_mask = np.isfinite(sc_dihedrals).astype(np.float32)
        else:
            #chi_mask[chi_mask == 0.] = torch.nan
            sc_dihedrals = sc_dihedrals * chi_mask
            sc_dihedrals_mask = (sc_dihedrals == 0.).to(torch.float32)
        
        if return_mask:    
            return sc_dihedrals, sc_dihedrals_mask
        else:
            return sc_dihedrals
            
    def _sincos_encoding(self, array: np.ndarray) -> np.ndarray:
        return np.stack((np.sin(array), np.cos(array)), axis=-1)
    
    def _bin_encoding(self, array: np.ndarray, a_min: float, a_max: float, n_bins: int, extra_bin_for_nan: bool = True, return_offset: bool = False) -> np.ndarray:
    
        # Determine the bin for each value
        bins = np.arange(a_min, a_max, (a_max - a_min) / n_bins, dtype=np.float32)
        binned_array = np.argmin(np.abs(array[..., None] - bins), axis=-1)
        if extra_bin_for_nan:
            binned_array[np.isnan(array)] = n_bins
        
        if return_offset:
            # Compute offset within bin for each value
            offset_array = np.min(np.abs(array[..., None] - bins), axis=-1)
            offset_array[np.isnan(array)] = 0.0
            return binned_array, offset_array
        else:
            return binned_array
    
    def _dihedral_bin_encoding(self, array: np.ndarray, n_bins: int, return_offset: bool = False) -> np.ndarray:
        return self._bin_encoding(array, a_min=-np.pi, a_max=np.pi, n_bins=n_bins, extra_bin_for_nan=True, return_offset=return_offset)

    def _bb_exists(self, atom_positions: np.ndarray) -> np.ndarray:
        return np.isfinite(np.sum(atom_positions[:, :4], axis=(-1, -2))).astype(np.float32)
    
    @staticmethod
    def chi_mask_from_b_factors(aatype: np.ndarray, b_factors: np.ndarray) -> np.ndarray:
        
        # Load default chi_mask based on sequence alone.
        chi_mask = np.array(rc.chi_mask_atom14, dtype=np.float32)[aatype] # [..., N, 4]
        
        # Get atom14 indices for chi atoms
        chi_atom_indices = np.array(rc.chi_atom_indices_atom14, dtype=np.int32)[aatype]
        chi_atom_indices = np.stack([chi_atom_indices[:, :4], chi_atom_indices[:, 1:5], chi_atom_indices[:, 2:6], chi_atom_indices[:, 3:]], axis=-1)
        
        # Get b_factors at all chi atoms
        chi_atom_b_factors = np.take_along_axis(b_factors[..., None].repeat(4, axis=-1), chi_atom_indices, axis=-2)
        
        # Create mask for chi angles that have at least 1 atom with B-factor > 40
        chi_b_factor_mask = np.sum(chi_atom_b_factors > 40, axis=-1) == 0
        
        # Combine default chi_mask with b_factor mask
        chi_mask = chi_mask * chi_b_factor_mask
        
        return chi_mask.astype(np.float32)


def _calc_dihedrals(atom_positions: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Unit vectors
    uvecs = robust_normalize(atom_positions[..., 1:, :] - atom_positions[..., :-1, :], eps=eps)
    uvec_2 = uvecs[..., :-2, :]
    uvec_1 = uvecs[..., 1:-1, :]
    uvec_0 = uvecs[..., 2:, :]
    
    # Normals
    nvec_2 = robust_normalize(torch.cross(uvec_2, uvec_1, dim=-1), eps=eps)
    nvec_1 = robust_normalize(torch.cross(uvec_1, uvec_0, dim=-1), eps=eps)

    # Angle between normals
    cos_dihedral = torch.sum(nvec_2 * nvec_1, dim=-1)
    cos_dihedral = torch.clamp(cos_dihedral, -1 + eps, 1 - eps)
    #print(torch.any(torch.isnan(cos_dihedral)))
    dihedral = torch.sign(torch.sum(uvec_2 * nvec_1, dim=-1)) * torch.acos(cos_dihedral)
    
    return dihedral


def calc_sc_dihedrals(atom_positions: torch.Tensor, aatype: torch.Tensor, return_mask: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Get atom indicies for atoms that make up chi angles and chi mask
    chi_atom_indices = torch.tensor(rc.chi_atom_indices_atom14).to(aatype.device)[aatype]
    chi_mask = torch.tensor(rc.chi_mask_atom14).to(aatype.device)[aatype]
    
    # Get coordinates for chi atoms
    chi_atom_positions = torch.gather(atom_positions, -2, chi_atom_indices[..., None].expand(*chi_atom_indices.shape, 3).long())
    sc_dihedrals = _calc_dihedrals(chi_atom_positions)
    
    # Chi angles that are missing an atom will be NaN, so turn all those to 0.
    sc_dihedrals = torch.nan_to_num(sc_dihedrals)

    # Mask nonexistent chis based on sequence.
    sc_dihedrals = sc_dihedrals * chi_mask
    sc_dihedrals_mask = (sc_dihedrals != 0.).to(torch.float32)

    if return_mask:    
        return sc_dihedrals, sc_dihedrals_mask
    else:
        return sc_dihedrals


def chi_mask_from_b_factors(aatype: torch.Tensor, b_factors: torch.Tensor, b_factor_cutoff: float = 40) -> torch.Tensor:
    # Load default chi_mask based on sequence alone.
    chi_mask = torch.tensor(rc.chi_mask_atom14)[aatype] # [..., N, 4]
    
    # Get atom14 indices for chi atoms
    chi_atom_indices = torch.tensor(rc.chi_atom_indices_atom14)[aatype]
    chi_atom_indices = torch.stack([chi_atom_indices[..., :4], chi_atom_indices[..., 1:5], chi_atom_indices[..., 2:6], chi_atom_indices[..., 3:]], dim=-1)
    
    # Get b_factors at all chi atoms
    chi_atom_b_factors = torch.gather(b_factors[..., None].expand(*aatype.shape, -1, 4), -2, chi_atom_indices.long())
    
    # Create mask for chi angles that have at least 1 atom with B-factor > b_factor_cutoff
    chi_b_factor_mask = torch.sum(chi_atom_b_factors > b_factor_cutoff, dim=-1) == 0
    
    # Combine default chi_mask with b_factor mask
    chi_mask = chi_mask * chi_b_factor_mask
    
    return chi_mask.float()
