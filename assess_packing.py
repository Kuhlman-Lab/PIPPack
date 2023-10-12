# Assess sidechain packing

import pickle
import argparse
import os
from typing import *
import numpy as np
import torch
import torch.nn.functional as F

import data.residue_constants as rc
from data.protein import from_pdb_file, Protein
from data.featurizer import calc_sc_dihedrals, chi_mask_from_b_factors
from model.loss import masked_mean, get_renamed_coords


def get_pdb_targets_from_dir(dir: str, tag: str = '') -> Sequence[str]:
    target_list = [file[:-(4 + len(tag))] for file in os.listdir(dir) 
                   if file[-(4 + len(tag)):] == f"{tag}.pdb"]
    return target_list


def get_alt_CH(CH: torch.Tensor, S: torch.Tensor, pseudo_periodic: bool = False, return_periodic_chi: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Determine which residues have a periodic chi angle.
    if pseudo_periodic:
        periodic_chi = torch.tensor(rc.chi_pseudo_pi_periodic, device=S.device)[S.long()]
    else:
        periodic_chi = torch.tensor(rc.chi_pi_periodic, device=S.device)[S.long()]
    
    # Get alternative chis.
    alt_CH = CH.clone()
    alt_CH += torch.where(CH * periodic_chi > 0.0, -torch.pi * periodic_chi, torch.pi * periodic_chi)
    
    if return_periodic_chi:
        return alt_CH, periodic_chi
    else:
        return alt_CH


def compute_chi_error(native_protein: Protein, decoy_protein: Protein, sc_b_factor_cutoff: float = 1000, compute_pseudo_ae: bool = False, device=torch.device('cpu')) -> Dict[str, torch.Tensor]:
    # Get dihedral angles for native.
    native_X = torch.from_numpy(native_protein.atom_positions).clone().to(device=device)
    native_S = torch.from_numpy(native_protein.aatype).clone().to(device=device)
    native_CH, native_CH_mask = calc_sc_dihedrals(native_X, native_S)
    # Include b_factor mask for native_protein.
    native_BF = torch.from_numpy(native_protein.b_factors).clone().to(device=device)
    native_CH_mask *= chi_mask_from_b_factors(native_S, native_BF, sc_b_factor_cutoff)
    
    # Get dihedral angles and alternatives for decoy.
    decoy_X = torch.from_numpy(decoy_protein.atom_positions).clone().to(device=device)
    decoy_S = torch.from_numpy(decoy_protein.aatype).clone().to(device=device)
    decoy_CH, decoy_CH_mask = calc_sc_dihedrals(decoy_X, decoy_S)
    decoy_alt_CH, periodic_CH = get_alt_CH(decoy_CH, decoy_S, return_periodic_chi=True)
    
    # Compute a rotamer mask. If all chi angles aren't present, don't use that residue for rotamer recovery.
    chi_aatype_mask = torch.tensor(rc.chi_mask_atom14).to(device=device)[native_S]
    chi_present_mask = native_CH_mask * decoy_CH_mask
    rotamer_mask = torch.sum(chi_aatype_mask, -1) == torch.sum(chi_present_mask, -1)
    rotamer_mask[torch.sum(chi_aatype_mask, -1) == 0] = 0.
    
    # Compute angle difference (in radians), accounting for periodicity
    angle_diff = native_CH - decoy_CH
    angle_diff[angle_diff > torch.pi] = angle_diff[angle_diff > torch.pi] - 2 * torch.pi
    angle_diff[angle_diff < -torch.pi] = angle_diff[angle_diff < -torch.pi] + 2 * torch.pi
    
    # Compute alternative angle difference (in radians), accounting for periodicity
    alt_angle_diff = native_CH - decoy_alt_CH
    alt_angle_diff[alt_angle_diff > torch.pi] = alt_angle_diff[alt_angle_diff > torch.pi] - 2 * torch.pi
    alt_angle_diff[alt_angle_diff < -torch.pi] = alt_angle_diff[alt_angle_diff < -torch.pi] + 2 * torch.pi
    
    # Compute absolute error (in radians)
    ae = torch.minimum(torch.abs(angle_diff), torch.abs(alt_angle_diff))
    
    # Construct outputs
    chi_error = {
        "chi_mae": ae,
        "chi_mask": native_CH_mask * decoy_CH_mask,
        "rotamer_mask": rotamer_mask.unsqueeze(-1)
    }
    
    if compute_pseudo_ae:
        # Determine where the alternative chi is better
        alt_better = (ae == torch.abs(alt_angle_diff)) * periodic_CH
        decoy_CH = decoy_CH * (1. - alt_better) + decoy_alt_CH * alt_better
        
        # Compute pseudo pi periodic angle differences (in radians), accounting for periodicity
        decoy_pseudo_alt_CH = get_alt_CH(decoy_CH, decoy_S, pseudo_periodic=True)
        pseudo_alt_angle_diff = native_CH - decoy_pseudo_alt_CH
        pseudo_alt_angle_diff[pseudo_alt_angle_diff > torch.pi] = pseudo_alt_angle_diff[pseudo_alt_angle_diff > torch.pi] - 2 * torch.pi
        pseudo_alt_angle_diff[pseudo_alt_angle_diff < -torch.pi] = pseudo_alt_angle_diff[pseudo_alt_angle_diff < -torch.pi] + 2 * torch.pi
        
        # Compute masked absolute error (in radians).
        ae_pseudo = torch.minimum(ae, torch.abs(pseudo_alt_angle_diff))
        chi_error["chi_mae_pseudo"] = ae_pseudo
        
    return chi_error


def compute_centrality(protein: Protein, basis_atom: str = "CB", threshold: float = 10.0, backup_atom: str = "CA", device=torch.device('cpu')) -> torch.Tensor:
    # Get coordinates of atoms for centrality computation.
    protein_X = torch.from_numpy(protein.atom_positions).clone().to(device=device)
    coords = protein_X[..., rc.atom_order[basis_atom], :]
    coords[~torch.isfinite(torch.sum(coords, dim=-1))] = protein_X[..., rc.atom_order[backup_atom], :][~torch.isfinite(torch.sum(coords, dim=-1))]
    
    # Compute distances and centrality.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * threshold)
    centrality = torch.sum(pairwise_dists < threshold, dim=-1) - 1
    
    return centrality

# TODO: ADD ARGUMENT THAT COMPUTES BASED ON B-FACTOR MASK
# COMPARE ATTNPACKER EXAMPLE TO THIS CODE
# SHOULD RMSD INCLUDE CB RMSD??
def compute_sc_rmsd(native_protein: Protein, decoy_protein: Protein, sc_b_factor_cutoff: float = 1000, compute_pseudo_rmsd: bool = False, per_res: bool = True, device=torch.device('cpu')) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Compute atom deviation based on original coordinates
    native_X = torch.from_numpy(native_protein.atom_positions).clone().to(device=device)
    decoy_X = torch.from_numpy(decoy_protein.atom_positions).clone().to(device=device)
    atom_deviation = torch.sum(torch.square(native_X - decoy_X), dim=-1)
    
    # Compute atom deviation based on alternative coordinates
    decoy_S = torch.from_numpy(decoy_protein.aatype).clone().to(device=device)
    decoy_renamed_X = get_renamed_coords(decoy_X, decoy_S)
    renamed_atom_deviation = torch.sum(torch.square(native_X - decoy_renamed_X), dim=-1)
    
    # Get atom mask, including masked backbone atoms
    atom_mask = torch.from_numpy(native_protein.atom_mask).clone().to(device=device)
    atom_mask *= torch.from_numpy(decoy_protein.atom_mask).to(device=device)
    atom_mask[..., :4] = 0.0  # N, CA, C, O
    # Include b_factors from native_protein.
    native_BF = torch.from_numpy(native_protein.b_factors).clone().to(device=device)
    atom_mask *= (native_BF < sc_b_factor_cutoff)
    
    # Compute RMSD
    dim = -1 if per_res else (-1, -2)
    rmsd_og = torch.sqrt(masked_mean(atom_mask, torch.nan_to_num(atom_deviation), dim))
    rmsd_renamed = torch.sqrt(masked_mean(atom_mask, torch.nan_to_num(renamed_atom_deviation), dim))
    rmsd = torch.minimum(
        rmsd_og,
        rmsd_renamed
    )
    
    # Construct mask of residues with at least one sidechain atom
    sc_mask = (torch.sum(atom_mask, dim=-1) > 0.0).float()
    
    # Construct outputs.
    rmsd_outputs = {
        'sc_rmsd': rmsd,
        "sc_mask": sc_mask,
    }
    
    if compute_pseudo_rmsd:
        # Determine where the renamed atoms are better
        renamed_better = (rmsd == rmsd_renamed)[..., None, None]
        decoy_X = decoy_X * ~renamed_better + decoy_renamed_X * renamed_better
        
        # Compute RMSD based on pseudo renamed alternative coordinates
        decoy_pseudo_renamed_X = get_renamed_coords(decoy_X, decoy_S, pseudo_renaming=True)
        pseudo_renamed_atom_deviation = torch.sum(torch.square(native_X - decoy_pseudo_renamed_X), dim=-1)
        rmsd_pseudo_renamed = torch.sqrt(masked_mean(atom_mask, torch.nan_to_num(pseudo_renamed_atom_deviation), dim))
        rmsd_pseudo = torch.minimum(
            rmsd,
            rmsd_pseudo_renamed
        )
        rmsd_outputs["sc_rmsd_pseudo"] = rmsd_pseudo
        
    return rmsd_outputs


def compute_pairwise_vdw_table_and_masks(
    sequence: torch.Tensor,
    atom_mask: torch.Tensor,
):
    
    # Van der waals radii sum for each pair of atoms
    atomtype_radius = torch.tensor(rc.restype_atom_radius_atom14, dtype=torch.float32, device=sequence.device)[sequence]
    atomtype_radius = atomtype_radius[atom_mask.bool()]
    atom_pair_vdw_sum = atomtype_radius[:, None] + atomtype_radius[None, :]
    
    # Batch index
    if len(sequence.shape) == 2:
        batch_index = torch.arange(sequence.shape[0], dtype=torch.long, device=sequence.device)
        batch_index = batch_index.view(-1, 1, 1).expand(*atom_mask.shape)
        batch_index = batch_index[atom_mask.bool()]
    
    # Residue index
    residue_index = torch.arange(
        sequence.shape[-1], dtype=torch.long, device=sequence.device)
    residue_index = residue_index.expand((*sequence.shape[:-1], -1))[..., None]
    residue_index = residue_index.repeat(*((1,) * len(residue_index.shape[:-1])), atom_mask.shape[-1])
    residue_index = residue_index[atom_mask.bool()]
    
    # Atom index
    # atom_index = torch.arange(
    #     atom_mask.shape[-1], dtype=torch.long, device=atom_mask.device)
    # atom_index = atom_index.expand((*sequence.shape, -1))
    # atom_index = atom_index[atom_mask.bool()]
    
    # Mask out all the duplicate entries in the lower triangular matrix.
    vdw_mask = (
        residue_index[:, None]
        < residue_index[None, :]
    )
    
    # Mask atom pairs from different proteins
    if len(sequence.shape) == 2:
        same_protein_mask = batch_index[:, None] == batch_index[None, :]
        vdw_mask = vdw_mask * same_protein_mask.bool()
    
    # Disulfide bridge between two cysteines is no clash.
    sg_mask = torch.zeros_like(atom_mask).float()
    sg_mask[sequence == rc.restype_order["C"], rc.restype_name_to_atom14_names["CYS"].index("SG")] = 1.0
    sg_mask = sg_mask[atom_mask.bool()]
    disulfide_bonds = sg_mask[:, None] * sg_mask[None, :]
    vdw_mask = vdw_mask * ~disulfide_bonds.bool()

    # Backbone-backbone mask due to fixed backbone
    bb_mask = torch.zeros_like(atom_mask).float()
    bb_mask[..., :4] = 1.0
    bb_mask = bb_mask[atom_mask.bool()]
    bb_pair_mask = bb_mask[:, None] * bb_mask[None, :]
    vdw_mask = vdw_mask * ~bb_pair_mask.bool()
    
    # Proline CD is bonded with BB nitrogen - clashes should be ignored 
    # between this and neighboring residues
    is_proline = sequence == rc.restype_order["P"]
    before_or_after_proline = is_proline.clone().float()
    before_or_after_proline[..., 1:] += is_proline[..., :-1]
    before_or_after_proline[..., :-1] += is_proline[..., 1:]
    before_or_after_proline = before_or_after_proline.unsqueeze(-1)
    pro_mask = torch.zeros_like(atom_mask).float()
    CD_mask, bb_mask = torch.zeros_like(atom_mask), torch.zeros_like(atom_mask)
    CD_mask[sequence == rc.restype_order["P"], rc.restype_name_to_atom14_names["PRO"].index("CD")] = 1.0
    bb_mask[..., :4] = 1.0
    pro_mask[(before_or_after_proline * bb_mask).bool()] = 1.0
    pro_mask[CD_mask.bool()] = 1.0
    pro_mask = pro_mask[atom_mask.bool()][:, None] * pro_mask[atom_mask.bool()][None, :]
    vdw_mask = vdw_mask * ~pro_mask.bool()
    
    # Bonded atoms within each residue are no clash
    # restype_bonds = torch.from_numpy(rc.restype_bonded_atoms(True, False))
    # restype_bonds = restype_bonds.to(dtype=torch.float32, device=sequence.device)
    # restype_bonds = restype_bonds[sequence] # [B, N, 37, 37]
    
    # same_res_mask = residue_index[:, None] == residue_index[None, :]
    # if len(sequence.shape) == 2:
    #     same_res_mask = same_res_mask * same_protein_mask.bool()
        
    # same_res_bonded_mask = torch.zeros_like(same_res_mask)
    # for i, j in zip(*torch.where(same_res_mask)):
    #     res_idx = residue_index[i]
    #     atom_idx_i, atom_idx_j = atom_index[i], atom_index[j]
    #     if len(sequence.shape) == 2:
    #         bonded = restype_bonds[batch_index[i], res_idx, atom_idx_i, atom_idx_j]
    #     else:
    #         bonded = restype_bonds[res_idx, atom_idx_i, atom_idx_j]
    #     same_res_bonded_mask[i, j] = bonded
        
    # vdw_mask = vdw_mask * ~same_res_bonded_mask.bool()

    # Correct VDW tolerance for H-Bonds
    hbond_donors = torch.tensor(rc.restype_hbond_donors_atom14, device=sequence.device)[sequence][atom_mask.bool()]
    hbond_acceptors = torch.tensor(rc.restype_hbond_acceptors_atom14, device=sequence.device)[sequence][atom_mask.bool()]
    hbond_mask = (hbond_donors[:, None] * hbond_acceptors[None, :]).bool()
    hbond_mask = torch.logical_or(hbond_mask, hbond_mask.T)
    
    return atom_pair_vdw_sum, vdw_mask, hbond_mask


def compute_clashes(
        protein: Protein,
        global_allowance: float = 0.0, 
        global_tol_frac: Union[float, Sequence[float]] = 1.0,
        hbond_allowance: float = 0.6,
        eps: float = 1e-8,
        device=torch.device('cpu')
    ):
    """
    Special Cases:
        - only atoms separated by at least 4 bonds are considered
        - Backbone-Backbone atom interactions are ignored
        - potential disulfide bridges SG-SG in cysteines are ignored
        - An allowance is subtracted from all atom pairs where one atom is an H-bond donor
        and the other atom is an H-bond acceptor

    Parameters:
        sequence: encoding of sequence (according to pc.AA_IDX_MAP)
        atom_mask: (according to pc.ALL_ATOM_POSITIONS)
        global_allowance: reduce the sum of vdw radii by this amount
        hbond_allowance: reduce the sum of of vdw radii by this amount, when one of the atoms is
        a hydrogen bond donor, and the other is a hydrogen bond acceptor.
        (See e.g. : https://pubmed.ncbi.nlm.nih.gov/9672047/ )

    Returns:
        (1) Table of the form 
            T[i,j] = rVDW[i] + rVDW[j] – allowance[i,j]
        (2) Mask of the form:
            M[i,j] = True if and only if steric overlap should be computed for pair (i,j)

    """

    atom_positions = torch.from_numpy(protein.atom_positions).clone().to(device=device)
    sequence = torch.from_numpy(protein.aatype).clone().to(device=device)
    atom_mask = torch.from_numpy(protein.atom_mask).clone().to(device=device)

    # Get Van der waals pair table and mask
    vdw_table, vdw_mask, hbond_mask = compute_pairwise_vdw_table_and_masks(sequence, atom_mask)

    # Get list of tolerance fractions
    tol_fracs = global_tol_frac if isinstance(global_tol_frac, list) else [global_tol_frac]

    # Compute clashes for each tolerance level
    clash_info = {}
    for tol_frac in tol_fracs:
        vdw_threshold = (vdw_table.clone() - global_allowance) * tol_frac
        vdw_threshold[hbond_mask] -= hbond_allowance
    
        # Perform actual distance calculation and determine clashes
        dists = torch.sqrt(
            eps + torch.sum(
                (atom_positions[atom_mask.bool()][:, None]
                - atom_positions[atom_mask.bool()][None, :])
                ** 2,
                dim=-1))
        atom_pair_vdw_loss = F.relu(vdw_threshold - dists)
        atom_pair_vdw_loss[~vdw_mask] = 0.0

        tol_dict = dict(
            loss_avg = masked_mean(vdw_mask, atom_pair_vdw_loss, (-1, -2)),
            num_clashes = torch.sum(atom_pair_vdw_loss > 0.0),
        )
        
        clash_info[str(tol_frac)] = tol_dict

    return clash_info


def assess_sidechains(native_pdb_path: str, decoy_pdb_path: str, sc_b_factor_cutoff: float = 1000, clash_tolerances: Sequence[float] = [0.8, 0.9, 1.0], hbond_allowance: float = 0.6, convert_mse: bool = False, device=torch.device('cpu')):
    # Load native protein
    native_protein = from_pdb_file(native_pdb_path, mse_to_met=convert_mse)
    native_seq = "".join([rc.restypes_with_x[idx] for idx in native_protein.aatype])
    
    # Load decoy protein
    if not os.path.exists(decoy_pdb_path):
        return None
    decoy_protein = from_pdb_file(decoy_pdb_path, mse_to_met=convert_mse)
    decoy_seq = "".join([rc.restypes_with_x[idx] for idx in decoy_protein.aatype])
    
    if native_seq != decoy_seq:
        print(os.path.basename(native_pdb_path))
        print(native_seq)
        print(decoy_seq)
        assert native_seq == decoy_seq
    
    # Determine chi absolute errors
    chi_error = compute_chi_error(native_protein, decoy_protein, sc_b_factor_cutoff, device=device)
    
    # Determine centrality by number of CBs within 10A. If CB doesn't exist, use CA.
    centrality = compute_centrality(native_protein, device=device)

    # Determine sidechain RMSDs
    rmsd = compute_sc_rmsd(native_protein, decoy_protein, sc_b_factor_cutoff, device=device)

    # Determine clashes as different tolerance values
    clash_info = compute_clashes(
        decoy_protein,
        global_tol_frac=clash_tolerances,
        hbond_allowance=hbond_allowance, 
        device=device)

    return {
        'chi_error': chi_error,
        'centrality': centrality,
        'rmsd': rmsd,
        'clash_info': clash_info,
        'seq': torch.from_numpy(native_protein.aatype).to(device=device),
    }
    

def summarize(stats, per_aatype: bool = False):
    # WARNING: Clash info will be across all residues
    
    # Accumulate across targets
    total_stats = {
        "chi_error":
            {key: torch.cat([stats[target]["chi_error"][key] for target in stats], dim=-2)
             for key in stats[list(stats.keys())[0]]["chi_error"]},
        "centrality": torch.cat([stats[target]['centrality'] for target in stats], dim=-1),
        "rmsd":
            {key: torch.cat([stats[target]["rmsd"][key] for target in stats], dim=-1)
             for key in stats[list(stats.keys())[0]]["rmsd"]},
        'clash_info': {
            tol: {
                'num_clashes': np.mean([stats[target]['clash_info'][tol]['num_clashes'].cpu() for target in stats]).item(),
                'loss_avg': np.mean([stats[target]['clash_info'][tol]['loss_avg'].cpu() for target in stats]).item()}
            for tol in stats[list(stats.keys())[0]]['clash_info']},
        "seq": torch.cat([stats[target]["seq"] for target in stats], dim=-1)
    }
    
    aatypes = [res for res in rc.chi_angles_atoms if rc.chi_angles_atoms[res] != []]
    if not per_aatype:
        aatypes = ["all"]
    
    # Initialize summary dictionary.
    summary_dict = {res: {} for res in aatypes}
    summary_dict["clash_info"] = total_stats["clash_info"]
    
    # Loop over all aatypes and centrality levels and compute stats.
    for aatype in aatypes:
        # Get aatype mask.
        aatype_mask = torch.ones_like(total_stats['centrality']).bool()
        if aatype != "all":
            aatype_mask = total_stats["seq"] == rc.restype_order[rc.restype_3to1[aatype]]
        
        for centrality in ["all", "core", "surface"]:
            # Get centrality mask.
            centrality_mask = torch.ones_like(total_stats['centrality']).bool()
            if centrality == "core":
                centrality_mask = total_stats["centrality"] >= 20
            elif centrality == "surface":
                centrality_mask = total_stats["centrality"] <= 15
            
            # Apply aatype mask.
            centrality_mask = centrality_mask * aatype_mask
            
            # Chi MAE (in degrees)
            chi_mask = total_stats["chi_error"]["chi_mask"].clone()[centrality_mask]
            chi_ae_deg = total_stats["chi_error"]["chi_mae"].clone()[centrality_mask] * (180.0 / torch.pi)
            chi_mae = masked_mean(chi_mask, chi_ae_deg, dim=list(range(chi_mask.dim() - 1)))
            
            # Rotamer Recovery
            has_rotamer = total_stats["chi_error"]["rotamer_mask"].clone()[centrality_mask].squeeze(-1)
            all_chi_lt_20 = torch.sum(chi_ae_deg * chi_mask < 20.0, dim=-1) == 4
            mean_rr = masked_mean(has_rotamer, all_chi_lt_20)

            # RMSD
            mean_rmsd = masked_mean(total_stats["rmsd"]["sc_mask"].clone()[centrality_mask], total_stats["rmsd"]["sc_rmsd"].clone()[centrality_mask])

            # Construct centrality dict.
            centrality_dict = {
                "chi_mae": chi_mae.cpu().numpy(),
                "mean_rr": mean_rr.item(),
                "mean_rmsd": mean_rmsd.item(),
                "num_residues": len(has_rotamer),
                "num_rotamers": int(torch.sum(has_rotamer)),
                "num_sc": int(torch.sum(total_stats["rmsd"]["sc_mask"].clone()[centrality_mask]).item()),
                "num_chi": torch.sum(chi_mask, dim=list(range(chi_mask.dim() - 1))).cpu().numpy()
            }
            summary_dict[aatype][centrality] = centrality_dict
    
    if not per_aatype:
        summary_dict.update(summary_dict.pop("all"))
        
    return summary_dict


def main(native_dir: str, decoy_dir: str, decoy_tag: str = '', out_filename: str = "packing_stats", sc_b_factor_cutoff: float = 1000, clash_tolerances: str = "0.8,0.9,1.0", hbond_allowance: float = 0.6, convert_mse: bool = False, per_aatype: bool = False, truncate: int = -1, verbose: bool = False):
    # If have gpu, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get list of pdb targets from the native_dir
    target_list = get_pdb_targets_from_dir(native_dir)
    if len(target_list) == 0:
        raise ValueError(f"No PDBs found in native directory: {native_dir}")
    if truncate > 0:
        target_list = target_list[:truncate]
        
    # Reformat clash tolerances.
    clash_tolerances = [float(tol) for tol in clash_tolerances.split(",")]
    
    # Assess the sidechains for all targets.
    stats = {}
    for target in target_list:
        native_pdb = os.path.join(native_dir, target + '.pdb')
        decoy_pdb = os.path.join(decoy_dir, target + decoy_tag + '.pdb')

        target_stats = assess_sidechains(native_pdb, decoy_pdb, sc_b_factor_cutoff, clash_tolerances, hbond_allowance, convert_mse, device=device)
        if target_stats is None:
            continue
        
        stats[target] = target_stats
        if verbose:
            print(f"Finished assessing {target}.")
        
    # Accumulate stats and summarize
    stats_summary = summarize(stats, per_aatype=per_aatype)
    
    # Save summary stats to pickle file
    out_filename = out_filename + f"_aatype" if per_aatype else out_filename
    with open(os.path.join(decoy_dir, f'{out_filename}.pkl'), 'wb') as f:
        pickle.dump(stats_summary, f)
    
    # Write summary stats to text file
    with open(os.path.join(decoy_dir, f'{out_filename}.txt'), 'w') as f:
        for k, v in stats_summary.items():
            f.write(f"{k}\n")
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    f.write(f"\t{k2}:\n")
                    for k3, v3 in v2.items():
                        f.write(f"\t\t{k3}: {v3}\n")
                else:
                    f.write(f"\t{k2}: {v2}\n")

    return stats_summary


if __name__ == "__main__":
    # Construct argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('native_dir', type=str)
    parser.add_argument('decoy_dir', type=str)
    parser.add_argument('--decoy_tag', type=str, default='')
    parser.add_argument('--out_filename', type=str, default='packing_stats')
    parser.add_argument('--sc_b_factor_cutoff', type=float, default=1000)
    parser.add_argument('--clash_tolerances', type=str, default="0.8,0.9,1.0")
    parser.add_argument('--hbond_allowance', type=float, default=0.6)
    parser.add_argument('--convert_mse', action='store_true')
    parser.add_argument('--per_aatype', action='store_true')
    parser.add_argument('--truncate', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    # Run assessment.
    main(**vars(args))
