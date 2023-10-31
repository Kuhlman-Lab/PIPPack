from typing import Dict
import torch
import torch.nn.functional as F

from data import residue_constants as rc
from data.featurizer import Featurizer
from model.modules import get_atom14_coords


def local_interresidue_sc_clash_loss(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clash_overlap_tolerance: float, # OpenFold value is 1.5
    distance_threshold: float = 14.0,
    basis_atom: str = "CB",
    eps: float = 1e-10
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations resulting from sidechains.
    
    Note: This ignores intra-residue clashes and backbone-backbone clashes.
    """
    
    # Get needed components from batch.
    aatype = batch["S"].squeeze().clone()
    restype_atom14_to_atom37 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37, 
        dtype=torch.long, 
        device=aatype.device
    )
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    atom14_atom_exists = batch["X_mask"].squeeze().clone()
    residue_index = batch["residue_index"].squeeze().clone().long()
    residue_mask = batch["residue_mask"].squeeze().clone()
    atom14_pred_positions = atom14_pred_positions.squeeze().clone()
    
    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        rc.van_der_waals_radius[name[0]]
        for name in rc.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        atom14_atom_exists
        * atomtype_radius[residx_atom14_to_atom37]
    )
    
    # Get the basis atom xyz for each residue.
    # shape (N, 3)
    if basis_atom == "CB":
        basis_atom_idx = 4 * torch.ones_like(aatype)
        basis_atom_idx[aatype == rc.restype_order["G"]] = 1
    else:
        basis_atom_idx = rc.atom_order[basis_atom] * torch.ones_like(aatype)
    basis_xyz = torch.gather(atom14_pred_positions, 1, basis_atom_idx[..., None, None].expand(*atom14_pred_positions.shape))[:, 0, :]

    # Determine distances based on basis atoms.
    # shape (N, N)
    basis_dists = torch.sqrt(
        eps
        + torch.sum(
            (basis_xyz[None, :, :] - basis_xyz[:, None, :]) ** 2, dim=-1
        )
    )
    
    # Create the mask for valid residue pairs.
    # shape (N, N)
    fp_type = atom14_pred_positions.dtype
    dists_mask = (
        residue_mask[:, None]
        * residue_mask[None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (same residue pairs)
    dists_mask = dists_mask * (
        residue_index[:, None]
        < residue_index[None, :]
    )
    
    # Determine which residue pairs are within the distance threshold.
    # shape (N, N)
    dists_lower_bound = distance_threshold * torch.ones_like(dists_mask)
    dists_mask = dists_mask * (basis_dists < dists_lower_bound)
    valid_pairs = torch.where(dists_mask)
    
    # Get the atom14 coordinates for the valid residue pairs.
    # shape (N_pairs, 14, 3)
    res1_atom14_xyz = atom14_pred_positions.squeeze().clone()[valid_pairs[0]]
    res2_atom14_xyz = atom14_pred_positions.squeeze().clone()[valid_pairs[1]]
    
    # Get the atomic distances for the valid residue pairs.
    # shape (N_pairs, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (res1_atom14_xyz[..., None, :] - res2_atom14_xyz[..., None, :, :]) ** 2, 
            dim=-1
        )
    )
    
    # Initialize the mask for the allowed distances.
    # shape (N_pairs, 14, 14)
    dists_mask = torch.ones_like(dists)

    # Backbone-backbone clashes are ignored. CB is included in the backbone.
    bb_bb_mask = torch.zeros_like(dists_mask)
    bb_bb_mask[..., :5, :5] = 1.0
    dists_mask = dists_mask * (1.0 - bb_bb_mask)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = aatype.new_tensor(cys_sg_idx)
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14)
    cys_res1 = aatype[valid_pairs[0]] == rc.restype_order["C"]
    cys_res2 = aatype[valid_pairs[1]] == rc.restype_order["C"]
    cys_mask = torch.logical_and(cys_res1, cys_res2)
    disulfide_bonds = cys_mask[..., None, None] * (
        cys_sg_one_hot[None, :, None]
        * cys_sg_one_hot[None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)
    
    # Mask interactions between side chain and backbone when atoms are separated by less than 4 bonds.
    # For all residues, ignore Cb_i - N_i+1 and C_i - Cb_i+1.
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14).type(fp_type)
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14).type(fp_type)
    cb_one_hot = F.one_hot(residue_index.new_tensor(4), num_classes=14).type(fp_type)
    neighbor_mask = (residue_index[valid_pairs[0]] + 1) == residue_index[valid_pairs[1]]
    cb_n_dists = neighbor_mask[..., None, None] * cb_one_hot[None, :, None] * n_one_hot[None, None, :]
    c_cb_dists = neighbor_mask[..., None, None] * c_one_hot[None, :, None] * cb_one_hot[None, None, :]
    dists_mask = dists_mask * (1.0 - cb_n_dists) * (1.0 - c_cb_dists)
    
    # For PRO at i+1, also ignore 
    # C_i - Cg_i+1, C_i - Cd_i+1, O_i - Cd_i+1, and Ca_i - Cd_i+1.
    ca_one_hot = F.one_hot(residue_index.new_tensor(1), num_classes=14).type(fp_type)
    o_one_hot = F.one_hot(residue_index.new_tensor(3), num_classes=14).type(fp_type)    
    pro = rc.restype_name_to_atom14_names["PRO"]
    pro_cg_idx = pro.index("CG")
    pro_cg_idx = residue_index.new_tensor(pro_cg_idx)
    pro_cg_one_hot = F.one_hot(pro_cg_idx, num_classes=14).type(fp_type)
    pro_cd_idx = pro.index("CD")
    pro_cd_idx = residue_index.new_tensor(pro_cd_idx)
    pro_cd_one_hot = F.one_hot(pro_cd_idx, num_classes=14).type(fp_type)
    pro_res2 = aatype[valid_pairs[1]] == rc.restype_order["P"]
    pro_neighbor_mask = pro_res2 * neighbor_mask # [N_pairs]
    c_pro_cg_dists = pro_neighbor_mask[..., None, None] * c_one_hot[None, :, None] * pro_cg_one_hot[None, None, :]
    c_pro_cd_dists = pro_neighbor_mask[..., None, None] * c_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    o_pro_cd_dists = pro_neighbor_mask[..., None, None] * o_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    ca_pro_cd_dists = pro_neighbor_mask[..., None, None] * ca_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    dists_mask = dists_mask * (1.0 - c_pro_cg_dists) * (1.0 - c_pro_cd_dists) * (1.0 - o_pro_cd_dists) * (1.0 - ca_pro_cd_dists)
    
    # Compute the lower bound for the allowed distances.
    # shape (N_pairs, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[valid_pairs[0]][..., :, None]
        + atom14_atom_radius[valid_pairs[1]][..., None, :]
    )

    # Compute the error.
    # shape (N_pairs, 14, 14)
    dists_to_low_error = dists_mask * F.relu(
        dists_lower_bound - clash_overlap_tolerance - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (eps + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.zeros_like(atom14_atom_exists)
    per_atom_loss_sum = per_atom_loss_sum.index_add(0, valid_pairs[0], torch.sum(dists_to_low_error, dim=2))
    per_atom_loss_sum = per_atom_loss_sum.index_add(0, valid_pairs[1], torch.sum(dists_to_low_error, dim=1))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = (per_atom_loss_sum > 0.0).long()

    clash_info = {
            "mean_loss": mean_loss,  # shape ()
            "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
            "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

    return clash_info


def find_clashing_residues(batch, atom14_pred_positions, clash_overlap_tolerance=0.6) -> torch.Tensor:
    # NOTE: This assumes that batch has only 1 protein in it.
    
    # Find the clashing atoms and energy
    clash_info = local_interresidue_sc_clash_loss(batch, atom14_pred_positions, clash_overlap_tolerance)
    atom_clash_mask = clash_info["per_atom_clash_mask"].squeeze()
    clash_energy = clash_info["mean_loss"].squeeze()
    
    # Get residue indices of residues that have at least one clashing atom
    clashing_residues = torch.unique(torch.where(atom_clash_mask)[0])
    
    return clashing_residues, clash_energy


def find_unclosed_prolines(batch, atom14_pred_positions, tolerance_factor=12) -> torch.Tensor:
    # Mean and standard deviation of the CD-N bond length in proline
    # (from stereo_chemical_props.txt)
    pro_CD_N_mean = 1.474
    pro_CD_N_std = 0.014
    
    # Find proline residues
    pro_mask = batch["S"].squeeze() == rc.restype_order['P']
    
    # Get the CD-N bond lengths
    pro_N = pro_mask[..., None] * atom14_pred_positions.squeeze()[..., rc.restype_name_to_atom14_names["PRO"].index("N"), :]
    pro_CD = pro_mask[..., None] * atom14_pred_positions.squeeze()[..., rc.restype_name_to_atom14_names["PRO"].index("CD"), :]
    pro_CD_N = torch.norm(pro_CD - pro_N, dim=-1)
    
    # Find unclosed prolines and energy based on tolerance factor
    dists = pro_CD_N - (pro_CD_N_mean + tolerance_factor * pro_CD_N_std)
    pro_unclosed_mask = dists > 0.0
    
    # Get residue indices of unclosed prolines
    pro_unclosed = torch.unique(torch.where(pro_unclosed_mask)[0])
    
    return pro_unclosed


def get_proline_chi_bins(batch, atom14_pred_positions, proline_indices=None) -> torch.Tensor:
    if proline_indices is None:
        # Find proline residues
        pro_mask = batch["S"].squeeze() == rc.restype_order['P']
        
        # Get the proline chi values and bins
        SC_D, _ = Featurizer.calc_sc_dihedrals(atom14_pred_positions.squeeze()[pro_mask], batch["S"].squeeze()[pro_mask])
    else:
        # Get the proline chi values and bins
        SC_D, _ = Featurizer.calc_sc_dihedrals(atom14_pred_positions.squeeze()[proline_indices], batch["S"].squeeze()[proline_indices])
    SC_D_bin = (SC_D + torch.pi) // (2 * torch.pi / (batch['chi_logits'].shape[-1] - 1))
    
    return SC_D_bin


def resample_clashes(batch, atom14_pred_positions, clashing_indices, temperature=0.0) -> torch.Tensor:
    # Get X, S, BB_D, and chi_logits of clashing residues
    resampled_X = batch["X"].squeeze()[clashing_indices]
    resampled_S = batch["S"].squeeze()[clashing_indices]
    resampled_BB_D = batch["BB_D"].squeeze()[clashing_indices]
    resampled_chi_logits = batch["chi_logits"].squeeze()[clashing_indices]
    
    # Get resampled chi values
    if temperature > 0.0:
        logits = resampled_chi_logits / temperature
        chi_probs = F.softmax(logits, -1)
        chi_bin = torch.multinomial(chi_probs.view(-1, logits.shape[-1]), 1).view(*logits.shape[:2], -1).squeeze(-1)
    else:
        chi_bin = torch.argmax(F.softmax(resampled_chi_logits, -1), dim=-1)
    chi_bin_one_hot = F.one_hot(chi_bin, num_classes=resampled_chi_logits.shape[-1])
    chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / (resampled_chi_logits.shape[-1] - 1), device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
    pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
    chi_bin_offset = batch.get('chi_bin_offset', None)
    if chi_bin_offset is not None:
        bin_sample_update = chi_bin_offset.squeeze()[clashing_indices]
    else:
        bin_sample_update = (2 * torch.pi / (resampled_chi_logits.shape[-1] - 1)) * torch.rand(chi_bin.shape, device=chi_bin.device)
    chi_pred = pred_chi_bin + bin_sample_update
    
    # Construct resampled atom14 coordinates
    aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[resampled_S]
    chi_pred = aatype_chi_mask * chi_pred
    resampled_atom14_xyz = get_atom14_coords(resampled_X, resampled_S, resampled_BB_D, chi_pred)
    
    # Update coordinate tensor
    resampled_coords = atom14_pred_positions.clone().squeeze()
    resampled_coords[clashing_indices] = resampled_atom14_xyz
    
    return resampled_coords


def resample_prolines(batch, atom14_pred_positions, proline_indices, temperature=0.0, max_attempts=100) -> torch.Tensor:
    # Get X, S, BB_D, and chi_logits of unclosed proline residues
    resampled_X = batch["X"].squeeze()[proline_indices]
    resampled_S = batch["S"].squeeze()[proline_indices]
    resampled_BB_D = batch["BB_D"].squeeze()[proline_indices]
    resampled_chi_logits = batch["chi_logits"].squeeze()[proline_indices]
    
    # Get previous chi bins
    SC_D_bin = get_proline_chi_bins(batch, atom14_pred_positions, proline_indices).long()
    
    # Resample all unclosed prolines
    new_X = torch.zeros_like(resampled_X)
    for pro_i in range(len(proline_indices)):
        chi_history = [SC_D_bin[pro_i][:2].tolist()]
        temp = temperature
        
        replaced = False
        attempts = 0
        while not replaced and attempts < max_attempts:
            # Get resampled chi bins without repeating previous chis
            if temp > 0.0:
                logits = resampled_chi_logits[pro_i] / temp
                chi_probs = F.softmax(logits, -1)
                chi_bin = torch.multinomial(chi_probs, 1).squeeze(-1)
            else:
                chi_bin = torch.argmax(F.softmax(resampled_chi_logits[pro_i], -1), dim=-1)
            repeated_chis = chi_bin[:2].tolist() in chi_history
            if repeated_chis:
                temp += 0.2
                attempts += 1
                continue
            else:
                chi_history.append(chi_bin[:2].tolist())
                
                # Check if proline is closed
                chi_bin_one_hot = F.one_hot(chi_bin, num_classes=resampled_chi_logits.shape[-1])
                chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / (resampled_chi_logits.shape[-1] - 1), device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
                pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
                chi_bin_offset = batch.get('chi_bin_offset', None)
                if chi_bin_offset is not None:
                    bin_sample_update = chi_bin_offset.squeeze()[proline_indices][pro_i]
                else:
                    bin_sample_update = (2 * torch.pi / (resampled_chi_logits.shape[-1] - 1)) * torch.rand(chi_bin.shape, device=chi_bin.device)
                chi_pred = pred_chi_bin + bin_sample_update
                aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[rc.restype_order['P']]
                chi_pred = aatype_chi_mask * chi_pred
                resampled_atom14_xyz = get_atom14_coords(resampled_X[pro_i], resampled_S[pro_i], resampled_BB_D[pro_i], chi_pred)
                
                closed = len(find_unclosed_prolines({"S": resampled_S[pro_i]}, resampled_atom14_xyz)) == 0
                
                if closed:
                    replaced = True
                    new_X[pro_i] = resampled_atom14_xyz
                else:
                    temp += 0.2
                    attempts += 1
                    continue
        # If the proline was not replaced, use the previous coordinates
        if not replaced:
            new_X[pro_i] = atom14_pred_positions.squeeze()[proline_indices][pro_i]
    
    # Update coordinate tensor
    all_atom = atom14_pred_positions.clone().squeeze()
    all_atom[proline_indices] = new_X
    
    return all_atom


def resample_loop(batch, atom14_pred_positions, sample_temp=0.5, clash_overlap_tolerance=0.6, pro_tolerance_factor=12, max_iters=10, metropolis_temp=5e-6, verbose=0) -> torch.Tensor:
    # Find clashing residues and energy
    clashing_residues, clash_energy = find_clashing_residues(batch, atom14_pred_positions, clash_overlap_tolerance)
    if verbose:
        print('Number of initial clashing residues to resample:', len(clashing_residues))
        print('Initial clash energy:', clash_energy)
    
    # Resample clashing residues
    resampled_coords = atom14_pred_positions.clone()
    resampled_energy = clash_energy.clone()
    resampled_iter = -1
    for i in range(max_iters):
        # If there are no violations, break
        if resampled_energy == 0.0:
            break
        
        # Resample clashes
        temp_coords = resample_clashes(batch, resampled_coords, clashing_residues, sample_temp)
        
        # Find new clashing residues and energy
        clashing_residues, clash_energy = find_clashing_residues(batch, temp_coords, clash_overlap_tolerance)
        if verbose:
            print(f'Number of clashing residues to resample (iter {i}):', len(clashing_residues))
            print(f'Clash energy (iter {i}):', clash_energy)
        
        # Update bests based on Metropolis Criterion
        if clash_energy < resampled_energy:
            resampled_coords = temp_coords.clone()
            resampled_energy = clash_energy.clone()
            resampled_iter = i
        else:
            if torch.rand(1, device=atom14_pred_positions.device) < torch.exp(-(clash_energy - resampled_energy) / metropolis_temp):
                if verbose:
                    print('Metropolis criterion accepted.')
                resampled_coords = temp_coords.clone()
                resampled_energy = clash_energy.clone()
                resampled_iter = i
              
    if verbose:
        print('Final energy:', resampled_energy)
        print('Final iteration:', resampled_iter)
                 
    # Find unclosed prolines
    unclosed_prolines = find_unclosed_prolines(batch, resampled_coords, pro_tolerance_factor)
    if verbose:
        print('Number of unclosed prolines to resample:', len(unclosed_prolines))
        
    # Resample unclosed proline residues
    resampled_coords = resample_prolines(batch, resampled_coords, unclosed_prolines, sample_temp)
    unclosed_prolines = find_unclosed_prolines(batch, resampled_coords, pro_tolerance_factor)
    if verbose:
        print('Remaining unclosed prolines:', len(unclosed_prolines))
    
    return resampled_coords, resampled_energy