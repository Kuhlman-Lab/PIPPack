import os
import logging
from typing import *
import numpy as np
import torch
from torchmetrics import MeanMetric, MaxMetric, MinMetric, Metric
import torch.nn.functional as F

from pippack.data.featurizer import calc_sc_dihedrals
import pippack.data.residue_constants as rc


logger = logging.getLogger(__name__)


class BlackHole:
    """Dummy object."""
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, eps: float = 1e-4) -> torch.Tensor:
    
    mask = mask.expand(*value.shape)
    
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def supervised_chi_loss(
    pred_norm_chi_sincos: torch.Tensor,
    pred_unnorm_chi_sincos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    true_chi_sincos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    _metric=None,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)
        Args:
            pred_norm_chi_sincos:
                [*, N, 4, 2] predicted angles
            pred_unnorm_chi_sincos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 4] angle mask
            true_chi_sincos:
                [*, N, 4, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    
    residue_type_one_hot = F.one_hot(aatype, rc.restype_num + 1) # [*, Nres, 21]
    chi_pi_periodic = torch.einsum(
        "...ij,jk->...ik",
        residue_type_one_hot.to(pred_norm_chi_sincos.dtype),
        pred_norm_chi_sincos.new_tensor(rc.chi_pi_periodic)) # [*, Nres, 4]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1) # [*, Nres, 4, 1]
    true_chi_shifted = shifted_mask * true_chi_sincos # [*, Nres, 4, 2]
    sq_chi_error = torch.sum((true_chi_sincos - pred_norm_chi_sincos) ** 2, dim=-1) # [*, Nres, 4]
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_norm_chi_sincos) ** 2, dim=-1) # [*, Nres, 4]
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted) # [*, Nres, 4]

    # Update chi mask with seq mask
    chi_mask = chi_mask * seq_mask[..., None]

    # Chi loss
    sq_chi_loss = torch.sum(chi_mask * sq_chi_error, dim=(-1, -2)) / torch.sum(chi_mask + eps, dim=(-1, -2))
    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(pred_unnorm_chi_sincos ** 2, dim=-1) + eps) # [*, Nres, 4]
    norm_error = torch.abs(angle_norm - 1.0) # [*, Nres, 4]
    angle_norm_loss = torch.sum(chi_mask * norm_error, dim=(-1, -2)) / torch.sum(chi_mask + eps, dim=(-1, -2))

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        loss = _metric(sq_chi_error, chi_mask)

    return loss


def wrapped_chi_angle(CH_pred, CH_true):

    # Determine which rotamers are correct via wrapped from negative end
    dist_neg_edge = torch.abs(CH_pred - -180.0)
    close_to_neg_edge = dist_neg_edge <= 20.0
    wrapped_dist = 20.0 - dist_neg_edge
    wrapped_rot_neg = (CH_true > 180.0 - wrapped_dist) * close_to_neg_edge

    # Determine which rotamers are correct via wrapping from positive end
    dist_pos_edge = torch.abs(CH_pred - 180.0)
    close_to_pos_edge = dist_pos_edge <= 20.0
    wrapped_dist = 20.0 - dist_pos_edge
    wrapped_rot_pos = (CH_true < -180.0 + wrapped_dist) * close_to_pos_edge
    
    return wrapped_rot_neg + wrapped_rot_pos


def pi_periodic_rotamer(CH_pred, CH_true, S):
    
    # Get which chis are pi periodic
    residue_type_one_hot = F.one_hot(S, 21)
    chi_pi_periodic = torch.einsum(
        "...ij, jk->...ik",
        residue_type_one_hot.type(CH_pred.dtype),
        CH_pred.new_tensor(np.array(rc.chi_pi_periodic))
    )
    
    # Shift for the predicted chis
    shift = (CH_pred < CH_true) * 180.0 + (CH_pred > CH_true) * -180.0
    
    return (torch.abs(CH_pred + shift - CH_true) <= 20.0) * chi_pi_periodic


def rotamer_recovery_from_coords(S, true_SC_D, pred_X, residue_mask, SC_D_mask, return_raw=False, return_chis=False, exclude_AG=True, _metric=None):

    # Compute true and predicted chi dihedrals (in degrees)
    CH_true = true_SC_D * 180. / torch.pi
    CH_pred = torch.nan_to_num(calc_sc_dihedrals(pred_X, S, return_mask=False)) * 180. / torch.pi

    # Determine correct chis based on angle difference
    angle_diff_chis = (torch.abs(CH_true - CH_pred) <= 20.0) * SC_D_mask # [B, L, 4]

    # Determine correct chis based on non-existant chis
    nonexistent_chis = (1. - SC_D_mask) # [B, L, 4]

    # Determine correct chis based on wrapping of dihedral angles around -180. and 180.
    wrapped_chis = wrapped_chi_angle(CH_pred, CH_true) * SC_D_mask # [B, L, 4]
    
    # Determine correct chis based on periodic chis
    periodic_chis = pi_periodic_rotamer(CH_pred, CH_true, S) * SC_D_mask # [B, L, 4]
    
    # Sum to determine correct chis
    correct_chis = angle_diff_chis + nonexistent_chis + wrapped_chis + periodic_chis # [B, L, 4]

    # Determine correct rotamers based on all correct chi
    correct_rotamer = torch.sum(correct_chis, dim=-1) == 4 # [B, L]
    
    # Exclude Ala and Gly
    if exclude_AG:
        ala_mask = (S == rc.restype_order['A']).float() # [B, L]
        gly_mask = (S == rc.restype_order['G']).float() # [B, L]
        residue_mask = residue_mask * (1. - ala_mask) * (1. - gly_mask) # [B, L]

    # Determine average number of correct rotamers for each chain (depending on that chains length)
    if _metric is not None and not isinstance(_metric, BlackHole):
        rr = _metric(correct_rotamer, residue_mask)
    else:
        rr = torch.sum(correct_rotamer * residue_mask, dim=-1) / torch.sum(residue_mask, dim=-1)

    if return_raw:
        return correct_rotamer
    if return_chis:
        return correct_chis
    else:
        return torch.mean(rr)


def nll_chi_loss(chi_log_probs, true_chi_bin, sequence, chi_mask, _metric=None):
    """ Negative log probabilities for binned chi prediction """
    
    # Get which chis are pi periodic
    residue_type_one_hot = F.one_hot(sequence.long(), 21)
    chi_pi_periodic = torch.einsum(
        "...ij, jk->...ik",
        residue_type_one_hot.type(chi_log_probs.dtype),
        chi_mask.new_tensor(np.array(rc.chi_pi_periodic))
    )

    # Create shifted true chi bin for the pi periodic chis    
    n_bins = chi_log_probs.shape[-1]
    shift_val = (n_bins - 1) // 2
    shift = (true_chi_bin >= shift_val) * -shift_val + (true_chi_bin < shift_val) * shift_val
    true_chi_bin_shifted = true_chi_bin + shift * chi_pi_periodic

    # NLL loss for shifted and unshifted predictions
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        chi_log_probs.contiguous().view(-1, n_bins), true_chi_bin.long().contiguous().view(-1)
    ).view(true_chi_bin.size())
    loss_shifted = criterion(
        chi_log_probs.contiguous().view(-1, n_bins), true_chi_bin_shifted.long().contiguous().view(-1)
    ).view(true_chi_bin.size())
    
    # Determine masked loss and loss average
    loss = torch.minimum(loss, loss_shifted) * chi_mask
    loss_av = torch.sum(loss) / torch.sum(chi_mask)
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        loss_av = _metric(loss, chi_mask)
    
    return loss_av


def offset_mse(offset_pred, offset_true, mask, n_chi_bin=36, scale_pred=True, _metric=None):
    if scale_pred:
        offset_pred = (2 * torch.pi / n_chi_bin) * offset_pred
        
    err = torch.sum(mask * (offset_pred - offset_true) ** 2) / torch.sum(mask)
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        err = _metric((offset_pred - offset_true) ** 2, mask)
    
    return err


def get_renamed_coords(X: torch.Tensor, S: torch.Tensor, pseudo_renaming: bool = False) -> torch.Tensor:
    # Determine which atoms should be swapped.
    if pseudo_renaming:
        atom_renaming_swaps = rc.residue_atom_pseudo_renaming_swaps
    else:
        atom_renaming_swaps = rc.residue_atom_renaming_swaps
    
    # Rename symmetric atoms
    renamed_X = X.clone()
    for restype in atom_renaming_swaps:
        # Get mask based on restype
        restype_idx = rc.restype_order[rc.restype_3to1[restype]]
        restype_mask = S == restype_idx
        
        # Swap atom coordinates for restype
        restype_X = renamed_X * restype_mask[..., None, None]
        for atom_pair in atom_renaming_swaps[restype]:
            atom1, atom2 = atom_pair
            atom1_idx, atom2_idx = rc.restype_name_to_atom14_names[restype].index(atom1), rc.restype_name_to_atom14_names[restype].index(atom2)

            restype_X[..., atom1_idx, :] = X[..., atom2_idx, :]
            restype_X[..., atom2_idx, :] = X[..., atom1_idx, :]
        
        # Update full tensor
        restype_X = torch.nan_to_num(restype_X) * restype_mask[..., None, None]
        renamed_X = renamed_X * ~restype_mask[..., None, None] + restype_X
        
    return renamed_X


def sc_rmsd(decoy_X, true_X, S, X_mask, residue_mask, _metric=None, use_sqrt=False):
    # Compute atom deviation based on original coordinates
    atom_deviation = torch.sum(torch.square(decoy_X - true_X), dim=-1)

    # Compute atom deviation based on alternative coordinates
    true_renamed_X = get_renamed_coords(true_X, S)
    renamed_atom_deviation = torch.sum(torch.square(decoy_X - true_renamed_X), dim=-1)
    
    # Get atom mask including backbone atoms
    atom_mask = X_mask * residue_mask[..., None]
    atom_mask[..., :4] = 0.0

    # Compute RMSD based on original and alternative coordinates
    rmsd_og = masked_mean(atom_mask, atom_deviation, -1)
    rmsd_renamed = masked_mean(atom_mask, renamed_atom_deviation, -1)
    if use_sqrt:
        rmsd_og = torch.sqrt(rmsd_og)
        rmsd_renamed = torch.sqrt(rmsd_renamed)
    rmsd = torch.minimum(
        rmsd_og,
        rmsd_renamed
    )

    if _metric is not None and not isinstance(_metric, BlackHole):
        mse = _metric(rmsd)
    
    return mse


# TODO: Figure out if any changes are necessary for max and min metrics. I don't think that
# the logging functionality will be correct for these.
class MetricLogger:
    def __init__(
        self, 
        log_file: str,
        mean_metrics: Sequence[str] = [],
        max_metrics: Sequence[str] = [],
        min_metrics: Sequence[str] = [],
    ) -> None:
        self.log_file = log_file
        self.mean_metrics = {
            metric: MeanMetric() for metric in mean_metrics
        }
        self.max_metrics = {
            metric: MaxMetric() for metric in max_metrics
        }
        self.min_metrics = {
            metric: MinMetric() for metric in min_metrics
        }
        self._initialize_log()

    def _initialize_log(self) -> None:
        if os.path.exists(self.log_file):
            raise ValueError(f"Log file {self.log_file} already exists.")
        
        # Create logging column headers.
        log_cols = ["epoch"]
        log_cols.extend([metric + " mean" for metric in self.mean_metrics])
        log_cols.extend([metric + " max" for metric in self.max_metrics])
        log_cols.extend([metric + " min" for metric in self.min_metrics])
        
        # Create CSV file with header row.
        with open(self.log_file, "w") as f:
            f.write(",".join(log_cols) + "\n")

    def to(self, device: str):
        for metric in self.mean_metrics:
            self.mean_metrics[metric] = self.mean_metrics[metric].to(device)
        for metric in self.max_metrics:
            self.max_metrics[metric] = self.max_metrics[metric].to(device)
        for metric in self.min_metrics:
            self.min_metrics[metric] = self.min_metrics[metric].to(device)

        return self

    def update(self, key: str, value: Union[float, torch.Tensor], weight: Union[float, torch.Tensor] = 1.0) -> None:
        # Update all metrics matching <key>.
        if key in self.mean_metrics:
            self.mean_metrics[key].update(value, weight)
        if key in self.max_metrics:
            self.max_metrics[key].update(value)
        if key in self.min_metrics:
            self.min_metrics[key].update(value)

    def compute(self) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = {
            "mean": {
                key: self.mean_metrics[key].compute() for key in self.mean_metrics
            },
            "max": {
                key: self.max_metrics[key].compute() for key in self.max_metrics
            },
            "min": {
                key: self.min_metrics[key].compute() for key in self.min_metrics
            }
        }
        return metrics
    
    def log(self, epoch: int, precision: int = 5) -> None:
        # Compute metrics and create list for CSV logging.
        computed_metrics = self.compute()
        log_cols = [epoch, *[round(metric.item(), precision) for mode in computed_metrics for metric in computed_metrics[mode].values()]]
        log_cols = list(map(str, log_cols))
        
        # Write to log file.
        with open(self.log_file, "a") as f:
            f.write(",".join(log_cols) + "\n")
            
        # Reset metrics for next logging step.
        self.reset_metrics()

    def reset_metrics(self) -> None:
        # Reset all metrics.
        [metric.reset() for metric in self.mean_metrics.values()]
        [metric.reset() for metric in self.max_metrics.values()]
        [metric.reset() for metric in self.min_metrics.values()]

    def get_metric(self, key: str, type: str = "mean") -> Metric:
        if type == "mean":
            return self.mean_metrics[key]
        elif type == "max":
            return self.max_metrics[key]
        elif type == "min":
            return self.min_metrics[key]
        else:
            raise ValueError(f"Unrecognized type {type}. Use 'mean', 'max', or 'min'.")


def interresidue_sc_clash_loss(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clash_overlap_tolerance: float, # OpenFold value is 1.5
    eps: float = 1e-10,
    _metric = None,
    return_clashing_pairs: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations resulting from sidechains.
    
    Note: This ignores intra-residue clashes and backbone-backbone clashes.
    """
    
    # Get needed components from batch.
    aatype = batch["S"].clone()
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
    atom14_atom_exists = batch["X_mask"].clone()
    residue_index = batch["residue_index"].clone().long()
    
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
    
    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    fp_type = atom14_pred_positions.dtype
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone-backbone clashes are ignored. CB is included in the backbone.
    bb_bb_mask = torch.zeros_like(dists_mask)
    bb_bb_mask[..., :5, :5] = 1.0
    dists_mask = dists_mask * (1.0 - bb_bb_mask)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)
    
    # Mask interactions between side chain and backbone when atoms are separated by less than 4 bonds.
    # For all residues, ignore Cb_i - N_i+1 and C_i - Cb_i+1.
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    cb_one_hot = F.one_hot(residue_index.new_tensor(4), num_classes=14)
    cb_one_hot = cb_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *cb_one_hot.shape
    )
    cb_one_hot = cb_one_hot.type(fp_type)
    neighbor_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]
    cb_n_dists = (
        neighbor_mask
        * cb_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    c_cb_dists = (
        neighbor_mask
        * c_one_hot[..., None, None, :, None]
        * cb_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - cb_n_dists) * (1.0 - c_cb_dists)
    
    # For PRO at i+1, also ignore 
    # C_i - Cg_i+1, C_i - Cd_i+1, O_i - Cd_i+1, and Ca_i - Cd_i+1.
    ca_one_hot = F.one_hot(residue_index.new_tensor(1), num_classes=14)
    ca_one_hot = ca_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *ca_one_hot.shape
    )
    ca_one_hot = ca_one_hot.type(fp_type)
    o_one_hot = F.one_hot(residue_index.new_tensor(3), num_classes=14)
    o_one_hot = o_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *o_one_hot.shape
    )
    o_one_hot = o_one_hot.type(fp_type)
    pro = rc.restype_name_to_atom14_names["PRO"]
    pro_cg_idx = pro.index("CG")
    pro_cg_idx = residue_index.new_tensor(pro_cg_idx)
    pro_cg_idx = pro_cg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    pro_cg_one_hot = F.one_hot(pro_cg_idx, num_classes=14).type(fp_type)
    pro_cd_idx = pro.index("CD")
    pro_cd_idx = residue_index.new_tensor(pro_cd_idx)
    pro_cd_idx = pro_cd_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    pro_cd_one_hot = F.one_hot(pro_cd_idx, num_classes=14).type(fp_type)
    res_ip1_pro = aatype[..., 1:] == rc.restype_order["P"]
    res_ip1_pro = torch.cat(
        (
            res_ip1_pro,
            torch.zeros_like(res_ip1_pro[..., :1]),
        ),
        dim=-1,
    )
    pro_neighbor_mask = res_ip1_pro[..., None, None, None] * neighbor_mask
    c_pro_cg_dists = (
        pro_neighbor_mask
        * c_one_hot[..., None, None, :, None]
        * pro_cg_one_hot[..., None, None, None, :]
    )
    c_pro_cd_dists = (
        pro_neighbor_mask
        * c_one_hot[..., None, None, :, None]
        * pro_cd_one_hot[..., None, None, None, :]
    )
    o_pro_cd_dists = (
        pro_neighbor_mask
        * o_one_hot[..., None, None, :, None]
        * pro_cd_one_hot[..., None, None, None, :]
    )
    ca_pro_cd_dists = (
        pro_neighbor_mask
        * ca_one_hot[..., None, None, :, None]
        * pro_cd_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_pro_cg_dists) * (1.0 - c_pro_cd_dists) * (1.0 - o_pro_cd_dists) * (1.0 - ca_pro_cd_dists)
    
    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - clash_overlap_tolerance - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (eps + torch.sum(dists_mask))

    if _metric is not None and not isinstance(_metric, BlackHole):
        mean_loss = _metric(dists_to_low_error, dists_mask)

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - clash_overlap_tolerance)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    clash_info = {
            "mean_loss": mean_loss,  # shape ()
            "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
            "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

    if return_clashing_pairs:
        if len(batch["S"].shape) == 1:
            aatype = aatype.unsqueeze(0)
            clash_mask = clash_mask.unsqueeze(0)
            residue_index = residue_index.unsqueeze(0)
        
        clashing_pairs = []
        for b_idx in range(aatype.shape[0]):
            res1, res2, atom1, atom2 = map(list, torch.where(clash_mask[b_idx]))
            pairs = []
            for r1, r2, a1, a2 in zip(res1, res2, atom1, atom2):
                c1 = (
                    rc.restypes[aatype[b_idx][r1]]
                    + str(residue_index[b_idx][r1].item()) 
                    + ' ' 
                    + rc.restype_name_to_atom14_names[rc.restype_1to3[rc.restypes[aatype[b_idx][r1]]]][a1]
                )
                c2 = (
                    rc.restypes[aatype[b_idx][r2]]
                    + str(residue_index[b_idx][r2].item())
                    + ' '
                    + rc.restype_name_to_atom14_names[rc.restype_1to3[rc.restypes[aatype[b_idx][r2]]]][a2]
                )
                pairs.append((c1, c2))
            clashing_pairs.append(pairs)
        
        if len(batch["S"].shape) == 1:
            clashing_pairs = clashing_pairs[0]
            
        return clash_info, clashing_pairs
    else:
        return clash_info


def local_interresidue_sc_clash_loss(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    clash_overlap_tolerance: float, # OpenFold value is 1.5
    distance_threshold: float = 14.0,
    basis_atom: str = "CB",
    eps: float = 1e-10,
    _metric = None,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations resulting from sidechains.
    
    Note: This ignores intra-residue clashes and backbone-backbone clashes.
    """
    
    # Get needed components from batch.
    aatype = batch["S"].clone()
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
    atom14_atom_exists = batch["X_mask"].clone()
    residue_index = batch["residue_index"].clone().long()
    residue_mask = batch["residue_mask"].clone()
    atom14_pred_positions = atom14_pred_positions.clone()
    
    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (*, N, 14).
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
    # shape (*, N, 3)
    if basis_atom == "CB":
        basis_atom_idx = 4 * torch.ones_like(aatype)
        basis_atom_idx[aatype == rc.restype_order["G"]] = 1
    else:
        basis_atom_idx = rc.atom_order[basis_atom] * torch.ones_like(aatype)
    basis_xyz = torch.gather(atom14_pred_positions, -2, basis_atom_idx[..., None, None].expand(*atom14_pred_positions.shape))[..., :, 0, :]

    # Determine distances based on basis atoms.
    # shape (*, N, N)
    basis_dists = torch.sqrt(
        eps
        + torch.sum(
            (basis_xyz[..., None, :, :] - basis_xyz[..., :, None, :]) ** 2, dim=-1
        )
    )
    
    # Create the mask for valid residue pairs.
    # shape (*, N, N)
    fp_type = atom14_pred_positions.dtype
    dists_mask = (
        residue_mask[..., :, None]
        * residue_mask[..., None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (same residue pairs)
    dists_mask = dists_mask * (
        residue_index[..., :, None]
        < residue_index[..., None, :]
    )
    
    # Determine which residue pairs are within the distance threshold.
    # shape (*, N, N)
    dists_lower_bound = distance_threshold * torch.ones_like(dists_mask)
    dists_mask = dists_mask * (basis_dists < dists_lower_bound)
    valid_pairs = torch.where(dists_mask)
    
    # Get the atom14 coordinates for the valid residue pairs.
    # shape (N_pairs, 14, 3)
    res1_atom14_xyz = atom14_pred_positions.clone()[valid_pairs[0], valid_pairs[1]]
    res2_atom14_xyz = atom14_pred_positions.clone()[valid_pairs[0], valid_pairs[2]]
    
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
    cys_res1 = aatype[valid_pairs[0], valid_pairs[1]] == rc.restype_order["C"]
    cys_res2 = aatype[valid_pairs[0], valid_pairs[2]] == rc.restype_order["C"]
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
    neighbor_mask = (residue_index[valid_pairs[0], valid_pairs[1]] + 1) == residue_index[valid_pairs[0], valid_pairs[2]]
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
    pro_res2 = aatype[valid_pairs[0], valid_pairs[2]] == rc.restype_order["P"]
    pro_neighbor_mask = pro_res2 * neighbor_mask # [N_pairs]
    c_pro_cg_dists = pro_neighbor_mask[..., None, None] * c_one_hot[None, :, None] * pro_cg_one_hot[None, None, :]
    c_pro_cd_dists = pro_neighbor_mask[..., None, None] * c_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    o_pro_cd_dists = pro_neighbor_mask[..., None, None] * o_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    ca_pro_cd_dists = pro_neighbor_mask[..., None, None] * ca_one_hot[None, :, None] * pro_cd_one_hot[None, None, :]
    dists_mask = dists_mask * (1.0 - c_pro_cg_dists) * (1.0 - c_pro_cd_dists) * (1.0 - o_pro_cd_dists) * (1.0 - ca_pro_cd_dists)
    
    # Compute the lower bound for the allowed distances.
    # shape (N_pairs, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[valid_pairs[0], valid_pairs[1]][..., :, None]
        + atom14_atom_radius[valid_pairs[0], valid_pairs[2]][..., None, :]
    )

    # Compute the error.
    # shape (N_pairs, 14, 14)
    dists_to_low_error = dists_mask * F.relu(
        dists_lower_bound - clash_overlap_tolerance - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (eps + torch.sum(dists_mask))
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        mean_loss = _metric(dists_to_low_error, dists_mask)

    # Compute the per atom loss sum.
    # shape (N, 14)
    # TODO: Figure how to do this for batched data.
    #per_atom_loss_sum = torch.zeros_like(atom14_atom_exists)
    #per_atom_loss_sum = per_atom_loss_sum.index_add(1, valid_pairs[1], torch.sum(dists_to_low_error, dim=2))
    #per_atom_loss_sum = per_atom_loss_sum.index_add(1, valid_pairs[2], torch.sum(dists_to_low_error, dim=1))

    # Compute the per atom clash.
    # shape (N, 14)
    #per_atom_clash_mask = (per_atom_loss_sum > 0.0).long()

    clash_info = {
            "mean_loss": mean_loss,  # shape ()
    #        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
    #       "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

    return clash_info


def unclosed_proline_loss(batch, atom14_pred_positions, tolerance_factor=12, _metric=None, eps=1e-10) -> torch.Tensor:
    # Mean and standard deviation of the CD-N bond length in proline
    # (from stereo_chemical_props.txt)
    pro_CD_N_mean = 1.474
    pro_CD_N_std = 0.014
    
    # Find proline residues
    pro_mask = (batch["S"] == rc.restype_order['P']).float()
    pro_mask = pro_mask * batch.residue_mask
    
    # Get the CD-N bond lengths
    # (distances are summed rather than squared to avoid taking the square root)
    pro_atom_positions = atom14_pred_positions[pro_mask == 1]
    pro_N = pro_atom_positions[..., rc.restype_name_to_atom14_names["PRO"].index("N"), :]
    pro_CD = pro_atom_positions[..., rc.restype_name_to_atom14_names["PRO"].index("CD"), :]
    pro_CD_N = torch.sum((pro_CD - pro_N) ** 2, dim=-1)
    
    # Find unclosed prolines based on tolerance factor
    # (square the tolerance factor to avoid taking the square root)
    dists = F.relu(pro_CD_N - (pro_CD_N_mean + tolerance_factor * pro_CD_N_std) ** 2)

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists) / (eps + torch.sum(pro_mask))
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        mean_loss = _metric(dists, torch.ones_like(dists))
    
    return mean_loss
