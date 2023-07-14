import os
import logging
from typing import *
import numpy as np
import torch
from torchmetrics import MeanMetric, MaxMetric, MinMetric, Metric
import torch.nn.functional as F

from data.featurizer import calc_sc_dihedrals
import data.residue_constants as rc


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


def rmsd_loss(true_X, pred_X, X_mask, residue_mask, eps=1e-8, _return_frac=False, _metric=None):

    per_atom_sq_err = torch.sum((true_X - pred_X) ** 2, dim=-1) * X_mask * residue_mask[..., None]
    per_res_sq_err = torch.sum(per_atom_sq_err, dim=-1)
    per_res_atom_count = torch.sum(X_mask * residue_mask[..., None] + eps, dim=-1)
    
    total_sq_err = torch.sum(per_res_sq_err)
    total_atom_count = torch.sum(per_res_atom_count)
    rmsd = total_sq_err / total_atom_count
    #rmsd = torch.sqrt(total_sq_err / total_atom_count)
    
    if _metric is not None and not isinstance(_metric, BlackHole):
        rmsd = _metric(per_res_sq_err / per_res_atom_count, residue_mask)
    
    if _return_frac:
        return rmsd, (total_sq_err, total_atom_count)
    
    return rmsd

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
