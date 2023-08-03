# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io
import gzip
from typing import Optional, Sequence, Union
import data.residue_constants as rc
from Bio.PDB import PDBParser
import numpy as np


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.'
            )


def from_pdb_string(pdb_str: str, model_idx: int = 0, chain_id: Optional[Union[str, Sequence[str]]] = None, discard_water: bool = True, mse_to_met: bool = False, ignore_non_std: bool = True) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        model_idx: The specific model in the PDB file that will be
            parsed. This is 0-indexed. Default is 0.
        chain_id: If chain_id is specified (e.g. A), then only those chains
            are parsed. Otherwise all chains are parsed.
        discard_water: Boolean specifying whether to ignore water molecules.
            Default is True.
        mse_to_met: Boolean specifying whether to convert MSE residues to MET residues.
            Default is False.
        ignore_non_std: Boolean specifying whether to ignore nonstandard residue types.
            If False, then they will be converted to UNK. Default is True.

    Returns:
        A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if model_idx is not None and model_idx > len(models) - 1:
        raise ValueError(
            f'Requested model index is out of range. Found {len(models)} models.'
        )
    elif model_idx is not None:
        model = models[model_idx]
    else:
        model = models[0]
    if isinstance(chain_id, str):
        chain_id = [chain_id]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    insertion_code_offset = 0
    for chain in sorted(model, key=lambda x: x.id):
        if chain_id is not None and chain.id not in chain_id:
            continue
        for res in sorted(chain, key=lambda x: x.id[1]):
            # Discard water residues.     
            if discard_water:
                if res.resname == 'HOH':
                    continue
            
            # Convert MSE residues to MET by changing only necessary fields.
            if mse_to_met:
                if res.resname == 'MSE':
                    res.resname = 'MET'
                    for atom in res:
                        if atom.name == 'SE':
                            atom.name = 'SD'
                                    
            # Ignore non-standard residues
            res_shortname = rc.restype_3to1.get(res.resname, 'X')
            if ignore_non_std:
                if res_shortname == 'X':
                    continue
            
            # Increment residue index offset if insertion code is detected.
            if res.id[2] != ' ':
                insertion_code_offset += 1
            
            restype_idx = rc.restype_order.get(
                res_shortname, rc.restype_num)
            pos = np.full((14, 3), fill_value=(np.nan))
            mask = np.zeros((14,))
            res_b_factors = np.zeros((14,))
            for atom in res:
                if atom.name not in rc.restype_name_to_atom14_names[res.resname]:
                    continue
                pos[rc.restype_name_to_atom14_names[res.resname].index(atom.name)] = atom.coord
                mask[rc.restype_name_to_atom14_names[res.resname].index(atom.name)] = 1.
                res_b_factors[rc.restype_name_to_atom14_names[res.resname].index(atom.name)] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            
            # Update protein-level lists
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1] + insertion_code_offset)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
            atom_positions=np.array(atom_positions),
            atom_mask=np.array(atom_mask),
            aatype=np.array(aatype),
            residue_index=np.array(residue_index),
            chain_index=chain_index,
            b_factors=np.array(b_factors)
    )


def from_pdb_file(pdb_file: str, **kwargs) -> Protein:
    # Obtain PDB string from PDB file.
    if pdb_file[-3:] == "pdb":
        with open(pdb_file, 'r') as f:
            pdb_str = f.read()
    elif pdb_file[-6:] == "pdb.gz":
        with gzip.open(pdb_file, "rb") as f:
            pdb_str = f.read().decode()
    else:
        raise ValueError("Unrecognized file type.")
        
    # Parse the string and get Protein.
    return from_pdb_string(pdb_str, **kwargs)


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
        f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
        prot: The protein to convert to PDB.

    Returns:
        PDB string.
    """
    restypes = rc.restypes + ['X']
    res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], 'UNK')
    atom_types = rc.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > rc.restype_num):
        raise ValueError('Invalid aatypes.')

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''

            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                         f'{residue_index[i]:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.
