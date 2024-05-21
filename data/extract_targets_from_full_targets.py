import os
import argparse
from biopandas.pdb import PandasPdb


full_target_to_target = {
    'T1104': {
        'target': ['T1104'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1106': {
        'target': ['T1106s1', 'T1106s2'],
        'chain': ['B', 'A'],
        'exclude_res': ['', ''],
    },
    'T1109': {
        'target': ['T1109'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1112': {
        'target': ['T1112'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1113': {
        'target': ['T1113'],
        'chain': ['B'],
        'exclude_res': [''],
    },
    'T1114': {
        'target': ['T1114s1', 'T1114s2', 'T1114s3'],
        'chain': ['H', 'B', 'A'],
        'exclude_res': ['', '', ''],
    },
    'T1119': {
        'target': ['T1119'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1120': {
        'target': ['T1120'],
        'chain': ['B'],
        'exclude_res': [''],
    },
    'T1121': {
        'target': ['T1121'],
        'chain': ['B'],
        'exclude_res': ['1'],
    },
    'T1122': {
        'target': ['T1122'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1123': {
        'target': ['T1123'],
        'chain': ['B'],
        'exclude_res': [''],
    },
    'T1124': {
        'target': ['T1124'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1125': {
        'target': ['T1125'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1129': {
        'target': ['T1129s2'],
        'chain': ['B'],
        'exclude_res': [''],
    },
    'T1132': {
        'target': ['T1132'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1133': {
        'target': ['T1133'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1134': {
        'target': ['T1134s1', 'T1134s2'],
        'chain': ['A', 'B'],
        'exclude_res': ['1', '168'],
    },
    'T1137': {
        'target': ['T1137s1', 'T1137s2', 'T1137s3', 'T1137s4', 'T1137s5', 'T1137s6', 'T1137s7', 'T1137s8', 'T1137s9'],
        'chain': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'],
        'exclude_res': ['18,19', '', '', '42,43', '', '', '', '14,15', ''],
    },
    'T1139': {
        'target': ['T1139'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1145': {
        'target': ['T1145'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1147': {
        'target': ['T1147'],
        'chain': ['C'],
        'exclude_res': [''],
    },
    'T1150': {
        'target': ['T1150'],
        'chain': ['A'],
        'exclude_res': ['263,343'],
    },
    'T1151': {
        'target': ['T1151s2'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1152': {
        'target': ['T1152'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1154': {
        'target': ['T1154'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1157': {
        'target': ['T1157s1', 'T1157s2'],
        'chain': ['A', 'B'],
        'exclude_res': ['', ''],
    },
    'T1158': {
        'target': ['T1158'],
        'chain': ['A'],
        'exclude_res': ['395-408,1297,1298'],
    },
    'T1159': {
        'target': ['T1159'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1169': {
        'target': ['T1169'],
        'chain': ['A'],
        'exclude_res': ['346-377,2736-3042'],
    },
    'T1170': {
        'target': ['T1170'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1173': {
        'target': ['T1173'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1174': {
        'target': ['T1174'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1176': {
        'target': ['T1176'],
        'chain': ['A'],
        'exclude_res': ['202'],
    },
    'T1178': {
        'target': ['T1178'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1179': {
        'target': ['T1179'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1180': {
        'target': ['T1180'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1185': {
        'target': ['T1185s1', 'T1185s2', 'T1185s4'],
        'chain': ['A', 'B', 'D'],
        'exclude_res': ['', '10', ''],
    },
    'T1187': {
        'target': ['T1187'],
        'chain': ['A'],
        'exclude_res': ['1'],
    },
    'T1188': {
        'target': ['T1188'],
        'chain': ['A'],
        'exclude_res': [''],
    },
    'T1194': {
        'target': ['T1194'],
        'chain': ['A'],
        'exclude_res': ['22,23,185-187'],
    },
}


def extract_target(full_target_file, output_file, chain, exclude_res):
    # Read full target PDB file
    ppdb = PandasPdb().read_pdb(full_target_file)
    
    # Extract target chain
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id'] == chain]
    if len(ppdb.df['ATOM']) == 0:
        raise ValueError(f'Chain {chain} not found in {full_target_file}')
    
    # Exclude residues
    if exclude_res:
        exclude_res = exclude_res.split(',')
        for res in exclude_res:
            if '-' in res:
                start, end = res.split('-')
                ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df['ATOM']['residue_number'] < int(start)) | (ppdb.df['ATOM']['residue_number'] > int(end))]
            else:
                ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['residue_number'] != int(res)]
                
    # Write target PDB file
    ppdb.to_pdb(output_file, records=['ATOM'])


def main(full_target_dir, output_dir):
    
    # Get list of full target PDB files, keeping those in the dict above
    full_target_files = [f for f in os.listdir(full_target_dir) if f.endswith('.pdb')]
    full_target_files = [f for f in full_target_files if f.split('.')[0] in full_target_to_target]
    
    # Make output directory, if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract target PDB files
    for full_target_file in full_target_files:
        full_target = full_target_file.split('.')[0]
        target_info = full_target_to_target[full_target]
        
        for target, chain, exclude_res in zip(target_info['target'], target_info['chain'], target_info['exclude_res']):
            output_file = f'{target}.pdb'
            output_file = os.path.join(output_dir, output_file)
            
            # Extract target PDB file
            extract_target(os.path.join(full_target_dir, full_target_file), output_file, chain, exclude_res)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('full_target_dir', help='Directory containing full target PDB files')
    parser.add_argument('output_dir', help='Directory to output extracted target PDB files')
    args = parser.parse_args()
    
    main(**vars(args))