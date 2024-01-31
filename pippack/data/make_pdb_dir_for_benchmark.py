import os, json
import argparse
import shutil
from pippack.data.top2018_dataset import *

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = os.path.join(args.data_dir, f"top2018_{args.filter_level}_v{args.version}")
    ds = Top2018Dataset(path=data_dir, filter_level=args.filter_level, version=args.version)
    with open(os.path.join(data_dir, args.split_file), 'r') as f:
        chain_split = json.load(f)
    ds._prune_from_cluster_split(args.seq_id, chain_split[args.mode])

    for pdb_file in ds.pdb_files:
        shutil.copy(pdb_file, os.path.join(args.output_dir, os.path.basename(pdb_file)))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/proj/kuhl_lab/users/nzrandol/protein-datasets')
    parser.add_argument("--filter_level", type=str, default="mc")
    parser.add_argument("--version", type=str, default="2.01")
    parser.add_argument("--seq_id", type=str, default="40pc")
    parser.add_argument("--split_file", type=str, default='chain_splits_40pc.json')
    parser.add_argument("--mode", type=str, default="valid")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main()
