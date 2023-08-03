import argparse
import os, glob
from functools import partial
from multiprocessing import Pool, Manager


clashscore = "/proj/kuhl_lab/MolProbity/molprobity/cmdline/clashscore"


def _compute_clashscore(pdb_file, out_file, lock):
    # Write clash score to temporary file
    tmp_file = os.path.join(os.path.dirname(out_file), f"tmp_clashscore_{os.getpid()}.txt")
    os.system(f'{clashscore} {pdb_file} | grep "clashscore =" > {tmp_file}')
    
    with lock:
        with open(tmp_file, 'r') as f_in, open(out_file, 'a') as f_out:
            score_str = f_in.read()
            f_out.write(f"{os.path.basename(pdb_file)}: {score_str}") 
    
    
def main(args):
    # Get list of pdb files to score
    pdb_files = glob.glob(os.path.join(args.pdb_dir, '*.pdb'))
    
    # Create the output file
    out_file = os.path.join(args.pdb_dir, "clashscore_scores.txt")
    with open(out_file, 'w') as f:
        f.write("Clashscore Scores:\n")

    # Run scoring
    with Manager() as manager:
        lock = manager.Lock()
        with Pool(args.num_workers) as p:
            p.map(partial(_compute_clashscore, out_file=out_file, lock=lock), pdb_files)
            
    # Clean up
    for f in glob.glob(os.path.join(args.pdb_dir, "tmp_clashscore_*.txt")):
        os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, help='Directory containing pdb files to score.')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    main(args)
