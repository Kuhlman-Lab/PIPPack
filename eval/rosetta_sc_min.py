import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import time, glob
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.scoring import fa_rep


def sc_min_task(pdb_in, pdb_out, fa_rep_weight=0.55, restrict_aa=[]):
    t0 = time.time()

    # Load initial pose
    pose = pose_from_pdb(pdb_in)

    # Get score function and score pose
    scorefxn = get_fa_scorefxn()
    scorefxn.set_weight(fa_rep, fa_rep_weight)
    scorefxn(pose)

    # Set up MoveMap with only movable side chains
    movemap = MoveMap()
    movemap.set_bb(False)
    for res in range(1, pose.total_residue() + 1):
        if pose.residue(res).name3() not in restrict_aa:
            movemap.set_chi(res, True)
        
    # Construct MinMover
    minmover = MinMover()
    minmover.score_function(scorefxn)
    minmover.movemap(movemap)

    # Score, apply, and score
    print('\nPre minimzation score:', scorefxn(pose))
    minmover.apply(pose)
    print('Post minimization score:', scorefxn(pose))

    # Output
    pose.dump_pdb(pdb_out)

    print(f"Minimized side chains in {time.time() - t0:.3f} sec.")


def safe_run(arg, fa_rep_weight=0.55, restrict_aa=[]):
    try:
        sc_min_task(*arg, fa_rep_weight=fa_rep_weight, restrict_aa=restrict_aa)
    except Exception:
        print(f"Skipping {arg}...")


def main(args):
    # Initialize PyRosetta
    init('-ignore_unrecognized_res 1 -detect_disulf 0')

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Construct work list of input PDBs and output PDBs
    pdb_list = sorted(glob.glob(os.path.join(args.data_dir, "*.pdb")))
    work_list = []
    for pdb_in in pdb_list:
        pdb_out = os.path.join(args.out_dir, os.path.basename(pdb_in))
        if args.overwrite or os.path.basename(pdb_out) not in os.listdir(args.out_dir):
            work_list.append((pdb_in, pdb_out))
    
    # Run minimization
    with Pool(args.n_thread) as p:
        p.map(partial(safe_run, fa_rep_weight=args.fa_rep, restrict_aa=args.restrict_aa), work_list)

        
if __name__ == "__main__":
    parser = ArgumentParser(description="Rosetta Side Chain Minimization",
                            epilog="Run Rosetta minimization on protein side chains.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir',
                        type=str, default=None,
                        help='Directory containing pdb files to minimize side chains.')
    parser.add_argument('--out_dir',
                        default=None, type=str,
                        help="Directory to store outputs. Default is <data_dir>_rosetta_min.")
    parser.add_argument('--fa_rep',
                        type=float, default=0.55,
                        help="Weight for fa_rep score term in ref2015")
    parser.add_argument("--n_thread", type=int, default=24,
                        help="Number of CPU threads to use for minimization.")
    parser.add_argument("--overwrite",
                        action="store_true", 
                        help="Optionally overwrite previous results.")
    parser.add_argument("--restrict_aa",
                        type=str, default='',
                        help="Comma-separated list of which amino acids to restrict from minimization.")
    args = parser.parse_args()

    if args.out_dir is None:
        args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == os.path.sep else args.data_dir
        args.out_dir = args.data_dir + f"_rosetta_min"

    args.restrict_aa = args.restrict_aa.split(',')

    main(args)
