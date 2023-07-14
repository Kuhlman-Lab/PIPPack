import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'


import time, glob, shutil
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn, Pose, standard_packer_task
from pyrosetta.toolbox.generate_resfile import generate_resfile_from_pose
from pyrosetta.rosetta.core.pack.task import parse_resfile
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover


def packer_init():
    init('-out:levels core.conformation.Conformation:error '
        'core.pack.pack_missing_sidechains:error '
        'core.pack.dunbrack.RotamerLibrary:error '
        'core.scoring.etable:error '
        '-ex1 -ex2 -ex3 -ex4 '
        '-multi_cool_annealer 5 '
        '-no_his_his_pairE '
        '-linmem_ig 1 '
        '-ignore_unrecognized_res 1 '
        '-detect_disulf 0')


def packer_task(pdb_in, pdb_out, n_decoy=1, tmp_dir="./tmp"):
    # Create temporary location for resfile
    uid = "res_file" + str(time.time_ns())
    resfile = os.path.join(tmp_dir, uid)

    # Generate resfile
    pose = pose_from_pdb(pdb_in)
    generate_resfile_from_pose(pose, resfile, pack=True, design=False, input_sc=False)

    # Run Rosetta
    best_score, best_pose = 1e10, Pose()
    scorefxn = get_fa_scorefxn()
    scorefxn(pose)
    for _ in range(n_decoy):
        test_pose = Pose()
        test_pose.assign(pose)

        # Set up Packer
        pose_packer = standard_packer_task(test_pose)
        parse_resfile(test_pose, pose_packer, resfile)
        packmover = PackRotamersMover(scorefxn, pose_packer)

        # Score, apply, and score
        print('\nPre packing score:', scorefxn(test_pose))
        packmover.apply(test_pose)
        print('Post packing score:', scorefxn(test_pose))
        score = scorefxn(test_pose)

        # Update best model
        if score < best_score:
            best_score = score
            best_pose = best_pose.assign(test_pose)

    # Output
    best_pose.dump_pdb(pdb_out)


def safe_run(arg, n_decoy, tmp_dir):
    try:
        packer_task(*arg, n_decoy=n_decoy, tmp_dir=tmp_dir)
    except Exception:
        print(f"Skipping {arg}...")


def main(args):
    # Initialize PyRosetta
    packer_init()

    # Create output directory and temporary directory (for resfiles)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = "./tmp" + str(time.time_ns())
    os.makedirs(tmp_dir, exist_ok=True)

    # Construct work list of input PDBs and output PDBs
    pdb_list = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    work_list = []
    for pdb_in in pdb_list:
        pdb_out = os.path.join(args.out_dir, os.path.basename(pdb_in))
        if args.overwrite or os.path.basename(pdb_out) not in os.listdir(args.out_dir):
            work_list.append((pdb_in, pdb_out))
    
    # Run benchmark
    with Pool(args.n_thread) as p:
        p.map(partial(safe_run, n_decoy=args.n_decoy, tmp_dir=tmp_dir), work_list)
    
    # Clean up
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rosetta Packer",
                            epilog="Run Rosetta's Packer on the specified set of PDBs.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('pdb_dir',
                        type=str,
                        help='Directory containing pdb files to benchmark.')
    parser.add_argument("--out_dir",
                        default=None, type=str,
                        help="Directory to store outputs. Default is <pdb_dir>_rosetta_packer.")
    parser.add_argument("--n_thread", type=int, default=24,
                        help="Number of CPU threads to use for packing.")
    parser.add_argument("--n_decoy", type=int, default=1,
                        help="Number of decoys for each target to run before selecting the decoy "
                        "with the best energy.")
    parser.add_argument("--overwrite",
                        action="store_true", 
                        help="Optionally overwrite previous results.")
    args = parser.parse_args()

    if args.out_dir is None:
        args.pdb_dir = args.pdb_dir[:-1] if args.pdb_dir[-1] == os.path.sep else args.pdb_dir
        args.out_dir = args.pdb_dir + f"_rosetta_packer"

    main(args)
