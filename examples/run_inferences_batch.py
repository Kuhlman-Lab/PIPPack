

import os
import hydra
from omegaconf import DictConfig
import pippack
from pippack import PIPPack

script_path=os.path.abspath(os.path.dirname(pippack.__file__))


@hydra.main(version_base=None, config_path=os.path.join(script_path,'config'), config_name="inference")
def main(cfg: DictConfig) -> None:
    
    packer=PIPPack(model=cfg.inference.model_name)
    packer.weights_path=cfg.inference.weights_path
    if not packer.use_ensemble:
        packer._initialize_with_a_model()
    else:
        packer._initialize_with_ensemble()

    packer._run_repack_batch(pdb_path=cfg.inference.pdb_path,output_dir=cfg.inference.output_dir)

if __name__ == '__main__':
    main()