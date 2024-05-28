# PIPPack
Implementation of Protein Invariant Point Packer (PIPPack)

PIPPack is a graph neural network (GNN) that utilizes geometry-aware invariant point message passing (IPMP) updates and recycling to rapidly generate accurate protein side chains. 

**PIPPack has now been published!** Please check out the [manuscript in Proteins](https://doi.org/10.1002/prot.26705)!

![PIPPack Architecture](./images/pippack_architecture.png)

## Quickstart
To get started right in your browser, click this button to open the PIPPack notebook in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kuhlman-Lab/PIPPack/blob/main/notebooks/PIPPack.ipynb)

Please let us know if you have any issues or suggestions!

## Getting started
To build the environment from scratch:
```
# Create and activate the pippack environment
conda create -n pippack
conda activate pippack

# Install PyTorch (see https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Lightning (see https://lightning.ai/docs/pytorch/stable/starter/installation.html)
conda install lightning=2.0.1 -c conda-forge

# Pip installs:
#  - PyTorch Geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 
#  - BioPython (see https://biopython.org/wiki/Download)
#  - Hydra (see https://hydra.cc/docs/intro/#installation)
python -m pip install torch-geometric biopython hydra-core -U
```

Alternatively, you can use the environment file `env/pippack_env.yaml` to build the environment:
```
# Build pippack environment from yaml file
conda env create -f env/pippack_env.yaml
```
## Data and Results
All test datasets (CASP13, CASP14, CASP15, CASP15+context, and Top2018) and prediction results are publicly available at https://zenodo.org/records/11236817.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find PIPPack useful in your research or project, please cite our paper:
```
@article{randolph2024pippack,
  title={Invariant point message passing for protein side chain packing},
  author={Randolph, Nicholas Z. and Kuhlman, Brian},
  journal={Proteins: Structure, Function, and Bioinformatics},
  year={2024},
  pages={1-14},
  doi={10.1002/prot.26705},
}
```
