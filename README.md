# KG-DQN

Preliminary release of the code from the paper "Playing Text-Adventure Games with Graph-Based Deep Reinforcement Learning
", Prithviraj Ammanabrolu and Mark O. Riedl, NAACL-HLT 2019, Minneapolis, MN - https://arxiv.org/abs/1812.01628

Disclaimer: Code is not upkept

## Data/Pre-training
- Games are created using [Textworld's](https://github.com/Microsoft/TextWorld) `tw-make` as specified in the paper.
- Pre-training is done using [DrQA](https://github.com/facebookresearch/DrQA) by generating traces using the WalkthroughAgent in Textworld.

## Running the code
- Code is run using an [Anaconda](https://www.anaconda.com/download/#linux "Anaconda 2") environment for Python 3.6. The environment is defined in **env.yml**. Run `conda env create -f env.yml` and then `source activate kgdqn` to enter the correct environment.
- Baseline BOW-DQN implementation is in `dqn/`
- KG-DQN implementation is in `kgdqn/`, run using `python train.py` after defining the required parameters and game in `train.py`
- `w2id.txt` and `id2act.txt` are required and are the dictionaries for the vocab and full action set for the specific game
- Similarly, `entity2id.txt`/`relation2id.txt` defines the entities and relations that can be extracted by OpenIE for the game in a dictionary format 