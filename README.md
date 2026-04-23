# TTPI: Generalized Policy Iteration using Tensor Train

A PyTorch implementation of TTPI algorithm 

Website: 

Paper: 

### Pre-requistes
- Install Pytorch (Cuda compatible for GPU acceleration)
- Install celluloid (for plotting and animation)
- Install the *tntorch* library for TT-Cross (*pip install tntorch*) (https://tntorch.readthedocs.io/en/latest/)
    - This requires you to have the *maxvolpy* package (*pip install maxvolpy*)


### Overview
- *ttpi.py*: the TTPI algorithm is defined in this class
- *dynamic_systems.py* : This has forward dynamics for various systems
- Notebook *tt_vs_nn.ipynb* demonstrates the comparision of TT vs NN for function approximation
- Example notebooks are:
  - *PointMass.ipynb*: velocity or acceleration controlled point-mass reaching task (with obstacle avoidance)
  - *SinglePendulum.ipynb*: Pendulum swingup
  - *Pushing.ipynb*: Object pushing
  - *Pivoting.ipynb* : Object Pivoting task
Note: All the implementations are fully compatible for use with GPU. For faster computation, it is highly recommended to use GPU
