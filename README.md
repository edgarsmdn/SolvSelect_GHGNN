# Solvent pre-selection for extractive distillation

This repo contains the code used in the paper Solvent pre-selection for extractive distillation using Gibbs-Helmholtz Graph Neural Networks submitted to [ESCAPE-33](https://escape33-ath.gr/). This work uses the version v1.0.0 of GH-GNN which is contained in its [own repo](https://github.com/edgarsmdn/GH-GNN).

To cite this work use:

```
Reference coming soon
```

To cite GH-GNN use:

```
@article{sanchez2022ghgnn,
  title={Gibbs-Helmholtz Graph Neural Network: capturing the temperature dependency of activity coefficients at infinite dilution},
  author={Sanchez Medina, Edgar Ivan and Linke, Steffen and Stoll, Martin and Sundmacher, Kai},
  journal={arXiv preprint arXiv:2212.01199},
  year={2022}
}
```

## How to use it?

Solvent preselection using the GH-GNN model can be easily carried out using the ```solvent_preselection``` class contained in the file  ```solvent_preselection.py```.

As described in the above mentioned paper, the selction can be performed by either:

* Relative volatility at infinite dilution
* Minimum solvent-to-feed ratio

Both methods are conviniently implemented as functions of the class ``` solvent_preselection```.

Case studies for the separation of aliphatic/aromatic and olefin/paraffin mixtures are provided on the other respective files.

#### Example

```
from solvent_preselection import solvent_preselection

mixture = {
        'c_i': {'smiles':'CCCCCC', 'name':'n-hexane'},
        'c_j': {'smiles':'c1ccccc1', 'name':'benzene'},
        'mixture_type': 'aliphatic_aromatic',
        'T_range': (25 + 273.15, 85 + 273.15),
        }

solvents = [...] # list of solvents SMILES
AD = ... # Applicability domain strategy to be applied to GH-GNN (either 'both' or None)

sp = solvent_preselection(mixture, solvents, AD)
sp.screen_with_rv()
sp.screen_with_minSF()

```

You could also give a star :star: to this repo!

### License

Note that this code has an [MIT license](https://github.com/edgarsmdn/SolvSelect_GHGNN/blob/main/LICENSE) that needs to be respected at all times

