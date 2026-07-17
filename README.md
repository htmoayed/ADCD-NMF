# ADCD-NMF: Attribute-Driven Community Detection in Attributed Networks via Non-negative Matrix Factorization

Official implementation of the paper

> **Attribute-Driven Community Detection in Attributed Networks via Non-negative Matrix Factorization (ADCD-NMF)**

---

# Overview

ADCD-NMF is a novel Non-negative Matrix Factorization (NMF) framework for community detection in attributed networks. The proposed method jointly exploits structural connectivity and node attributes through two complementary attribute-driven matrix factorization models together with dual graph manifold regularization.

The framework consists of

- Attribute-Driven Structural Community Factorization (SCF)
- Attribute-Driven Attribute Community Factorization (ACF)
- Dual structural and attribute graph regularization
- Alternating multiplicative optimization
- Interpretable community representations

Extensive experiments on nine benchmark attributed networks demonstrate the effectiveness of the proposed method.

---


# Requirements

The implementation has been tested using

- Python 3.11
- NumPy
- PyTorch
- PyTorch Geometric

Install all required packages using

```bash
pip install -r requirements.txt
```


---


# Datasets

The experiments in this work are conducted on nine publicly available attributed network datasets: **Cornell, Wiki, Cora, CiteSeer, Facebook, DBLP, BlogCatalog, Flickr,** and **PubMed**.

Most of these datasets can be conveniently loaded using the **PyTorch Geometric** library through the `AttributedGraphDataset` interface:

https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.datasets.AttributedGraphDataset.html

Alternatively, the datasets can be downloaded from their corresponding public repositories. The **Facebook** dataset is available from the Stanford SNAP repository:

https://snap.stanford.edu/data/ego-Facebook.html

The **Cora** dataset can be downloaded from:

https://graphsandnetworks.com/the-cora-dataset/

The **CiteSeer** dataset is available at:

https://networkrepository.com/citeseer.php

The **PubMed** dataset can be obtained from:

https://linqs.org/datasets/#pubmed-diabetes

The remaining datasets (**Wiki, DBLP, BlogCatalog, Flickr**, and several other benchmark attributed graphs) are available from the following graph dataset repositories:

https://renchi.ac.cn/datasets/

or

https://github.com/HKBU-LAGAS/Awesome-Graph-Datasets


---

# Hyperparameter Configuration

The regularization parameters are selected using grid search.

Search space

```
{10^-3, 10^-2, 10^-1, 1, 10, 100, 1000}
```

The number of communities for each dataset is fixed according to the corresponding ground-truth labels.

The optimization terminates after reaching the maximum number of iterations or satisfying the convergence criterion.

---

# Evaluation Protocol

The proposed method is evaluated using

- Accuracy (ACC)
- Normalized Mutual Information (NMI)
- Modularity (Q)

To reduce the influence of random initialization, every experiment is independently repeated **15 times**, and the reported results correspond to the average performance.

---

# Reproducibility

To facilitate reproducibility, this repository provides

- Complete implementation of ADCD-NMF
- Dependency information
- Scripts for reproducing all experiments
- Hyperparameter search configuration
- Evaluation scripts
- Random initialization protocol


All experiments reported in the paper can be reproduced using the provided code and the publicly available datasets.

---


# Citation

If you use this repository in your research, please cite

```bibtex
@article{Moayed2026,
  title={Attribute-Driven Community Detection in Attributed Networks via Non-negative Matrix Factorization},
  author={Moayed, Hojjat and Saberi-Movahed, Farid and Berahmand, Kamal and Moradi, Parham},
  journal={Pattern Recognition},
  year={2026}
}
```

---

# Contact

**Hojjat Moayed**

Department of Computer Engineering

Esfarayen University of Technology

Email: **moayed@esfarayen.ac.ir**
