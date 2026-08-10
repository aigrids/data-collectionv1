# The AI.grids v1 collection of standardized machine learning tasks and datasets for enhancing renewable energy

Innovative research and development on machine learning (ML) for electric power systems remains limited because the field lacks standardized benchmarking tasks and datasets. To address this gap, we introduce AI.grids v1, a standardized collection of real-world and synthetic datasets for a variety ML applications in renewable power systems. Leveraging the unified structure of the collected datasets, we introduce a suite of task- and dataset-characterization metrics that quantify and visualize key properties. We further propose a set of general-purpose and task-specific performance evaluation metrics that provide richer insights into the performance of models than commonly used ML metrics. For each task, we benchmark one ML model which serves as a simple, reproducible baseline against which newly proposed models can be rigorously compared.

## Citation

Aryandoust, A. The AI.grids v1 collection of standardized machine learning tasks and datasets for enhancing renewable energy. Preprint on Arxiv (2026).

```bibtex
@article{aryandoust2026aigridsv1,
  title={The AI.grids v1 collection of standardized machine learning tasks and datasets for enhancing renewable energy},
  author={Aryandoust, Arsam and Monti, Antonello},
  journal={Preprint on Arxiv}
}
```

## Requirements

- Python version >= 3.12
- CUDA
- ~ 10 TB storage capacity (for all datasets)


## Overview

1. [Quick start](#1-quick-start)
2. [Run analyses and experiments](#2-run-analyses-and-experiments)
3. [Contributing](#3-contributing)


## 1. Quick start

Clone the repository:
```bash
git clone https://github.com/aigrids/data-collectionv1

cd data-collectionv1
```

(Optional) Create virtual enviornment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate 

# alternatively, using conda:
conda create -n collectionv1-venv python=3.12
conda activate collectionv1-venv
```

Install requirements:
```bash
pip install -r requirements.txt
```

Install source code of repository in editbale mode. This makes src/** visible to notebooks and all other entry points, without having to change sys.path:
```bash
pip install -e .
```

Start Jupyter notebook:

```bash
python3.12 -m notebook
```

## 2. Run analyses and experiments

Run analysis for a specific task, for example for BuildingElectricity. Before doing so, check the config.yml file and set for example `root_path_datasets` to a sufficiently large partition:

```bash
python3.12 scripts/analyse.py BE
```

Available arguments for each task are:
- BE (BuildingElectricity)
- WF (WindFarm)
- SC (SolarCube)
- PG (PowerGraph)
- OD (OPFData)

This will save dataset characteristics analysis results in on the path `results/analysis/`

## 3. Contributing

Project collaborators, please read [CONTRIBUTING.md](CONTRIBUTING.md) before 
opening an issue or pull request.