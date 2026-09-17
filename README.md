# RoBERTa Toxic Comment Classification with Reply-Based Features

## Overview

This project investigates if and how 

This project was completed as part of an NSF-funded Research Experience for Undergraduates (REU), Award #2447577, focused on safety and health in digital environments.

## Approach

- **Model**: RoBERTa-based classifier with sentiment trajectory features (VADER + Cardiff RoBERTa)
- **Experimental design**: Four-condition context ablation (before / after / both / none)
- **Datasets**: CAD (Conversational Abuse Dataset) and Conversations Gone Awry
- **Core contribution**: [describe your novelty claim here, e.g. "Use of discrete emotion trajectories as classifier input for implicit toxicity detection — a gap identified through systematic literature review"]

## Repository Structure

```
[FILL IN — e.g.]
├── roberta_common/          # Shared modules for feature configs
├── scripts/                 # Driver scripts (meta_text, sentiment_score, etc.)
├── data_analysis/           # 16-chart EDA pipeline (toxicity, sentiment, thread depth, etc.)
├── hpc/                     # SLURM job scripts / cluster configs
└── README.md
```

## Setup / Installation

[FILL IN — e.g. Python version, key dependencies (transformers, torch, pandas, vaderSentiment, etc.), how to install]

```bash
pip install -r requirements.txt
```

## Usage

[FILL IN — how to run the driver scripts / reproduce experiments]

```bash
[example command]
```

## HPC / Cluster Notes

Experiments were run on `hpcgpu1` (partition `gpuq`). [Add any notes future users/collaborators would need — GPU selection, HuggingFace offline mode setup, SLURM job submission format, etc., if you want this documented for others to reproduce.]

## Results

[FILL IN — key findings, metrics (F1, accuracy, etc.), and any figures. Since the project is complete, this is a good place to summarize your final results/conclusions.]

## References

Key papers that grounded this work:

- Almerekhi et al., "PROVOKE: Toxicity Trigger Detection in Conversations," 2022
- Zhang et al., 2018
- Brassard-Gourdeau & Khoury, 2020
- Pavlopoulos et al., 2020
- PONOS (arXiv:2503.16072)

[Add full citations / BibTeX if desired]

## Acknowledgments

This work was supported by the National Science Foundation under Award #2447577. Thank you to my mentors, Dr. Artan, Dr. Dong, and Dr. Gu, for their guidance throughout this project.

## License

[FILL IN — e.g. MIT, or "For academic use only," or leave blank if not decided]
