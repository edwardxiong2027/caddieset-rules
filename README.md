# CaddieSet Rules — From Prediction to Prescription

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red.svg)](paper/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/site-live-brightgreen)](https://edwardxiong2027.github.io/caddieset-rules/)

A secondary analysis of the **CaddieSet** golf-swing dataset
([Jung et al., 2025](https://arxiv.org/abs/2508.20491)).
Where the dataset paper focuses on *predicting* ball trajectories, this project shifts the lens to **prescription**:
which body-posture features, at which swing phases, are most strongly associated with long drives — and can we
compress those signals into a human-readable coaching rule tree?

> **TL;DR.** Across 1,757 swings from 8 golfers, the 83-yard gap between top-20% and bottom-20% driver distances is
> concentrated in just four body-posture signals: **lead-arm extension at mid-backswing**, **lateral hip shift
> through the downswing**, **quiet hips during the backswing**, and **preserved trail-leg flex at the top**. These
> are captured by a depth-3 decision tree with cross-validated R² = 0.34 — simple enough to use as coaching cues.

---

## Paper

- **Title:** *From Prediction to Prescription: Mining Interpretable Biomechanical Rules and Latent Swing Archetypes from Multi-Phase Pose Data*
- **PDF:** [`paper/paper.pdf`](paper/paper.pdf)
- **LaTeX source:** [`paper/paper.tex`](paper/paper.tex), [`paper/references.bib`](paper/references.bib)
- **Figures (PDF + PNG):** [`paper/figures/`](paper/figures/)

### Key findings

| Finding | Effect size |
|---|---|
| Long hitters keep lead arm ~19° straighter at mid-backswing (168° vs 149°) | Cohen's d = +1.19 |
| Long hitters shift hips target-ward in early downswing (+0.02 vs –0.25) | Cohen's d = +1.36 |
| Long hitters keep hip rotation quiet in backswing (~11° vs 43°) | Cohen's d = –0.64 |
| Long hitters keep trail leg flexed at the top (166° vs 178°, DTL view) | Cohen's d = –1.40 |
| Pose-only R² is 0.50 for distance but only 0.19 for direction accuracy | — |
| 4 interpretable swing archetypes discovered via UMAP + K-means (silhouette 0.54) | — |

## Interactive site

A landing page summarising the paper and findings is deployed via GitHub Pages at
<https://edwardxiong2027.github.io/caddieset-rules/>.

---

## Repository layout

```
caddieset-rules/
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── requirements.txt
├── .gitignore
├── paper/
│   ├── paper.tex             ← LaTeX source
│   ├── paper.pdf             ← compiled PDF (11 pages)
│   ├── references.bib        ← 19 citations
│   └── figures/              ← 6 figures (PDF + PNG)
├── src/
│   ├── 01_explore.py         ← dataset profiling
│   ├── 02_effect_sizes.py    ← Cohen's d per phase/feature
│   ├── 03_archetypes.py      ← UMAP + K-means clustering
│   ├── 04_rules.py           ← depth-3 decision tree
│   └── 05_make_figures.py    ← reproduces all 6 figures
├── data/
│   └── README.md             ← pointer to CaddieSet
└── docs/                     ← GitHub Pages site
    ├── index.html
    └── assets/
```

## Reproducing results

```bash
# 1. Clone and enter
git clone https://github.com/edwardxiong2027/caddieset-rules.git
cd caddieset-rules

# 2. Install (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Place CaddieSet.csv in data/ (see data/README.md for source)
cp /path/to/CaddieSet.csv data/

# 4. Run the pipeline
python src/01_explore.py
python src/02_effect_sizes.py
python src/03_archetypes.py
python src/04_rules.py
python src/05_make_figures.py

# 5. Re-compile the paper (optional)
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Seeds are fixed (42) throughout; rerunning the pipeline reproduces every figure and table in the paper exactly.

## Citing

If you use this code or the findings in this paper, please cite both this repository and the original CaddieSet
paper:

```bibtex
@misc{caddiesetrules2026,
  title  = {From Prediction to Prescription: Mining Interpretable Biomechanical Rules and Latent Swing Archetypes from Multi-Phase Pose Data},
  author = {Xiong, Edward},
  year   = {2026},
  url    = {https://github.com/edwardxiong2027/caddieset-rules}
}

@article{jung2025caddieset,
  title   = {CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information},
  author  = {Jung, Seunghyeon and Hong, Seoyoung and Jeong, Jiwoo and Jeong, Seungwon and Choi, Jaerim and Kim, Hoki and Lee, Woojin},
  journal = {arXiv preprint arXiv:2508.20491},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.20491}
}
```

See [`paper/references.bib`](paper/references.bib) for the full list of cited works (pose estimation, interpretable
ML, and golf-biomechanics literature).

## Data availability

The CaddieSet dataset itself is **not redistributed** in this repository. Please obtain it from the original authors
via the arXiv paper ([2508.20491](https://arxiv.org/abs/2508.20491)). See [`data/README.md`](data/README.md) for
details.

## License

Code and figures in this repository are released under the [MIT License](LICENSE). The paper text and manuscript
PDF are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to share and adapt with
attribution.

The underlying CaddieSet dataset is the property of its original authors and is not re-licensed here.

## Acknowledgements

This work would not exist without the public release of CaddieSet by
[Jung et al., 2025](https://arxiv.org/abs/2508.20491). We also thank the authors of `scikit-learn`, `umap-learn`,
`matplotlib`, and LaTeX for the open tooling that makes this kind of secondary analysis possible.
