# CaddieSet Rules — Phase-localised pose correlates of driver distance (v2, revised per peer review)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red.svg)](paper/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/site-live-brightgreen)](https://edwardxiong2027.github.io/caddieset-rules/)

A secondary analysis of the **CaddieSet** golf-swing dataset
([Jung et al., 2025](https://arxiv.org/abs/2508.20491)).
This project asks which body-posture features, at which swing phases, are associated with driver distance, ball
speed, and directional accuracy — and, critically, how much apparent predictive power is real biomechanics
versus memorised golfer identity in a corpus of **only N = 8 golfers**.

> **TL;DR (v2).** Under random 5-fold CV, pose-only Random Forests reach R² ≈ 0.50 for driver distance. Under
> **GroupKFold(8)** grouped by golfer, the same ceiling collapses to R² ≈ −4.36. Across 252 feature × phase
> × outcome tests, 98 survive Benjamini–Hochberg FDR q < 0.05. Extreme-groups Cohen's *d* up to 1.36
> back-transforms (Fitzsimons 2008) to a continuous Pearson *r* of ≈ 0.45. Four-cluster K-means in pose space
> has silhouette 0.54 and mean LOGO ARI 0.956, but two of four clusters are 82–89% a single golfer. The
> paper reads as **descriptive / associative**, not prescriptive.

---

## What changed in v2

This release addresses a peer-review report (Major Revision) with the following changes:

1. **GroupKFold(8) grouped by golfer everywhere.** Every R² in the paper is now reported under both random
   5-fold and golfer-aware CV. The gap is itself the headline finding.
2. **Benjamini–Hochberg FDR correction** across all 252 (feature × phase × outcome) tests.
3. **Continuous-r back-transform.** Extreme-groups *d* is reported alongside point-biserial *r*, the
   Fitzsimons continuous-r back-transform, full-sample Pearson *r*, and Spearman *ρ*.
4. **Archetype validation.** K sweep with permuted-label null baseline, per-golfer × per-cluster crosstab,
   entropy over golfers per cluster, and leave-one-golfer-out Adjusted Rand Index.
5. **No feature-selection leak** in the regression-tree CV; depth sweep + per-leaf 95% bootstrap CIs.
6. **Accuracy-ceiling robustness grid:** three target transforms (signed, |·|, √|·|) × two models (RF, HGB)
   × two CV schemes.
7. **Retitle and reframe.** The paper and website no longer use "prescription" language for a corpus that
   doesn't cross golfer boundaries.

## Paper

- **Title:** *Phase-Localised Pose Correlates of Driver Distance in a Small-N Golf Corpus: A Secondary, Associative Analysis of CaddieSet*
- **PDF:** [`paper/paper.pdf`](paper/paper.pdf) (14 pages)
- **LaTeX source:** [`paper/paper.tex`](paper/paper.tex), [`paper/references.bib`](paper/references.bib)
- **Figures (PDF + PNG):** [`paper/figures/`](paper/figures/) (8 figures)

### Headline numbers

| Claim | Random 5-fold | GroupKFold(8) |
|---|---|---|
| Pose-only R² for Distance (Face-On) | +0.50 | **−4.36** |
| Pose-only R² for Ball Speed (Face-On) | +0.56 | **−4.67** |
| Pose-only R² for \|Direction error\| (Face-On) | +0.19 | −0.49 |
| Depth-3 tree on Distance | +0.34 | **−3.36** |

| Association | Extreme-groups *d* | Continuous *r* (Fitzsimons) | BH-FDR q |
|---|---|---|---|
| Early-downswing hip shift target-ward | +1.36 | +0.45 | <10⁻⁴ |
| Mid-backswing lead-arm angle (straighter = longer) | +1.19 | +0.39 | <10⁻⁴ |
| Trail-leg flex at top of backswing (DTL, more flex = longer) | −1.40 | −0.46 | <10⁻⁴ |
| Mid-backswing hip rotation (quieter = longer) | −0.64 | −0.19 | <10⁻³ |

98 of 252 (view × outcome × feature) tests survive BH-FDR at q < 0.05.

## Interactive site

A revised landing page summarising the paper is deployed via GitHub Pages at
<https://edwardxiong2027.github.io/caddieset-rules/>.

---

## Repository layout

```
caddieset-rules/
├── README.md
├── LICENSE
├── requirements.txt
├── paper/
│   ├── paper.tex
│   ├── paper.pdf                 ← compiled PDF (14 pages)
│   ├── references.bib            ← 27 citations
│   └── figures/                  ← 8 figures (PDF + PNG)
├── src/
│   ├── 01_explore.py             ← dataset profiling
│   ├── 02_effect_sizes.py        ← Cohen's d + Pearson r + BH-FDR + GroupKFold ceilings
│   ├── 03_archetypes.py          ← K sweep, per-golfer crosstab, LOGO ARI
│   ├── 04_rules.py               ← tree w/o feature-selection leak, depth sweep, leaf CIs
│   ├── 06_accuracy_robust.py     ← target-transform × model × CV-scheme robustness grid
│   └── 05_make_figures.py        ← reproduces all 8 figures
├── artifacts/                    ← every CSV + pickle + JSON produced by the pipeline
├── data/
│   └── README.md                 ← pointer to CaddieSet
└── docs/                         ← GitHub Pages site
    ├── index.html
    └── assets/
```

## Reproducing results

```bash
git clone https://github.com/edwardxiong2027/caddieset-rules.git
cd caddieset-rules
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place CaddieSet.csv in data/ (see data/README.md)
cp /path/to/CaddieSet.csv data/

python src/01_explore.py
python src/02_effect_sizes.py      # effect sizes + FDR + GroupKFold ceilings
python src/03_archetypes.py        # cluster validation
python src/04_rules.py             # tree surrogate
python src/06_accuracy_robust.py   # robustness grid
python src/05_make_figures.py      # 8 figures

# Re-compile the paper (optional)
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Seeds are fixed (42) throughout; rerunning the pipeline reproduces every figure and table in the paper exactly.

## Citing

```bibtex
@misc{caddiesetrules2026,
  title  = {Phase-Localised Pose Correlates of Driver Distance in a Small-N Golf Corpus:
            A Secondary, Associative Analysis of CaddieSet},
  author = {Xiong, Edward},
  year   = {2026},
  url    = {https://github.com/edwardxiong2027/caddieset-rules}
}

@article{jung2025caddieset,
  title   = {CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information},
  author  = {Jung, Seunghyeon and Hong, Seoyoung and Jeong, Jiwoo and
             Jeong, Seungwon and Choi, Jaerim and Kim, Hoki and Lee, Woojin},
  journal = {arXiv preprint arXiv:2508.20491},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.20491}
}
```

## Data availability

The CaddieSet dataset itself is **not redistributed** here. Obtain it from the original authors
via the arXiv paper ([2508.20491](https://arxiv.org/abs/2508.20491)). See [`data/README.md`](data/README.md)
for details.

## License

Code and figures: [MIT License](LICENSE). Paper text and manuscript PDF:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying CaddieSet dataset is the property of
its original authors and is not re-licensed here.

## Acknowledgements

This work would not exist without the public release of CaddieSet by
[Jung et al., 2025](https://arxiv.org/abs/2508.20491). We also thank the anonymous peer reviewer whose
detailed critique prompted every major change in v2.
