# HTRU2 Pulsar Candidate Classification
## Machine Learning II – Unsupervised Learning Capstone

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Project Overview
This project applies three unsupervised clustering algorithms — 
**K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** — 
alongside **PCA** dimensionality reduction to the HTRU2 Pulsar 
dataset from the UCI Machine Learning Repository.

The goal is to discover natural groupings within pulsar candidate 
data and evaluate clustering quality using both internal and 
external metrics.

---


---

## Dataset
- **Name:** HTRU2 — Pulsar Candidate Classification
- **Source:** UCI Machine Learning Repository
- **Instances:** 17,898
- **Features:** 8 continuous numeric features
- **Target:** Binary (1 = Pulsar, 0 = Not Pulsar)
- **Download:** https://archive.ics.uci.edu/ml/datasets/HTRU2

> Do NOT commit the raw CSV file. See `data/README.md` for 
> download instructions.

---

## Project Structure
pulsar-candidate-classification/
│
├── data/
│   ├── HTRU_2.csv          # Dataset (not committed to GitHub)
│   └── README.md           # Download instructions
│
├── notebooks/
│   ├── member1_abdirahman/ # Abdirahman's workspace
│   ├── member2_kalik/      # Kalik's workspace
│   ├── member3_esther/     # Esther's workspace
│   └── final/              # Combined final notebook
│
├── plots/                  # All saved visualisations
├── reports/                # Final 5-page report PDF
├── src/                    # Shared Python functions
├── tests/                  # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
---

## Pipeline Steps
1. **Data Acquisition & EDA** — Load dataset, summary stats, 
   correlation heatmap, pair plots, class distribution
2. **Preprocessing** — StandardScaler (mean=0, std=1), 
   feature-target separation
3. **PCA** — Reduce to 2 principal components (78.48% variance)
4. **K-Means** — Elbow method + Silhouette tuning, k=2
5. **Hierarchical** — Dendrogram + linkage comparison, average linkage
6. **DBSCAN** — k-Distance graph + grid search, eps=1.7, min_samples=3
7. **Evaluation** — Silhouette, DBI, CHI, ARI, NMI
8. **Comparative Analysis** — All algorithms vs all metrics

---

## Results Summary
| Algorithm | Silhouette | DBI | ARI |
|---|---|---|---|
| K-Means (k=2) | 0.6010 | 1.0558 | **0.6066** |
| Hierarchical (avg) | **0.6750** | **0.5347** | 0.5203 |
| DBSCAN (eps=1.7) | 0.6760 | 0.3082 | 0.0008 |

**Best overall: K-Means** — highest ARI (0.6066), 
best recovery of true class structure.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Abdirahmanyussuf/pulsar-candidate-classification.git
cd pulsar-candidate-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
See `data/README.md` for instructions.

### 4. Open the notebook
```bash
jupyter lab
```
Navigate to `notebooks/member1_abdirahman/abdirahman_eda.ipynb`

---

## Required Libraries
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.10
jupyterlab>=4.0
---

## GitHub Repository
https://github.com/Abdirahmanyussuf/pulsar-candidate-classification

---

## References
1. R. J. Lyon et al. Fifty Years of Pulsar Candidate Selection. 
   Monthly Notices of the Royal Astronomical Society, 2016.
2. UCI ML Repository: HTRU2 Dataset. archive.ics.uci.edu
3. scikit-learn Documentation. scikit-learn.org
4. Ester et al. DBSCAN Algorithm. KDD-96, 1996.
5. Ward, J. H. Hierarchical Grouping. JASA, 1963.