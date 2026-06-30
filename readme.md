## GREIN–GTEx Batch Effect Correction using ComBat

### Overview

This script integrates bulk RNA-seq gene expression datasets from **GREIN** and **GTEx**, performs **batch effect correction** using **ComBat (neuroCombat)**, and evaluates the correction by comparing **PCA plots before and after batch correction**.

The workflow harmonizes both datasets by matching common genes, applying preprocessing, correcting for dataset-specific technical variation, and generating corrected expression matrices for downstream analyses.

---

### Workflow

1. Load GREIN gene expression matrix and sample metadata.
2. Load GTEx gene expression data (GCT format).
3. Remove duplicate gene symbols by averaging expression values.
4. Identify common genes between GREIN and GTEx.
5. Apply log₂(x + 1) transformation.
6. Merge the datasets into a single expression matrix.
7. Select the top 2,000 most variable genes.
8. Perform PCA before batch correction.
9. Correct batch effects using ComBat (neuroCombat).
10. Perform PCA after batch correction.
11. Save corrected expression matrix and PCA comparison plots.

---

### Input Files

- GREIN gene expression matrix (.csv)
- GREIN sample metadata (.csv)
- GTEx gene expression matrix (.gct)

Update the file paths in the **INPUT FILES** section before running the script.

---

## #Output Files

```
output/
│
├── Combined_Before_ComBat.csv
│     Combined expression matrix before batch correction
│
├── Combined_ComBat_Corrected.csv
│     Batch-corrected expression matrix
│
└── PCA_Before_After_ComBat.png
      PCA comparison before and after ComBat
```

---

### Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- neuroCombat

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn neuroCombat
```

---

### Running the Script

```bash
python batch_correction.py
```

---

### Notes

- Batch labels correspond to dataset origin (**GREIN** and **GTEx**).
- PCA is performed on standardized expression values.
- ComBat removes technical variation associated with sequencing batches while preserving overall biological structure.
- The PCA plots provide a visual assessment of batch correction effectiveness.
- The corrected expression matrix can be used for downstream analyses such as differential expression, clustering, machine learning, and biomarker discovery.