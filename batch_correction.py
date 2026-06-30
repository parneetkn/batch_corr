"""
===============================================================================
GREIN + GTEx Integration and Batch Effect Correction using ComBat
===============================================================================

Author: Parneet
Date: June 2026

Description
-----------
This script integrates RNA-seq gene expression data from GREIN and GTEx,
performs batch effect correction using ComBat, and visualizes the effect of
batch correction using Principal Component Analysis (PCA).

The batch_corrflow harmonizes the two datasets by matching common gene symbols,
removing duplicate genes, applying log-transformation, selecting highly
variable genes, correcting for dataset-specific batch effects, and comparing
sample clustering before and after batch correction.

batch_corrflow
--------
1. Load GREIN gene expression matrix and sample metadata.
2. Load GTEx whole blood gene expression data (GCT format).
3. Remove duplicate gene symbols by averaging expression values.
4. Identify common genes between GREIN and GTEx.
5. Apply log2(x + 1) transformation.
6. Merge both datasets into a unified expression matrix.
7. Select the top 2,000 most variable genes.
8. Perform PCA before batch correction.
9. Correct batch effects using ComBat (neuroCombat).
10. Perform PCA after batch correction.
11. Generate side-by-side PCA comparison plots.
12. Save corrected datasets and figures.

Inputs
------
- GREIN gene expression CSV for study GSE112057
- GREIN sample metadata CSV for study GSE112057
- GTEx GCT v11 expression file for whole blood

Outputs
-------
output/
│
├── Combined_Before_ComBat.csv
│     Expression matrix prior to batch correction.
│
├── Combined_ComBat_Corrected.csv
│     Batch-corrected expression matrix.
│
└── PCA_Before_After_ComBat.png
      Side-by-side PCA plots before and after ComBat correction.

Dependencies
------------
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- neuroCombat

Usage
-----
Update the input file paths in the "INPUT FILES" section and execute:

    python GREIN_GTEx_ComBat_PCA_Comparison.py

Notes
-----
- Gene symbols are used as the common feature identifiers.
- Duplicate gene symbols are collapsed by averaging expression values.
- PCA is performed on standardized expression values.
- ComBat removes technical variation associated with dataset origin (GREIN vs GTEx)
  while preserving biological variation.
- The side-by-side PCA comparison provides a visual assessment of batch correction
  effectiveness.

===============================================================================
"""

import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from neuroCombat import neuroCombat

# =====================================================
# OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# INPUT FILES
# =====================================================

grein_expr_file = "/Users/parneet/Desktop/ICGEB/batch_corr/Data/GSE112057_GeneLevel_Raw_data.csv"
grein_meta_file = "/Users/parneet/Desktop/ICGEB/batch_corr/Data/GSE112057_full_metadata.csv"
gtex_file = "/Users/parneet/Desktop/ICGEB/batch_corr/Data/gene_reads_adult_gtex_v11_whole_blood.gct"

# =====================================================
# LOAD GREIN METADATA
# =====================================================

print("\nLoading GREIN metadata...")

meta = pd.read_csv(grein_meta_file)

sample_col = "geo_accession"
disease_col = "characteristics_ch1.3"

print(meta[disease_col].value_counts())

meta[disease_col] = meta[disease_col].str.replace(
    "disease state (diagnosis): ", "", regex=False
)

meta["Group"] = np.where(
    meta[disease_col] == "Control",
    "Control_GREIN",
    "Disease_GREIN"
)

print(meta["Group"].value_counts())

# =====================================================
# LOAD GREIN EXPRESSION
# =====================================================

print("\nLoading GREIN expression matrix...")

grein = pd.read_csv(grein_expr_file)

grein = grein.drop(columns=grein.columns[0])
grein = grein.set_index("gene_symbol")
grein = grein.T

print("GREIN samples x genes:", grein.shape)

# =====================================================
# MATCH GREIN SAMPLES
# =====================================================

common_samples = grein.index.intersection(meta[sample_col])

grein = grein.loc[common_samples]
meta = meta.set_index(sample_col).loc[common_samples]

print("Matched GREIN samples:", len(common_samples))

# =====================================================
# REMOVE DUPLICATE GREIN GENES
# =====================================================

print("\nChecking GREIN duplicates...")

print("Duplicate GREIN gene symbols:", grein.columns.duplicated().sum())

grein = grein.T.groupby(level=0).mean().T
grein = grein.loc[:, grein.columns.notna()]
grein = grein.loc[:, grein.columns != ""]

print("GREIN after cleanup:", grein.shape)

# =====================================================
# LOAD GTEx
# =====================================================

print("\nLoading GTEx GCT...")

gtex = pd.read_csv(
    gtex_file,
    sep="\t",
    skiprows=2
)

print("Raw GTEx shape:", gtex.shape)

gtex = gtex.dropna(subset=["Description"])
gtex = gtex.set_index("Description")
gtex = gtex.drop(columns=["Name"])
gtex = gtex.T

print("GTEx samples x genes:", gtex.shape)

# =====================================================
# REMOVE DUPLICATE GTEx GENES
# =====================================================

print("\nChecking GTEx duplicates...")

print("Duplicate GTEx gene symbols:", gtex.columns.duplicated().sum())

gtex = gtex.T.groupby(level=0).mean().T
gtex = gtex.loc[:, gtex.columns.notna()]
gtex = gtex.loc[:, gtex.columns != ""]

print("GTEx after cleanup:", gtex.shape)

# =====================================================
# COMMON GENES
# =====================================================

common_genes = grein.columns.intersection(gtex.columns)

print("\nCommon genes:", len(common_genes))

grein = grein[common_genes]
gtex = gtex[common_genes]

# =====================================================
# LOG2
# =====================================================

print("\nLog2 transforming...")

grein = np.log2(grein.astype(float) + 1)
gtex = np.log2(gtex.astype(float) + 1)

# =====================================================
# COMBINE
# =====================================================

combined = pd.concat([grein, gtex], axis=0)

labels = (
    meta["Group"].tolist()
    + ["Healthy_GTEx"] * len(gtex)
)

batch_labels = (
    ["GREIN"] * len(grein)
    + ["GTEx"] * len(gtex)
)

print("\nCombined matrix:", combined.shape)

# =====================================================
# TOP VARIABLE GENES
# =====================================================

print("\nSelecting top variable genes...")

gene_var = combined.var(axis=0)
top_genes = gene_var.nlargest(2000).index
combined = combined[top_genes]

combined.to_csv(os.path.join(
    OUTPUT_DIR,
    "Combined_Before_ComBat.csv"
))

print("Using genes:", len(top_genes))

# =====================================================
# PCA BEFORE COMBAT
# =====================================================

print("\nRunning PCA BEFORE ComBat...")

X_before = StandardScaler().fit_transform(combined)

pca_before = PCA(n_components=2)

pcs_before = pca_before.fit_transform(X_before)

pc1_before = pca_before.explained_variance_ratio_[0] * 100
pc2_before = pca_before.explained_variance_ratio_[1] * 100

plot_before = pd.DataFrame({
    "PC1": pcs_before[:, 0],
    "PC2": pcs_before[:, 1],
    "Group": labels
})

print(f"PC1 = {pc1_before:.2f}%")
print(f"PC2 = {pc2_before:.2f}%")

# =====================================================
# COMBAT
# =====================================================

print("\nRunning ComBat...")

combat_input = combined.T

batch_df = pd.DataFrame({
    "batch": batch_labels
})

combat_result = neuroCombat(
    dat=combat_input,
    covars=batch_df,
    batch_col="batch"
)

combat_corrected = pd.DataFrame(
    combat_result["data"],
    index=combat_input.index,
    columns=combat_input.columns
)

combined_corrected = combat_corrected.T

combined_corrected.to_csv(os.path.join(
    OUTPUT_DIR,
    "Combined_ComBat_Corrected.csv"
))

# =====================================================
# PCA AFTER COMBAT
# =====================================================

print("\nRunning PCA AFTER ComBat...")

X = StandardScaler().fit_transform(combined_corrected)

pca = PCA(n_components=2)

pcs = pca.fit_transform(X)

pc1 = pca.explained_variance_ratio_[0] * 100
pc2 = pca.explained_variance_ratio_[1] * 100

plot_after = pd.DataFrame({
    "PC1": pcs[:, 0],
    "PC2": pcs[:, 1],
    "Group": labels
})

print(f"PC1 = {pc1:.2f}%")
print(f"PC2 = {pc2:.2f}%")

# =====================================================
# SIDE-BY-SIDE PCA
# =====================================================

palette = {
    "Healthy_GTEx": "green",
    "Control_GREIN": "blue",
    "Disease_GREIN": "red"
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.scatterplot(
    data=plot_before,
    x="PC1",
    y="PC2",
    hue="Group",
    palette=palette,
    s=65,
    alpha=0.75,
    ax=axes[0]
)

axes[0].set_title("Before ComBat")
axes[0].set_xlabel(f"PC1 ({pc1_before:.2f}%)")
axes[0].set_ylabel(f"PC2 ({pc2_before:.2f}%)")
axes[0].legend().remove()

sns.scatterplot(
    data=plot_after,
    x="PC1",
    y="PC2",
    hue="Group",
    palette=palette,
    s=65,
    alpha=0.75,
    ax=axes[1]
)

axes[1].set_title("After ComBat")
axes[1].set_xlabel(f"PC1 ({pc1:.2f}%)")
axes[1].set_ylabel(f"PC2 ({pc2:.2f}%)")
axes[1].legend(
    title="Group",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.suptitle(
    "PCA Before and After ComBat",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "PCA_Before_After_ComBat.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFinished successfully.")
print("Outputs written to:", OUTPUT_DIR)
