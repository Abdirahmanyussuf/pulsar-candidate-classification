HTRU2 Pulsar Candidate Classification
Machine Learning II – Unsupervised Learning Capstone
Date: May 2026
GitHub: github.com/Abdirahmanyussuf/pulsar-candidate-classification

1. Introduction & Dataset
The HTRU2 dataset contains 17,898 pulsar candidate observations collected during the High Time Resolution Universe Survey (HTRU). Pulsars are rare rotating neutron stars that emit detectable radio waves. Identifying real pulsars among radio frequency interference is a critical challenge in modern astrophysics. This project applies three unsupervised clustering algorithms — K-Means, Hierarchical (Agglomerative), and DBSCAN — alongside Principal Component Analysis (PCA) to discover natural groupings within the data without using class labels during training.
1.1 Problem Statement
Manual identification of pulsar candidates is time-consuming and resource-intensive. The volume of data produced by radio telescopes makes automated methods essential. An unsupervised clustering pipeline can reveal hidden structure in the data, assist astronomers in prioritising candidates for follow-up observation, and provide a reproducible methodology that can scale to larger surveys.
1.2 Dataset Description
The HTRU2 dataset is sourced from the UCI Machine Learning Repository. Each of the 17,898 instances represents a pulsar candidate described by 8 continuous numeric features derived from two signal profiles: the integrated pulse profile and the DM-SNR curve.
FeatureDescriptionmean_integratedMean of the integrated pulse profilestd_integratedStandard deviation of the integrated profileexcess_kurtosisExcess kurtosis of the integrated profileskewnessSkewness of the integrated profilemean_dm_snrMean of the DM-SNR curvestd_dm_snrStandard deviation of the DM-SNR curveexcess_kurtosis_dmExcess kurtosis of the DM-SNR curveskewness_dmSkewness of the DM-SNR curve
1.3 Dataset Justification

All 8 features are continuous and numeric — ideal for distance-based clustering algorithms
Zero missing values across all 17,898 rows — no imputation required
Ground truth labels available (1 = Pulsar, 0 = Not Pulsar) — enables external metric evaluation (ARI, NMI)
Significant class imbalance: 90.8% Not Pulsar vs 9.2% Pulsar — tests algorithm robustness
Feature scales vary by up to 100x (std range: 1.06 to 106.51) — confirms need for StandardScaler


2. Methodology & Pipeline
The project follows a modular, reproducible pipeline implemented in a Jupyter Notebook using scikit-learn. The pipeline consists of six stages: data acquisition, EDA, preprocessing, PCA, clustering, and evaluation.
2.1 Preprocessing
Two preprocessing steps were applied before clustering:

Feature-Target Separation: The target column was separated from the 8 feature columns, producing a feature matrix X of shape (17,898 x 8).
StandardScaler: All 8 features were standardised to mean=0 and std=1. This was critical because skewness_dm had a standard deviation of 106.51 while excess_kurtosis had only 1.06 — a 100x scale difference that would severely distort distance-based algorithms without scaling.

2.2 Dimensionality Reduction — PCA
Principal Component Analysis (PCA) was applied to the scaled features to reduce dimensionality and enable 2D visualisation of clusters.
ComponentVariance ExplainedCumulativePC151.68%51.68%PC226.81%78.48%PC310.12%88.60%PC45.72%94.32%PC5–PC85.68%100.00%
The scree plot revealed a clear elbow at PC3, confirming that 2 principal components (78.48% variance) are sufficient for visualisation. All cluster scatter plots are rendered in this 2D PCA space.
2.3 Clustering Algorithms
K-Means: K-Means was tuned using the Elbow Method and Silhouette Score over k=2 to k=10. Both methods agreed on k=2 as the optimal number of clusters, with the highest Silhouette Score of 0.6010 at k=2, matching the known 2-class structure of the dataset. The k-means++ initialisation was used to avoid poor random starts.
Hierarchical Clustering: Agglomerative Clustering was applied with n_clusters=2 across three linkage methods: Ward, Complete, and Average. The dendrogram (Ward linkage, n=500 sample) confirmed 2 dominant branches with a clear cut point at distance 20. Average linkage produced the best Silhouette Score (0.6750) and was selected for the final model.
DBSCAN: DBSCAN hyperparameters were tuned using a k-Distance Graph (k=5) to identify the knee point for eps selection, followed by a grid search over eps (0.5 to 1.7) and min_samples (3 to 10). The configuration eps=1.7, min_samples=3 produced 2 clusters with 0.1% noise and was selected as the final model.

3. Results & Visualisations
3.1 K-Means Results (k=2)
ClusterSamplesPercentageCluster 0 (Not Pulsar)15,83988.5%Cluster 1 (Pulsar)2,05911.5%
The K-Means PCA scatter plot shows two clearly separated regions. Cluster 0 forms a tight dense group in the left region of PCA space, while Cluster 1 spreads along the right and upper region. The cluster size distribution (88.5% / 11.5%) closely approximates the true class split (90.8% / 9.2%), confirming strong recovery of the underlying structure.
3.2 Hierarchical Clustering Results (Average Linkage)
ClusterSamplesPercentageCluster 017,21596.2%Cluster 16833.8%
The Hierarchical PCA scatter plot shows tighter and more compact cluster boundaries compared to K-Means. However, the algorithm underestimated the minority class — identifying only 683 pulsar candidates compared to the true count of 1,639. The dendrogram revealed two dominant branches with a clear merger at high distance, validating the 2-cluster choice.
3.3 DBSCAN Results (eps=1.7, min_samples=3)
LabelSamplesPercentageCluster 017,88499.9%Cluster 130.0%Noise110.1%
The DBSCAN scatter plot reveals that the algorithm grouped 99.9% of all data into a single cluster. Comparing to the Ground Truth, DBSCAN completely failed to separate the two classes. This is because the HTRU2 dataset has uniform density throughout — both classes overlap heavily in feature space, making it impossible for DBSCAN to find meaningful density boundaries.
3.4 Evaluation Metrics Summary
AlgorithmSilhouette ↑DBI ↓CHI ↑ARI ↑NMI ↑K-Means (k=2)0.60101.05589,891.650.60660.4051Hierarchical (avg)0.67500.53478,309.710.52030.4057DBSCAN (1.7, ms=3)0.67600.308235.420.00080.0003

4. Comparative Discussion
4.1 Internal Metrics Analysis
Internal metrics evaluate cluster quality without using ground truth labels. DBSCAN achieved the highest Silhouette Score (0.6760) and lowest Davies-Bouldin Index (0.3082), which appear excellent. However, these scores are misleading — they reflect a single dominant cluster containing 99.9% of data, not meaningful separation. Hierarchical clustering produced the best genuine internal scores: Silhouette of 0.6750 and DBI of 0.5347, indicating compact and well-separated clusters. K-Means achieved the highest Calinski-Harabasz Index (9,891.65), reflecting the densest and most compact cluster structure among algorithms that actually found meaningful clusters.
4.2 External Metrics Analysis
External metrics compare clustering results against known ground truth labels, providing the most meaningful evaluation for this dataset. K-Means achieved the highest Adjusted Rand Index (ARI = 0.6066), indicating strong agreement with the true class labels. A score of 0.61 on an imbalanced dataset is considered good performance. Hierarchical clustering achieved ARI = 0.5203 — moderate agreement, limited by its underestimation of the minority pulsar class. DBSCAN's ARI of 0.0008 and NMI of 0.0003 confirm complete failure to recover class structure, making its high internal scores irrelevant.
4.3 Agreement Between Metrics
There is significant disagreement between internal and external metrics for DBSCAN. This highlights a fundamental limitation: internal metrics can be high even when clustering is meaningless, particularly when one cluster dominates. For the HTRU2 dataset, external metrics (ARI and NMI) are the most reliable evaluation criteria because ground truth is available. Both K-Means and Hierarchical show reasonable agreement between internal and external metrics, confirming their results are genuinely meaningful.
4.4 Impact of PCA on Clustering
PCA reduced 8 features to 2 principal components capturing 78.48% of total variance. This enabled clear visual inspection of cluster quality. The 2D PCA scatter plots confirmed that K-Means and Hierarchical produce interpretable cluster boundaries that approximately mirror the true class separation. The strong correlation between features (e.g., excess_kurtosis and skewness at r=0.95) made PCA particularly effective — highly correlated features were compressed without significant information loss.
4.5 Algorithm Strengths & Weaknesses
AlgorithmStrengthsWeaknessesK-MeansFast, good ARI, matches true structure, scalableAssumes spherical clusters, sensitive to initialisationHierarchicalNo k needed for discovery, dendrogram insight, tight clustersSlow on large datasets, underestimates minority classDBSCANHandles noise, arbitrary cluster shapesFails on uniform-density datasets like HTRU2

5. Conclusion & Future Work
5.1 Summary of Findings
This project successfully implemented a complete unsupervised learning pipeline on the HTRU2 Pulsar dataset. Three clustering algorithms were applied, evaluated using five metrics, and compared against ground truth labels.

Preprocessing: StandardScaler was essential due to a 100x difference in feature scales. Zero missing values meant no imputation was required.
PCA: Reduced 8 dimensions to 2 principal components capturing 78.48% of total variance, enabling effective visualisation and confirming high inter-feature correlation.
K-Means: Best overall algorithm for HTRU2 with ARI=0.6066 and CHI=9,891.65. Cluster distribution (88.5%/11.5%) closely matches the true class split (90.8%/9.2%).
Hierarchical: Produced the tightest internal clusters (Silhouette=0.6750, DBI=0.5347) using average linkage but underestimated the minority pulsar class (683 vs 1,639 true).
DBSCAN: Failed on this dataset due to uniform data density, achieving ARI=0.0008. High internal scores were misleading — caused by one dominant cluster containing 99.9% of data.

5.2 Limitations

Class imbalance (90.8% vs 9.2%) makes it difficult for clustering algorithms to correctly size the minority cluster without supervision.
DBSCAN is fundamentally unsuitable for uniformly dense datasets regardless of hyperparameter tuning.
Hierarchical clustering required significantly more computation time than K-Means on 17,898 samples.
PCA loses 21.52% of variance when reducing to 2 components, potentially discarding discriminative information.

5.3 Future Work

Gaussian Mixture Models (GMM): Probabilistic clustering that handles overlapping clusters and soft assignments — better suited for imbalanced datasets like HTRU2.
Ensemble Clustering: Combine K-Means and Hierarchical results using consensus clustering to improve minority class detection.
UMAP: Replace PCA with non-linear dimensionality reduction for potentially cleaner cluster separation in 2D space.
Different Distance Metrics: Evaluate Manhattan and Cosine distances in Hierarchical clustering to improve minority class recovery.
Resampling: Apply SMOTE oversampling before clustering to reduce the impact of class imbalance on algorithm performance.

5.4 References

R. J. Lyon et al. Fifty Years of Pulsar Candidate Selection. Monthly Notices of the Royal Astronomical Society, 2016.
UCI Machine Learning Repository: HTRU2 Dataset. archive.ics.uci.edu
scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, 2011.
Ester, M., et al. A Density-Based Algorithm for Discovering Clusters. KDD-96, 1996.
Ward, J. H. Hierarchical Grouping to Optimize an Objective Function. JASA, 1963.
