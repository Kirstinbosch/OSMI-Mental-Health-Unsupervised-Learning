# OSMI Mental Health in Tech Survey 2016 – Unsupervised Learning

## IU International University of Applied Sciences

**Course:** Unsupervised Learning (DLBDSMLUSL01)

This repository contains the code used for an unsupervised learning analysis of the **OSMI Mental Health in Tech Survey 2016**.

The aim of the analysis is to explore patterns among respondents based on their survey responses and investigate whether meaningful groups can be identified using distance-based dimensionality reduction and clustering techniques.

## Analysis

The analysis consists of the following stages:

1. Data preprocessing and cleaning
2. Exploratory data analysis
3. Gower distance calculation
4. Non-Metric Multidimensional Scaling (NMDS)
5. Hierarchical clustering
6. Cluster interpretation

Because the survey contains a mixture of numerical, binary and categorical variables, **Gower's distance** was used to calculate dissimilarities between respondents.

The resulting Gower distance matrix was then used for **Non-Metric Multidimensional Scaling (NMDS)** to obtain a two-dimensional representation of the respondent dissimilarities.

Finally, **hierarchical clustering** was applied using complete and average linkage. Solutions containing two to five clusters were compared using silhouette scores and cluster sizes.

## Dataset

**Dataset:** OSMI Mental Health in Tech Survey 2016

**Source:** Open Sourcing Mental Health (OSMI)

The dataset is available through Kaggle:

https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016

The original dataset is not included in this repository.

## Preprocessing

The original survey data was cleaned before the unsupervised learning analysis.

The preprocessing included:

- removing respondents with less than 50% response completeness
- removing variables with 50% or more missing values
- cleaning and standardising gender responses
- converting the `Work Position` variable into binary role indicators
- identifying invalid age values and replacing them with missing values
- imputing missing age values using the median
- removing open-text responses that were not suitable for the distance-based analysis

The resulting dataset contained **1,413 respondents and 60 variables**.

## Methods

### Gower Distance

Gower's distance was used because the dataset contains different types of variables.

For numerical variables, differences were normalised by the observed range. Binary and categorical variables were compared according to whether their responses matched.

The resulting pairwise distance matrix contains values between 0 and 1, where smaller values represent more similar respondents.

### Non-Metric Multidimensional Scaling

NMDS was applied to the Gower distance matrix to produce a two-dimensional representation of the respondent dissimilarities.

Multiple random seeds were evaluated and the solution with the lowest stress was retained.

The selected solution produced a normalized stress value of approximately **0.3015**. This indicates considerable distortion in the two-dimensional representation, so the NMDS plot was interpreted cautiously.

### Hierarchical Clustering

Agglomerative hierarchical clustering was performed using:

- Complete linkage
- Average linkage

Solutions containing between two and five clusters were evaluated.

Silhouette scores and cluster sizes were considered when comparing the resulting solutions.

Although average linkage with two clusters produced the highest silhouette score (**0.1650**), the resulting clusters contained only 5 and 1,408 respondents and were therefore unsuitable as a meaningful segmentation.

Complete linkage with three clusters produced more substantial groups:

- Cluster 1: 29 respondents
- Cluster 2: 442 respondents
- Cluster 3: 942 respondents

Although the silhouette score for this solution was low (**0.0612**), it was selected as the most interpretable exploratory solution.

## Main Findings

The three-cluster solution showed differences in reported mental-health experiences and attitudes towards discussing mental health in the workplace.

Clusters 1 and 3 contained higher proportions of respondents reporting previous or current mental-health experiences than Cluster 2.

Cluster 1 was substantially smaller than the other clusters, but a relatively high proportion of respondents in this group reported that discussing mental health in the workplace could have negative consequences.

These patterns may provide useful areas for further investigation, particularly concerning workplace psychological safety and access to mental-health support.

However, the clusters were **weakly separated**, and therefore the results should be interpreted as exploratory rather than as definitive respondent classifications.

The survey is also based on self-reported responses, and the analysis cannot establish causal relationships.

## Repository Structure

```text
OSMI-Mental-Health-Unsupervised-Learning/
│
├── data/
│   └── README.md
│
├── results/
│   └── figures/
│
├── src/
│   ├── 01_data_preprocessing.py
│   ├── 02_exploratory_data_analysis.py
│   ├── 03_gower_distance.py
│   ├── 04_nmds.py
│   ├── 05_hierarchical_clustering.py
│   └── 06_cluster_interpretation.py
│
├── .gitignore
├── README.md
└── requirements.txt
