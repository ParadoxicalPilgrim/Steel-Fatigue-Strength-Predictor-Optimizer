# Steel-Fatigue-Strength-Predictor
An end-to-end machine learning pipeline replicating and extending this paper's [https://link.springer.com/article/10.1186/2193-9772-3-8#additional-information] research, featuring LOOCV, rigorous EDA, and genetic algorithm optimization.


# Steel Fatigue Strength Predictor

A robust machine learning pipeline and data-driven analytical framework for predicting the rotating bending fatigue strength of various steel grades based on their chemical composition and thermal history. This project replicates and extends published research from the paper [https://link.springer.com/article/10.1186/2193-9772-3-8#additional-information] on the National Institute for Materials Science (NIMS), Japan, dataset incorporating advanced model evaluation and Genetic Algorithm (GA) optimization.

## Project Overview
This repository implements an end-to-end materials informatics workflow. It bridges metallurgical principles with machine learning, transitioning from thermodynamic data normalization and exploratory data analysis (EDA) to rigorous algorithmic modeling and performance evaluation. Furthermore, the framework integrates Genetic Algorithms for advanced optimization tasks.

## Key Pipeline Features
* **Thermodynamic Data Normalization:** Standardizes multi-grade thermal histories, correctly managing missing phase-transformation parameters through domain-specific imputation and zero-time bounding.
* **Exploratory Data Analysis (EDA):** 
  * Singular Value Decomposition / Principal Component Analysis (SVD-PCA) coupled with K-Means clustering ($K=3$) to visualize intrinsic metallurgical groupings.
  * Entropy-based Information Gain ($IG$) evaluation to assess the predictive capability of all 25 individual features.
* **Rigorous Validation Protocol:** Employs **Leave-One-Out Cross Validation (LOOCV)** across 12 distinct machine learning architectures, executing over thousands of training loops to eliminate over-fitting on small-sample material datasets.
* **State-of-the-Art Modeling:** Benchmarks traditional linear and regression transformations against advanced tree ensembles and Artificial Neural Networks (ANN), identifying the optimal predictor ($R^2 > 0.97$).
* **Genetic Algorithm Integration:** Incorporates Genetic Algorithm (GA) optimization to further refine predictive performance and feature configurations.

## Repository Status & Deployment Note
* **Backend Analytical Pipeline:** Fully complete, optimized, and validated with comprehensive Jupyter notebook and model evaluation scripts.
* **Web Application UI:** The live Streamlit interface link is currently unavailable as it is undergoing active modifications and enhancements related to the Genetic Algorithm integration. 
