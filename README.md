# ⚙️ Steel Fatigue Strength Predictor & AI Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[TERA_LIVE_LINK_YAHAN_DAAL])
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-lightblue.svg)](https://scipy.org/)

A full-stack, data-driven web application and analytical framework that bridges the gap between core metallurgy and machine learning. This tool predicts the rotating bending fatigue strength of various steel grades based on their exact chemical composition and thermal history, and utilizes AI to reverse-engineer optimal manufacturing recipes.

🚀 **[Experience the Live Application Here](https://steel-strength-optimizer.streamlit.app)**

---

## 🔬 Research & Academic Foundation

This project directly replicates and significantly extends the published research from the paper:  
**(https://link.springer.com/article/10.1186/2193-9772-3-8#additional-information)** *(Springer)*. 

Using the National Institute for Materials Science (NIMS), Japan dataset, this project transitions the foundational thermodynamic data normalization into a fully interactive, optimized predictive pipeline.

---

## ✨ Key Application Features

### 1. Forward Predictor (Artificial Neural Network)
* Enter the exact composition across 25 specific features (Chemical Composition, Heat Treatment phases, and Inclusions).
* The integrated **Artificial Neural Network (ANN)** instantly calculates the predicted fatigue strength (MPa) with high accuracy ($R^2 > 0.97$), completely eliminating the need for expensive physical trial-and-error testing.

### 2. Smart Inverse Optimizer (Genetic Algorithm)
* **Target-Driven Engineering:** Input a desired fatigue strength target (e.g., 650 MPa).
* **Constraint Locking:** Engineers can lock specific parameters (like Carbon % or Tempering Temperature) based on inventory or plant constraints.
* **AI Optimization:** The application uses **SciPy's Differential Evolution (Genetic Algorithm)** to explore thousands of parameter combinations and reverse-engineer the most optimal, physically viable 25-parameter recipe to achieve the exact target strength.

---

## 📊 Backend Pipeline & Data Architecture

Behind the intuitive Streamlit UI lies a rigorous machine learning pipeline:

* **Thermodynamic Data Normalization:** Standardizes multi-grade thermal histories, correctly managing missing phase-transformation parameters through domain-specific imputation and zero-time bounding.
* **Exploratory Data Analysis (EDA):**
  * SVD/PCA coupled with K-Means clustering ($K = 3$) to visualize intrinsic metallurgical groupings.
  * Entropy-based Information Gain ($IG$) evaluation to assess the predictive capability of all individual features.
* **Rigorous Validation Protocol:** Employs **Leave-One-Out Cross-Validation (LOOCV)** across 12 distinct machine learning architectures, executing over thousands of training loops to aggressively eliminate over-fitting on small-sample material datasets.
* **State-of-the-Art Modeling:** Benchmarked traditional linear and regression transformations against advanced tree ensembles and Neural Networks, finalizing the ANN for production deployment.

---

## 💻 Tech Stack

* **Frontend / UI:** Streamlit
* **Core Machine Learning:** Scikit-Learn (ANN, LOOCV, PCA, K-Means)
* **Mathematical Optimization:** SciPy (Differential Evolution / Genetic Algorithm)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

---
*Developed as an end-to-end Machine Learning portfolio project, translating research-grade metallurgical data into a scalable software solution.*
