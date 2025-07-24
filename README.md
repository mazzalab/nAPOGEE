# APOGEE: Predicting Pathogenicity of mt-rRNA and tRNA SNVs

This repository contains the implementation of two models, **rAPOGEE** and **tAPOGEE**, designed to predict the pathogenicity of mt-rRNA and tRNA SNVs, respectively. The repository is structured to allow users to replicate the analysis, train the models, and perform downstream analyses.

---

## Repository Structure

### 1. **rAPOGEE** and **tAPOGEE** Folders
Each folder contains the following components:

#### **Key Notebooks**
1. **Feature Extraction (`features.ipynb`)**:  
   Extracts features from the input data for model training.  
   - Input: Raw data files from the `data` folder.  
   - Output: Feature matrices saved in the `checkpoints` folder.

2. **Model Selection (`model_selection.ipynb`)**:  
   Tunes hyperparameters, trains candidate models, and selects the most performant model.  
   - Input: Feature matrices from the `checkpoints` folder.  
   - Output: Trained models and evaluation metrics saved in the `checkpoints` folder.

3. **Prediction (`predict.ipynb`)**:  
   Generalizes model predictions to the entire SNV population and converts scores into Bayesian posteriors.  
   - Input: Trained models and feature matrices from the `checkpoints` folder.  
   - Output: Prediction scores and posterior probabilities saved in the `checkpoints` folder.

#### **Subfolders**
- **`downstream_analysis/`**:  
  Contains additional notebooks for analyzing the fitted models and predictions. Examples include:  
  - Population frequency investigations.  
  - Spatial autocorrelation of pathogenicity.  
  - Feature importance analysis.

- **`data/`**:  
  Contains all input files required to run the analysis, such as raw feature data and embeddings.

- **`checkpoints/`**:  
  Stores intermediate results, trained models, and predictions.  
  - **Purpose**: Allows users to run any notebook independently without regenerating intermediate results.  
  - **Note**: To replicate the pipeline from scratch, delete the contents of this folder.

---

## How to Use This Repository

### 1. Prerequisites
This repository requires Python 3.8 or higher. Install the required Python packages listed in the `requirements.txt` file.  

### 2. Running the Pipeline

You can either:

#### Run the entire pipeline from scratch:
1. Delete the contents of the `checkpoints` folder in both `rAPOGEE` and `tAPOGEE`.
2. Execute the notebooks in the following order:
   - `features.ipynb`  
   - `model_selection.ipynb`  
   - `predict.ipynb`  

#### Run specific notebooks:
- Use the pre-computed checkpoints to skip intermediate steps.
- Ensure the required files are present in the `checkpoints` folder.

---

## Notes for Reviewers and Users

- The `downstream_analysis` subfolder in both `rAPOGEE` and `tAPOGEE` contains additional analyses on the predictions and models, such as population frequency investigations, spatial autocorrelation, and feature importance analysis.
- The `data` folder includes all input files required to run the analysis.
- The `checkpoints` folder contains pre-computed intermediate results to facilitate running specific notebooks without regenerating all intermediate steps.
- To replicate the pipeline from scratch, delete the contents of the `checkpoints` folder and ensure all required input files are present in the `data` folder.

This structure ensures that users and reviewers can easily replicate the results or adapt the pipeline for their own datasets.