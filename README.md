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


## Folder structure

```bash
./
├── common_data
│   └── nAPOGEE_Gnomad_het.csv # Contains gnomad frequencies and heteroplasmies for both mitochondrial tRNA and rRNA SNVs.
├── CustomEstimators # Custom scikit-learn estimators
│   ├── DataFrameUtils.py # Allows to work with pandas.DataFrame objects
│   ├── FilterByCorrelation.py # Correlation-based feature filter
│   ├── GridSearchOOB_BinaryClassifier.py # GridSearch tuning based on the OOB scores rather than the cross validation (specific for Bagging binary classifiers)
│   ├── PartialPCA.py # PCA transformation on a subset of the total features
│   └── ZeroOneInflatedBeta.py # Beta distribution dealing with extreame values
├── customFunctions
│   └── utils.py # just utils
├── figures
│   ├── about_feature_importance_and_shap.ipynb
│   ├── distributions.ipynb
│   ├── genes_boxplot.ipynb
│   ├── ROC_curves.ipynb
│   └── svg
│       ├── feature_importance.svg
│       ├── LISA_tRNA.svg
│       ├── posterior_genes_box.svg
│       ├── rAPOGEE_distribution.svg
│       ├── rAPOGEE_genes_box.svg
│       ├── rAPOGEE_likelihood.svg
│       ├── rAPOGEE_misclassification.svg
│       ├── rAPOGEE_posterior.svg
│       ├── robosome_scatter.svg
│       ├── ROC_curves.svg
│       ├── rRNA_gnomad_pathogenicity.svg
│       ├── shap_contributions_tAPOGEE.svg
│       ├── shap_mithril_1830.svg
│       ├── suzuki_average_score.svg
│       ├── tAPOGEE_distribution.svg
│       ├── tAPOGEE_genes_box.svg
│       ├── tAPOGEE_likelihood.svg
│       ├── tAPOGEE_misclassification.svg
│       ├── tAPOGEE_posterior.svg
│       └── tRNA_gnomad_pathogenicity.svg
├── rAPOGEE
│   ├── checkpoints
│   │   ├── altered_sequences.fa # altered RNA sequences
│   │   ├── candidate_models
│   │   │   ├── model_knn.pk
│   │   │   ├── model_rf.pk
│   │   │   └── model_svc.pk
│   │   ├── cross_validation_test.pk # `y_true` and `y_pred` resulting from cross-validating the rAPOGEE model on its training set
│   │   ├── lisa.csv # LISA values from spatial autocorrelation analysis
│   │   ├── lisa.html # 3D visualization of the positive-LISA residues pathogenicity on the rRNA complex
│   │   ├── mithril_features.csv # features extracted of rRNA SNVs
│   │   └── rAPOGEE_predictions.csv # scores and pathogenicity probability of rRNA SNVs
│   ├── data
│   │   ├── alignments.zip # philogenetic alignment of MIDORI sequences of MT-RNR1 and MT-RNR2
│   │   ├── metazoa_lineage.txt
│   │   ├── Mithril_embedding_variants_256windows.csv # RNA-MSM embedding channels for each rRNA SNV (using the 256 window build on the variant in exam)
│   │   ├── mithril_V053.txt # MITIMPACT (version 5.3)
│   │   ├── PTM.txt # residues affected by post transcriptional modifications
│   │   ├── structures
│   │   │   ├── alphafold_wt_mt_ribosome.pdb # WT mt-rRNA complex structure (in PDB format) retrieved by using alphafold
│   │   │   ├── Mito_rRNA_variant_folded_R2DT_enerFinal.txt
│   │   │   ├── residues_3Dcoord.csv # residues coordinates according to the WT rRNA PDB
│   │   │   └── rRNA_homoSapiens_RNAcentralPred.txt
│   │   ├── test_set.txt # Dataset 3 and 4
│   │   └── training_test_set_unified.txt # Dataset 1 and 2
│   ├── downstream_analysis
│   │   ├── gnomad_variants.ipynb # study of the predicted pathogenicity of observed population mt-rRNA SNVs in function of heteroplasmy
│   │   └── spatial.ipynb # ribosome spatial autocorrelation analysis
│   ├── features.ipynb # feature extraction
│   ├── model_selection.ipynb # tuning and training of the proposed classifiers to choose the best estimator and to estimate its performace on unseen variables 
│   └── predict.ipynb # generalization fo the best model on the whole rRNA SNVs domain and pathogenicity probability estimation
├── README.md
├── requirements.txt
└── tAPOGEE
    ├── checkpoints
    │   ├── candidate_models
    │   │   ├── model_knn.pk
    │   │   ├── model_rf.pk
    │   │   └── model_svc.pk
    │   ├── lisa.csv
    │   ├── mithril_features.csv
    │   ├── shap_values.pk
    │   └── tAPOGEE_predictions.csv
    ├── data
    │   ├── aligned_position_suzuki.csv
    │   ├── HomoSapiens_mttRNA_Seq.aln
    │   ├── HomoSapiens_mttRNA_Str_simpl.aln
    │   ├── Mithril_msm_embedding.csv
    │   ├── mithril_V03.txt # MITIMPACT (version 3.0)
    │   ├── Mito_tRNA_variants_folded.txt
    │   ├── oeuf_lake_et_al.tsv
    │   ├── PTM_and_domains.txt
    │   ├── taxa_Metazoa.txt
    │   ├── test_set.txt
    │   ├── training_set.txt
    │   ├── tRNA_Alignments # mt-tRNA functional alignment (just sequences) from trnadb
    │   │   ├── Metazoa_Ala.aln
    │   │   ├── Metazoa_Arg.aln
    │   │   ├── Metazoa_Asn.aln
    │   │   ├── Metazoa_Asp.aln
    │   │   ├── Metazoa_Cys.aln
    │   │   ├── Metazoa_Gln.aln
    │   │   ├── Metazoa_Glu.aln
    │   │   ├── Metazoa_Gly.aln
    │   │   ├── Metazoa_His.aln
    │   │   ├── Metazoa_Ile.aln
    │   │   ├── Metazoa_Leu1.aln
    │   │   ├── Metazoa_Leu2.aln
    │   │   ├── Metazoa_Lys.aln
    │   │   ├── Metazoa_Met.aln
    │   │   ├── Metazoa_Phe.aln
    │   │   ├── Metazoa_Pro.aln
    │   │   ├── Metazoa_Ser1.aln
    │   │   ├── Metazoa_Ser2.aln
    │   │   ├── Metazoa_Thr.aln
    │   │   ├── Metazoa_Trp.aln
    │   │   ├── Metazoa_Tyr.aln
    │   │   └── Metazoa_Val.aln
    │   └── tRNA_Alignments_structure # mt-tRNA functional alignment (just dot-braket secondary structures) from trnadb
    │       ├── Metazoa_Ala.aln
    │       ├── Metazoa_Arg.aln
    │       ├── Metazoa_Asn.aln
    │       ├── Metazoa_Asp.aln
    │       ├── Metazoa_Cys.aln
    │       ├── Metazoa_Gln.aln
    │       ├── Metazoa_Glu.aln
    │       ├── Metazoa_Gly.aln
    │       ├── Metazoa_His.aln
    │       ├── Metazoa_Ile.aln
    │       ├── Metazoa_Leu1.aln
    │       ├── Metazoa_Leu2.aln
    │       ├── Metazoa_Lys.aln
    │       ├── Metazoa_Met.aln
    │       ├── Metazoa_Phe.aln
    │       ├── Metazoa_Pro.aln
    │       ├── Metazoa_Ser1.aln
    │       ├── Metazoa_Ser2.aln
    │       ├── Metazoa_Thr.aln
    │       ├── Metazoa_Trp.aln
    │       ├── Metazoa_Tyr.aln
    │       └── Metazoa_Val.aln
    ├── downstream_analysis
    │   ├── feature_importance.ipynb # shap feature importance estimation
    │   ├── gnomad_variants.ipynb
    │   ├── spatial.ipynb
    │   └── wt_ss_figures
    │       ├── Ala_ss.ps
    │       ├── Arg_ss.ps
    │       ├── Asn_ss.ps
    │       ├── Asp_ss.ps
    │       ├── Cys_ss.ps
    │       ├── Gln_ss.ps
    │       ├── Glu_ss.ps
    │       ├── Gly_ss.ps
    │       ├── His_ss.ps
    │       ├── Ile_ss.ps
    │       ├── Leu1_ss.ps
    │       ├── Leu2_ss.ps
    │       ├── Lys_ss.ps
    │       ├── Met_ss.ps
    │       ├── Phe_ss.ps
    │       ├── Pro_ss.ps
    │       ├── Ser1_ss.ps
    │       ├── Ser2_ss.ps
    │       ├── _suzuki_ss.eps
    │       ├── Thr_ss.ps
    │       ├── Trp_ss.ps
    │       ├── Tyr_ss.ps
    │       └── Val_ss.ps
    ├── external_tools_comparison
    │   ├── mitotip_comparison.ipynb
    │   ├── mitotip_data
    │   │   ├── mitotip_2017_mask.csv
    │   │   ├── mitotip_benign_variants.csv
    │   │   └── mitotip_pathogenic_variants.csv
    │   ├── PON_comparison.ipynb
    │   └── PON_data
    │       ├── PON_predictions.txt
    │       └── PON_training_set.txt
    ├── features.ipynb
    ├── model_selection.ipynb
    ├── model_selection_knn.py
    ├── model_selection_rf.py
    ├── model_selection_svc.py
    └── predict.ipynb
```