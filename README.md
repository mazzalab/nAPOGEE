# nAPOGEE: Predicting the pathogenicity of mt-rRNA and mt-tRNA SNVs

This repository contains the implementation of two models, **rAPOGEE** and **tAPOGEE**, designed to predict the pathogenicity of mt-rRNA and tRNA SNVs, respectively. The repository is structured to allow users to replicate the analysis, train the models, and perform downstream analyses.


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
  - Ablation studies and grid-search diagnostics.
  - External benchmarking against PON and MitoTIP.

- **`data/`**:  
  Contains all input files required to run the analysis, such as raw feature data and embeddings.

- **`checkpoints/`**:  
  Stores intermediate results, trained models, and predictions.  
  - **Purpose**: Allows users to run any notebook independently without regenerating intermediate results.  
  - **Note**: To replicate the pipeline from scratch, delete the contents of this folder.


## Directory Tree

- The `downstream_analysis` subfolder in both `rAPOGEE` and `tAPOGEE` contains additional analyses on the predictions and models, such as population frequency investigations, spatial autocorrelation, feature importance analysis, ablation studies, and external benchmarking.
- The `data` folder includes all input files required to run the analysis.
- The `checkpoints` folder contains pre-computed intermediate results to facilitate running specific notebooks without regenerating all intermediate steps.
- To replicate the pipeline from scratch, delete the contents of the `checkpoints` folder and ensure all required input files are present in the `data` folder.

The following directory tree provides the folder structure along with details about each file:

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
│   ├── ROC_curves.ipynb
│   ├── about_feature_importance_and_shap.ipynb
│   ├── distributions.ipynb
│   ├── genes_boxplot.ipynb
│   └── svg
│       ├── ablation_study_rAPOGEE_cv(ap).svg
│       ├── ablation_study_rAPOGEE_cv(auc).svg
│       ├── ablation_study_tAPOGEE_training_and_test.svg
│       ├── ablation_study_tAPOGEE_training_cv.svg
│       ├── feature_importance.svg
│       ├── LISA_tRNA.svg
│       ├── posterior_genes_box.svg
│       ├── rAPOGEE_distribution.svg
│       ├── rAPOGEE_genes_box.svg
│       ├── rAPOGEE_likelihood.svg
│       ├── rAPOGEE_misclassification.svg
│       ├── rAPOGEE_posterior.svg
│       ├── ribosome_scatter.svg
│       ├── ROC_curves.svg
│       ├── rRNA_gnomad.svg
│       ├── rRNA_gnomad_pathogenicity.svg
│       ├── rRNA_gnomad_violins.svg
│       ├── shap_contributions_tAPOGEE.svg
│       ├── shap_mithril_1830.svg
│       ├── suzuki_average_score.svg
│       ├── tAPOGEE_distribution.svg
│       ├── tAPOGEE_genes_box.svg
│       ├── tAPOGEE_likelihood.svg
│       ├── tAPOGEE_misclassification.svg
│       ├── tAPOGEE_posterior.svg
│       ├── tRNA_gnomad.svg
│       ├── tRNA_gnomad_pathogenicity.svg
│       └── tRNA_gnomad_violins.svg
├── rAPOGEE
│   ├── checkpoints
│   │   ├── altered_sequences.fa # altered RNA sequences
│   │   ├── candidate_models
│   │   │   ├── model_knn.pk
│   │   │   ├── model_rf.pk # model chosed for rAPOGEE
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
│   │   ├── ablation_studies
│   │   │   ├── ablation_studies.ipynb # feature ablation analysis
│   │   │   ├── ablation_cv
│   │   │   │   ├── cv_predictions_no_conservation.pk
│   │   │   │   ├── cv_predictions_no_ddG.pk
│   │   │   │   ├── cv_predictions_no_mlc.pk
│   │   │   │   ├── cv_predictions_no_position.pk
│   │   │   │   └── cv_predictions_no_rnamsm.pk
│   │   │   ├── drop_conservation.py
│   │   │   ├── drop_ddg.py
│   │   │   ├── drop_mlc.py
│   │   │   ├── drop_position.py
│   │   │   └── drop_rnamsm.py
│   │   ├── collinearity.ipynb # feature collinearity analysis
│   │   ├── gnomad_variants.ipynb # study of the predicted pathogenicity of observed population mt-rRNA SNVs in function of heteroplasmy
│   │   ├── spatial.ipynb # ribosome spatial autocorrelation analysis
│   │   └── was_it_worth_gridsearching
│   │       ├── analize_gridsearch.ipynb # grid-search diagnostics
│   │       ├── gridsearch.pk
│   │       └── run_gridsearch.py
│   ├── features.ipynb # feature extraction
│   ├── model_selection.ipynb # tuning and training of the proposed classifiers to choose the best estimator and to estimate its performace on unseen variables 
│   └── predict.ipynb # generalization of the best model on the whole rRNA SNVs domain and pathogenicity probability estimation
├── README.md # what you're reading just right now
├── requirements.txt
└── tAPOGEE
    ├── checkpoints
    │   ├── candidate_models
    │   │   ├── model_knn.pk
    │   │   ├── model_rf.pk
    │   │   └── model_svc.pk # model chosed for tAPOGEE
    │   ├── lisa.csv # LISA values from spatial autocorrelation analysis
    │   ├── mithril_features.csv # features extracted of tRNA SNVs
    │   ├── shap_values.pk # feature contributions for each tRNA SNV computed via SHAP
    │   └── tAPOGEE_predictions.csv # scores and pathogenicity probability of tRNA SNVs
    ├── data
    │   ├── aligned_position_suzuki.csv # human WT mt-tRNA alignment according to "https://pmc.ncbi.nlm.nih.gov/articles/PMC539966/"
    │   ├── HomoSapiens_mttRNA_Seq.aln # human WT mt-tRNA alignment according to mttrnadb
    │   ├── HomoSapiens_mttRNA_Str_simpl.aln # structural motifs of human mttrnadb alignment
    │   ├── Mithril_msm_embedding.csv # RNA-MSM embedding channels for each tRNA SNV
    │   ├── mithril_V03.txt # MITIMPACT (version 3.0)
    │   ├── Mito_tRNA_variants_folded.txt # predicted secondary structures (in dot-braket) of both WT and variated tRNA and associated free energies
    │   ├── oeuf_lake_et_al.tsv # oeuf score from "https://www.nature.com/articles/s41586-024-08048-x"
    │   ├── PTM_and_domains.txt # tRNA residues involved in post transcriptional modifications
    │   ├── taxa_Metazoa.txt
    │   ├── test_set.txt # Dataset 1 and 2 (tRNA)
    │   ├── training_set.txt # Dataset 3 and 4 (tRNA)
    │   ├── tRNA_Alignments # phylogenetic mt-tRNA functional alignments from trnadb (gene specific)
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
    │   └── tRNA_Alignments_structure # structures from the phylogenetic mt-tRNA functional alignments from trnadb
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
    │   ├── ablation_studies
    │   │   ├── ablation_studies.ipynb # feature ablation analysis
    │   │   ├── drop_conservation.py
    │   │   ├── drop_mlc.py
    │   │   ├── drop_position.py
    │   │   ├── drop_rnamsm.py
    │   │   ├── drop_structure.py
    │   │   └── partial_models
    │   │       ├── model_svc_no_conservation.pk
    │   │       ├── model_svc_no_mlc.pk
    │   │       ├── model_svc_no_position.pk
    │   │       ├── model_svc_no_rnamsm.pk
    │   │       └── model_svc_no_structure.pk
    │   ├── collinearity.ipynb # feature collinearity analysis
    │   ├── feature_importance.ipynb # SHAP feature importance estimation
    │   ├── gnomad_variants.ipynb # study of the predicted pathogenicity of observed population mt-tRNA SNVs in function of heteroplasmy
    │   ├── spatial.ipynb # tRNA spatial autocorrelation analysis
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
    │   ├── PON_comparison.ipynb
    │   ├── PON_data
    │   │   ├── PON_predictions.txt
    │   │   └── PON_training_set.txt
    │   ├── mitotip_comparison.ipynb
    │   └── mitotip_data
    │       ├── mitotip_2017_mask.csv
    │       ├── mitotip_benign_variants.csv
    │       └── mitotip_pathogenic_variants.csv
    ├── features.ipynb # feature extraction
    ├── model_selection.ipynb # evaluate the proposed classifiers to choose the best estimator and test in on test variants 
    ├── model_selection_knn.py # tune and train the knn classifier
    ├── model_selection_rf.py # tune and train the random forest classifier
    ├── model_selection_svc.py # tune and train the SVM classifier
    └── predict.ipynb # generalization of the best model on the whole tRNA SNVs domain and pathogenicity probability estimation
```
