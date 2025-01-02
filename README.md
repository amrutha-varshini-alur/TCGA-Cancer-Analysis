# TCGA - Cancer Type Classification Using Mutation Data
This project implements a Random Forest Classifier to distinguish between two types of cancer — Prostate Adenocarcinoma (PRAD) and Breast Invasive Carcinoma (BRCA) — based on gene mutation data. The workflow includes data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation.


**Project Overview**
The goal of this project is to classify cancer types using mutation data from two datasets: PRAD and BRCA. A Random Forest Classifier is trained and evaluated to determine its performance in classifying these cancer types. The classifier is further optimized using feature selection and hyperparameter tuning.


<ins>**Dataset Description**</ins>

**Datasets:**
PRAD Mutation Data: Contains mutation information for Prostate Adenocarcinoma.
BRCA Mutation Data: Contains mutation information for Breast Invasive Carcinoma.


**File Format:** Tab-separated values (.txt).


**Columns Used:**
Tumor_Sample_Barcode: Identifies the sample.
Hugo_Symbol: Represents the mutated gene name.


**Dependencies**
This project is implemented in Python. Below are the primary libraries required:<br>
pandas: Data manipulation and analysis.<br>
numpy: Numerical computations.<br>
scikit-learn: Machine learning algorithms.<br>
GridSearchCV: Hyperparameter optimization.


**Install dependencies using:**
bash
Copy code
pip install pandas numpy scikit-learn


**Code Workflow**
1. Data Preprocessing
Mutation data is processed into a binary matrix where:
Rows represent samples.
Columns represent mutated genes (presence = 1, absence = 0).
Only genes shared between PRAD and BRCA datasets are retained.
2. Combining Data
PRAD and BRCA data are combined into a single dataset with a cancer_type column as the target variable.
3. Model Training and Evaluation
A Random Forest Classifier is trained using the combined dataset.
Cross-validation is performed to evaluate model stability.
4. Feature Selection
Top 50 important features (genes) are selected based on feature importance scores from the Random Forest model.
5. Hyperparameter Tuning
GridSearchCV is used to optimize the Random Forest hyperparameters:
n_estimators: Number of trees.
max_features: Number of features to consider for splits.
6. Performance Metrics
Accuracy is calculated for the full feature set and the top 50 features.


**Usage**
**Clone the repository:**<br>
bash<br>
Copy code<br>
git clone https://github.com/your-username/cancer-classification.git<br>
cd cancer-classification


**Run the script:**<br>
Ensure your dataset files are in the correct path (replace the paths in the script if necessary). Execute the Python script:<br>
bash<br>
Copy code<br>
python cancer_classification.py


**Expected Output:**<br>
Cross-validation accuracy scores.<br>
Best parameters from GridSearchCV.<br>
Feature importance ranking.<br>
Final model accuracy with the top 50 features.
