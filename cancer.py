import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# function to process the data
def process_muts(mutation_file):
    # Tumor_Sample_Barcode is the sample id
    tumor_sample_barcode = mutation_file['Tumor_Sample_Barcode']

    # Hugo_Symbol is the gene name
    Hugo_Symbol = mutation_file['Hugo_Symbol']

    # create a dictionary where the key is the sample id and the value is a list of genes that are mutated
    sample_genes = {}
    # create a list of all genes
    all_genes = []

    # loop through the data and populate the dictionary
    for i in tumor_sample_barcode.index:
        # check if the sample id is in the dictionary
        if tumor_sample_barcode[i] not in sample_genes:
            sample_genes[tumor_sample_barcode[i]] = []
        # check if the gene name is more than 1 character long
        if len(Hugo_Symbol[i]) > 1:
            sample_genes[tumor_sample_barcode[i]].append(Hugo_Symbol[i])
            all_genes.append(Hugo_Symbol[i])

    # get a unique list of all genes
    all_genes = list(set(all_genes))

    # create a dataframe where the rows are the sample ids and the columns are the genes
    gene_df = pd.DataFrame(0, index = all_genes, columns = list(sample_genes.keys()))

    # new_gene_df = gene_df.copy()

    # populate the dataframe
    for i in sample_genes.keys():
        for j in sample_genes[i]:
            gene_df[i][j] = 1

    # gene_df = new_gene_df
    
    # transpose the dataframe so that the rows are the samples and the columns are the genes
    gene_df = gene_df.T

    # remove any columns that sum to 0
    gene_df = gene_df.loc[:, (gene_df != 0).any(axis=0)]

    return gene_df

# read in the data
prad = pd.read_csv('/Users/amruthavarshini/Cancer_Gene_Atlas/prad_data_mutations.txt',sep='\t')
brca = pd.read_csv('/Users/amruthavarshini/Cancer_Gene_Atlas/brca_data_mutations.txt',sep='\t')

# process the data
prad_gene_df = process_muts(prad)
brca_gene_df = process_muts(brca)

# find the genes that are in both dataframes
common_genes = list(set(prad_gene_df.columns).intersection(set(brca_gene_df.columns)))

# keep only the genes that are in both dataframes
prad_gene_df = prad_gene_df[common_genes]
brca_gene_df = brca_gene_df[common_genes]

# combine the dataframes adding a column that indicates the cancer type
prad_gene_df['cancer_type'] = 'prad'
brca_gene_df['cancer_type'] = 'brca'

combined_df = pd.concat([prad_gene_df, brca_gene_df])
# print(combined_df)

# Dividing data into features (X) and prediction (y)
X = combined_df.drop('cancer_type', axis=1)
y = combined_df['cancer_type']

# Dividing data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# classifier
clf = RandomForestClassifier(n_estimators=5000, random_state=42)

# Perform cross-validation
scores = cross_val_score(clf, X, y, cv=10)

print(f'Cross-Validation Accuracy Scores: {scores}')
print(f'Average Score: {scores.mean()*100:.2f}%')

# training the classifier
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')

# define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# create the grid search object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# fit the grid search object to the data
grid_search.fit(X_train, y_train)

# print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# print the best score
print(f'Best Score: {grid_search.best_score_}')

# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame to view features and their importance
features = pd.DataFrame()
features['Feature'] = X_train.columns
features['Importance'] = importances

# Sort the features based on importance
features.sort_values(by=['Importance'], ascending=False, inplace=True)

# Top 50 important features
print(features.head(50))

# Get the top 50 features
top_50_features = features.head(50)['Feature'].tolist()

# Select only the top 50 features from X_train and X_test
X_train_top_50 = X_train[top_50_features]
X_test_top_50 = X_test[top_50_features]

# Now you can use these as inputs to your RandomForest model
clf_top_50 = RandomForestClassifier(n_estimators=5000, random_state=42)

# Perform cross-validation
scores_top_50 = cross_val_score(clf_top_50, X_train_top_50, y_train, cv=10)

print(f'Cross-Validation Accuracy Scores (Top 50 Features): {scores_top_50}')
print(f'Average Score (Top 50 Features): {scores_top_50.mean()*100:.2f}%')

# Train the classifier
clf_top_50.fit(X_train_top_50, y_train)

# Make predictions on the test set
y_pred_top_50 = clf_top_50.predict(X_test_top_50)

# Calculate the accuracy of the classifier
accuracy_top_50 = accuracy_score(y_test, y_pred_top_50)

print(f'Accuracy (Top 50 Features): {accuracy_top_50*100:.2f}%')
