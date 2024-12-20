import GMMchi
import pandas as pd
from tqdm import tqdm

# Updated dataset with more variability
data = {
    "Sample1": [1.2, 2.3, 3.1, 0.5],
    "Sample2": [2.1, 1.8, 3.4, 0.8],
    "Sample3": [1.5, 2.2, 2.9, 1.0],
    "Sample4": [2.0, 3.0, 4.0, 0.7],
    "Sample5": [1.8, 1.0, 2.5, 0.6]
}

# Create a DataFrame
input_data_cancer = pd.DataFrame(data, index=["Gene1", "Gene2", "Gene3", "Gene4"])

# Preprocess data for the selected ID
gene_data = input_data_cancer.loc["Gene1"].dropna()
if gene_data.empty:
    raise ValueError("No valid data available for the selected ID.")

# Ensure preprocessing meets function expectations
preprocessed_data = pd.DataFrame(
    data={sample: [val] for sample, val in zip(input_data_cancer.columns, gene_data)},
    index=["Gene1"]
)

# Run GMMchi
results = GMMchi.GMM_modelingt(
    ID="Gene1",
    input_datanormal=preprocessed_data,
    log2transform=True,
    verbosity=True,
    calc_back=False,
    filt=0.5,
    graphs=True
)

# Unpack results
[means, covars, xf, weights], classif, categories, chi, bins_cell, f = results

# Print relevant outputs
print("Means:", means)
print("Covariances:", covars)
print("Threshold (xf):", xf)
print("Weights:", weights)
print("Classification:", classif)
print("Categories:", categories)
print("Chi-square:", chi)

input_dataf = GMMchi.probe_filter(datainputnorm=preprocessed_data,
                                  log2transform=True,
                                  filt=-0.829)

genes = input_dataf.index  # the index of the dataframe or a list of all genes
categorize = []  # append as list of list of categorized data

for gene in tqdm(genes):
    info, classif, categories, chi, bins, f = GMMchi.GMM_modelingt(
        gene, input_dataf, log2transform=True, filt=6.5924, meanf=5.14, stdf=1.01
    )
    print(f"Categories for {gene}: {categories}")  # Debugging categories
    categorize.append(categories)
    del classif, categories, chi  # free up memory

categorized_df = pd.DataFrame(categorize, index=input_dataf.index, columns=input_dataf.columns)

print("Unique values in categorized_df:")
print(categorized_df.nunique(axis=1))  # Check the number of unique values per row

# Warn if categorized_df lacks variability
if categorized_df.nunique(axis=1).max() <= 1:
    print("Warning: categorized_df lacks variability. Ensure input data is diverse.")

# Run a 2x2 Table Analysis
print("categorized_df = ", categorized_df)
hits, significant_hits, table_sig_hits = GMMchi.find_hits(categorized_df, primary="Sample5")

# If no significant hits are found, include the full categorized_df
if significant_hits.empty or len(table_sig_hits) == 0:
    print("No significant hits found. Saving categorized_df to test.csv.")
    categorized_df.to_csv(r'test.csv')
    print("Saved test.csv with full categorized_df.")
    exit()

# Create the 2x2 table from significant hits
twobytwo_table = pd.DataFrame(
    table_sig_hits,
    columns=['+/+', '+/-', '-/+', '-/-', 'p-value', 'R value', 'Inclusion Criterion'],
    index=significant_hits.T.columns[1:]
)

# Debug: Print the 2x2 table
print("twobytwo_table:")
print(twobytwo_table)

# Save the table to a CSV file
twobytwo_table.sort_values('R value', ascending=False).to_csv(r'test.csv')
print("Saved test.csv successfully.")
