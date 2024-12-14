import GMMchi
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Expanded dataset with more samples to ensure robust calculations
data = {
    "Sample1": [1.2, 2.3, 3.1, 0.5],
    "Sample2": [2.1, 1.8, 3.4, 0.8],
    "Sample3": [1.5, 2.2, 2.9, 1.0],
    "Sample4": [2.0, 3.0, 4.0, 0.7],
    "Sample5": [1.8, 2.5, 3.5, 0.6]
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
    info, classif, categories, chi, bins, f = GMMchi.GMM_modelingt(gene, input_dataf, log2transform=True, filt=6.5924,
                                                                   meanf=5.14, stdf=1.01)

    categorize.append(categories)

    del classif, categories, chi  # free up memory

categorized_df = pd.DataFrame(categorize, index=input_dataf.index, columns=input_dataf.columns)

# Run a 2x2 Table Analysis
print("categorized_df = ", categorized_df)
hits, significant_hits, table_sig_hits = GMMchi.find_hits(categorized_df, primary="Sample5")

twobytwo_table = pd.DataFrame(table_sig_hits,
                              columns=['+/+', '+/-', '-/+', '-/-', 'p-value', 'R value', 'Inclusion Criterion'],
                              index=significant_hits.T.columns[1:])

# Save your 2x2 table for further analysis
twobytwo_table.sort_values('R value', ascending=False).to_csv(r'test.csv')
