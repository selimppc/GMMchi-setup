import GMMchi
import pandas as pd

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
    graphs=False
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
