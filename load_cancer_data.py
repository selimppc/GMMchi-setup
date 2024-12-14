import GMMchi
import numpy as np
import pandas as pd

# Load data from CSV
input_data_cancer = pd.read_csv(
    r'cancer_data.csv',  # Path to the CSV file
    index_col=[0],       # Set the first column as the index
    header=0,            # Use the first row as the header
    na_values='---'      # Treat '---' as NaN
)

# # Print the loaded data
# print("Loaded Cancer Data:")
print(input_data_cancer)

# #calculating background threshold

means, std, filt = GMMchi.GMM_modelingt('TCGA Colorectal Cancer', input_data_cancer, log2transform=True, verbosity = True, Single_tail_validation=False, calc_back = True)

print(means)
print(std)
print(filt)


# # Filter and Remove Non-expressing Genes

# input_dataf = GMMchi.probe_filter(input_data_cancer, log2transform=True, filt=-0.829)
# print(input_dataf)

