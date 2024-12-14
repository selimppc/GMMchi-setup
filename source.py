import GMMchi
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv


data = pd.read_csv(r'cancer_data.csv',
                                index_col=[0],
                                header=[0],
                                na_values='---')

# print("input data", input_data_cancer)
gene = 'MUC5AC' #Transforming growth factor beta 1

# preprocessed_data = pd.DataFrame(
#     data={sample: [val] for sample, val in zip(input_data_cancer.columns, input_data_cancer)},
#     index=[gene]
# )


# Create a DataFrame
input_data_cancer = pd.DataFrame(data)

# Preprocess data for the selected ID
gene_data = input_data_cancer.loc[gene].dropna()
if gene_data.empty:
    raise ValueError("No valid data available for the selected ID.")

# Ensure preprocessing meets function expectations
preprocessed_data = pd.DataFrame(
    data={sample: [val] for sample, val in zip(input_data_cancer.columns, gene_data)},
    index=[gene]
)

# preprocessed_data = pd.DataFrame(input_data_cancer)

# print("preprocessed data", preprocessed_data)

# Works!
# results = GMMchi.GMM_modelingt(ID='MUC5AC', 
#                                         input_datanormal=preprocessed_data, 
#                                         log2transform=True, 
#                                         verbosity = True,
#                                         Single_tail_validation=False, 
#                                         calc_back = False,
#                                         filt=0.5,
#                                         graphs=True)

# Works!
# results = GMMchi.GMM_modelingt(ID='MUC5AC', 
#                                input_datanormal=preprocessed_data, 
#                                log2transform=True, 
#                                verbosity = True, 
#                                Single_tail_validation=False, 
#                                calc_back = False, 
#                                unimodal_categories=False)


# print("GMM_modelingt..", results)

# Works!
input_dataf = GMMchi.probe_filter(datainputnorm=preprocessed_data, 
                                  log2transform=True, 
                                  filt=-0.829)

#Categorizing the Distribution a Single Gene

info, classif, categories, chi, bins, f = GMMchi.GMM_modelingt(gene, 
                                                               input_dataf, 
                                                               log2transform = True, 
                                                               filt=-0.83, 
                                                               meanf= -3.3, 
                                                               stdf = 1.95)

#Large-scale Categorization of the Input Data (All genes)

genes = input_dataf.index #the index of the dataframe or a list of all genes
categorize = [] #append as list of list of categorized data

for gene in tqdm(genes):
    info, classif, categories, chi, bins, f = GMMchi.GMM_modelingt(gene, input_dataf, log2transform=True, filt=6.5924, meanf= 5.14, stdf = 1.01)

    categorize.append(categories)
    
    del classif, categories, chi #free up memory
   
categorized_df = pd.DataFrame(categorize, index = input_dataf.index, columns = input_dataf.columns)


#Run a 2x2 Table Analysis

print("categorized_df = ", categorized_df)
hits, significant_hits, table_sig_hits = GMMchi.find_hits(categorized_df, primary=gene)

twobytwo_table = pd.DataFrame(table_sig_hits, columns = ['+/+', '+/-', '-/+', '-/-', 'p-value', 'R value', 'Inclusion Criterion'], index = significant_hits.T.columns[1:])

#save your 2x2 table for further analysis
twobytwo_table.sort_values('R value', ascending=False).to_csv(r'2by2table_tgfb1.csv')
