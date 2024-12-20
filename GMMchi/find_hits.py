from tqdm import tqdm
import pandas as pd
from scipy import stats

def find_hits(ip, primary, verbose=True, correction=True):
    """
    This function calculates the chi-square contingency table of a single gene
    or probe with all other genes or probes.

    ip: Input subcategorized data with 1 or 2s.
    primary: The probe or gene used to calculate chi-square contingency table with all other genes.

    Returns:
     - p-value of all matches
     - p-value <= 0.05 for all matches
     - table of the 2x2 table, p-value, r-value
    """
    p_val = []
    odds_ratio = []

    # Transpose the input and replace values
    ipa = ip.copy().T
    ipa.replace(3, 2, inplace=True)  # Replace 3s with 2s

    # Ensure the primary column is retained
    if primary not in ipa.columns:
        ipa[primary] = ip.loc[:, primary] if primary in ip.columns else 2  # Default to 2 if missing

    # Retain primary even if it does not meet the filtering criteria
    ipa = ipa.loc[:, (ipa != 2).any(axis=0) | (ipa.columns == primary)]

    # Debug: Print the filtered DataFrame
    print("Filtered ipa DataFrame:")
    print(ipa)

    # Ensure primary exists
    if primary not in ipa.columns:
        raise ValueError(f"Primary column '{primary}' is missing from the input data. Ensure it exists.")

    # Process each column
    for x in tqdm(ipa.columns):
        # Filter out NaN values
        ipan = ipa[~ipa[x].isna()]

        # Ensure primary exists in the filtered DataFrame
        if primary not in ipan.columns:
            print(f"Primary column '{primary}' is missing in ipan. Skipping {x}.")
            continue

        # Debug: Print contingency table inputs
        print(f"Processing contingency table for {primary} vs {x}:")
        print(f"ipan[primary]: {ipan[primary]}")
        print(f"ipan[x]: {ipan[x]}")

        try:
            # Perform Fisher's exact test
            o, p = stats.fisher_exact(pd.crosstab(ipan[primary], ipan[x]))
            p_val.append(p)
            odds_ratio.append(o)
        except Exception as e:
            print(f"Error in Fisher's exact test for {primary} vs {x}: {e}")
            p_val.append(1)
            odds_ratio.append(1)

    # Create a DataFrame for p-values
    new = pd.DataFrame({'P-value': p_val}, index=ipa.columns).sort_values('P-value', ascending=True)

    # Apply correction for multiple comparisons
    if correction:
        threshold = 0.05 / len(ipa.columns)  # Bonferroni correction
        filtnew = new[new['P-value'] < threshold]
    else:
        filtnew = new[new['P-value'] < 0.05]

    # Debug: Print filtered significant hits
    print("Filtered significant hits (filtnew):")
    print(filtnew)

    if filtnew.empty:
        print("No significant hits found. Returning empty results.")
        return new, filtnew, []

    # Drop the primary gene from index if it exists
    if primary in filtnew.index:
        index = filtnew.index.drop(primary)
    else:
        print(f"Warning: {primary} not found in filtnew index. Skipping.")
        index = filtnew.index

    # Initialize contingency table results
    ct = []

    # Process significant hits
    for x in index:
        ipan = ipa[~ipa[x].isna()]  # Filter NaN values
        if primary not in ipan.columns:
            continue

        o, p = stats.fisher_exact(pd.crosstab(ipan[primary], ipan[x]))
        first = ipan[primary].dropna()

        # Pearson correlation
        try:
            r, pp = stats.pearsonr(first, ipan.loc[first.index, x])
        except Exception as e:
            print(f"Error calculating Pearson correlation for {primary} vs {x}: {e}")
            r, pp = 0, 1

        # Extract values from the contingency table
        values = [y for x in pd.crosstab(ipan[primary], ipan[x]).values for y in x]
        ct.append([values[3], values[2], values[1], values[0], p, r, inclusion_criterion(values, r)])

        if verbose:
            print(pd.crosstab(ipan[primary], ipan[x]))
            print(f'P-value: {p}\n')
            print(f'R-value: {r}\n')

    return new, filtnew, ct
