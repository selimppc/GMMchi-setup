from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

def find_hits(ip, primary, verbose=True, correction=True):
    """ This function is used to take in pre-computed subcategorized data and calculate the chi-square contingency table
    of a single gene or probe with all other genes or probes.

    ip: Input subcategorized data with 1 or 2s.
    primary: The probe or gene used to calculate chi-square contingency table with all other genes.

    Returns:
     - p-value of all matches
     - p-value <= 0.05 for all matches
     - table of the 2x2 table, p-value, r-value
    """
    p_val = []
    odds_ratio = []
    ipa = ip.copy().T
    ipa.replace(3, 2, inplace=True)  # Replace string with integer
    ipa = ipa.loc[:, (ipa != 2).any(axis=0)]  # Drop columns with all 2s

    for x in tqdm(ipa.columns):  # Loop through columns
        ipan = ipa[ipa[x] != np.nan]  # Remove rows with NaN

        try:
            o, p = stats.fisher_exact(pd.crosstab(ipan[primary], ipan[x]))
            p_val.append(p)
            odds_ratio.append(o)
        except Exception as e:
            p_val.append(1)
            odds_ratio.append(1)

    new = pd.DataFrame(
        {'P-value': p_val}, index=ipa.columns).sort_values('P-value', ascending=True)

    if correction:
        filtnew = new[(new < (0.05 / len(ip)))['P-value']]
    else:
        filtnew = new[(new < 0.05)['P-value']]

    # Ensure primary is included if missing
    if primary not in filtnew.index and primary in new.index:
        filtnew.loc[primary] = new.loc[primary]

    if filtnew.empty:
        print("No significant hits found. Returning empty results.")
        return new, filtnew, []

    # Drop the primary gene from index if it exists
    if primary in filtnew.index:
        index = filtnew.index.drop(primary)
    else:
        print(f"Warning: {primary} not found in filtnew index. Skipping.")
        index = filtnew.index

    ipa = ip.copy().T
    ct = []

    for x in index:
        ipan = ipa[~np.isnan(ipa[x])]
        o, p = stats.fisher_exact(pd.crosstab(ipan[primary], ipan[x]))
        first = ipan[primary].dropna()

        r, pp = stats.pearsonr(first, ipan.loc[first.index, x])

        values = [y for x in pd.crosstab(ipan[primary], ipan[x]).values for y in x]
        ct.append([values[3], values[2], values[1], values[0], p, r, inclusion_criterion(values, r)])

        if verbose:
            print(pd.crosstab(ipan[primary], ipan[x]))
            print(f'P-value: {p}\n')
            print(f'R-value: {r}\n')

    return new, filtnew, ct
