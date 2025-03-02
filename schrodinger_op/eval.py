import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

import constants

def main():
    potentials = sorted(os.listdir(constants.results_dir))
    paper_results_table = {}

    for potential in potentials:
        results_dir = os.path.join(constants.results_dir, potential)
        estimator_to_result_fn = {
            "FNO"     : "fno.csv",
            "DeepONet": "onet.csv",
            "Linear"  : "linear.csv",
        }
        estimator_to_results = {}
        for estimator in estimator_to_result_fn:
            results = pd.read_csv(os.path.join(results_dir, estimator_to_result_fn[estimator]), index_col=0)
            estimator_to_results[estimator] = results.values

        estimator_col = lambda x : f"{np.format_float_scientific(x.mean(), precision=3)} ({np.format_float_scientific(x.std(), precision=3)})"
        paper_results_row = {}
        for estimator in estimator_to_result_fn:
            paper_results_row[estimator] = estimator_col(estimator_to_results[estimator])
            if estimator != "Linear":
                _, p_value = ttest_rel(estimator_to_results["Linear"], estimator_to_results[estimator], alternative='less')
                paper_results_row[r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{" + estimator + r"})}_{\text{err}}$"] = np.format_float_scientific(p_value, precision=3)
            else:
                paper_results_row[estimator] = r"\textbf{" + paper_results_row[estimator] + r"}"
        
        potential_row_title = " ".join([word.capitalize() for word in potential.split("_")])
        paper_results_table[potential_row_title] =  paper_results_row

    paper_results_df = pd.DataFrame.from_dict(paper_results_table).T
    paper_results_df = paper_results_df[[paper_results_df.columns[0],paper_results_df.columns[2],paper_results_df.columns[4],paper_results_df.columns[1],paper_results_df.columns[3]]] 
    print(paper_results_df.to_latex(column_format="c" * (1 + len(paper_results_df.columns))))

if __name__ == "__main__":
    main()