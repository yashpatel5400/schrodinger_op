import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

import constants

# Legacy filenames (noise=0 fallback)
ESTIMATOR_FILES_LEGACY = {
    "FNO"     : "fno.csv",
    "DeepONet": "onet.csv",
    "Linear"  : "linear.csv",
}

def file_for_estimator(results_dir: str, estimator_key: str, sigma: float) -> str:
    noise_tag = f"noise{sigma:.3g}"
    name_map = {
        "FNO"     : f"fno_{noise_tag}.csv",
        "DeepONet": f"onet_{noise_tag}.csv",
        "Linear"  : f"linear_{noise_tag}.csv",
    }
    candidate = os.path.join(results_dir, name_map[estimator_key])
    if os.path.exists(candidate):
        return candidate
    if sigma == 0.0:
        legacy = os.path.join(results_dir, ESTIMATOR_FILES_LEGACY[estimator_key])
        if os.path.exists(legacy):
            return legacy
    raise FileNotFoundError(f"Missing results for estimator={estimator_key}, sigma={sigma} in {results_dir}")

def build_table_for_sigma(sigma: float) -> pd.DataFrame:
    potentials = sorted(
        d for d in os.listdir(constants.results_dir)
        if os.path.isdir(os.path.join(constants.results_dir, d))
    )

    # pretty formatter: mean (std) in sci notation
    fmt = lambda x: f"{np.format_float_scientific(np.mean(x), precision=3)} ({np.format_float_scientific(np.std(x), precision=3)})"

    table = {}
    for potential in potentials:
        results_dir = os.path.join(constants.results_dir, potential)
        is_spherical = potential in ("coulomb", "dipole")
        estimators = ["Linear", "FNO"] if is_spherical else ["Linear", "FNO", "DeepONet"]

        # load results
        data = {}
        for est in estimators:
            fn = file_for_estimator(results_dir, est, sigma)
            arr = pd.read_csv(fn).values
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            data[est] = arr

        row = {}
        # error cells (bold Linear)
        for est in estimators:
            cell = fmt(data[est])
            row[est] = r"\textbf{" + cell + r"}" if est == "Linear" else cell

        # t-tests (paired, one-sided: Linear < Est)
        if "FNO" in estimators:
            _, p = ttest_rel(data["Linear"], data["FNO"], alternative="less")
            row[r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{FNO})}_{\text{err}}$"] = np.format_float_scientific(p, precision=3)
        if "DeepONet" in estimators:
            _, p = ttest_rel(data["Linear"], data["DeepONet"], alternative="less")
            row[r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{DeepONet})}_{\text{err}}$"] = np.format_float_scientific(p, precision=3)

        title = " ".join(w.capitalize() for w in potential.split("_"))
        table[title] = row

    df = pd.DataFrame.from_dict(table).T

    # Order columns: errors first (FNO, DeepONet, Linear), then p-values
    p_fno  = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{FNO})}_{\text{err}}$"
    p_onet = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{DeepONet})}_{\text{err}}$"
    preferred = ["FNO", "DeepONet", "Linear", p_fno, p_onet]
    df = df[[c for c in preferred if c in df.columns]]

    # render missing as '---'
    df = df.fillna("---")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_sigma", type=float, nargs="+", default=[0.0],
                        help="One or more noise levels, e.g. --noise_sigma 0 0.01 0.05")
    parser.add_argument("--outfile_prefix", type=str, default="paper_table",
                        help="Base filename for the LaTeX table(s) saved in results_dir.")
    args = parser.parse_args()

    for sigma in args.noise_sigma:
        df = build_table_for_sigma(sigma)
        noise_tag = f"noise{sigma:.3g}"
        out_tex = os.path.join(constants.results_dir, f"{args.outfile_prefix}_{noise_tag}.tex")

        latex = df.to_latex(
            column_format="c" * (1 + len(df.columns)),
            na_rep="---",
            index=True,     # keep potentials in first column
            escape=False    # keep LaTeX math in headers
        )
        with open(out_tex, "w") as f:
            f.write(latex)
        print(f"[{noise_tag}] wrote LaTeX table -> {out_tex}")
        print(latex)

if __name__ == "__main__":
    main()
