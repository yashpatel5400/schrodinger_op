import argparse
import os
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import ttest_rel

import constants


def _fmt_sci_mean_std(x: np.ndarray) -> str:
    return f"{np.format_float_scientific(np.mean(x), precision=3)} ({np.format_float_scientific(np.std(x), precision=3)})"


def _read_results_csv(path: str) -> np.ndarray:
    arr = pd.read_csv(path).values
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _file_for_estimator(results_dir: str, estimator_key: str, sigma: float, mask: float) -> str:
    core_map = {"FNO": "fno", "UNO": "uno", "DeepONet": "onet", "Linear": "linear"}
    core = core_map[estimator_key]
    noise_tag = f"noise{sigma:.3g}"
    mask_tag  = "" if mask == 0.0 else f"mask{mask:.3g}"
    fn = f"{core}_{noise_tag}_{mask_tag}.csv"
    path = os.path.join(results_dir, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {estimator_key}: {fn}")
    return path


def _build_rows_for_potential(results_dir: str, potential: str, sigma: float, mask: float):
    is_spherical = potential in ("coulomb", "dipole")
    # Euclidean: Linear, FNO, UNO, DeepONet
    # Spherical: Linear, FNO
    estimators = ["Linear", "FNO"] if is_spherical else ["Linear", "FNO", "UNO", "DeepONet"]

    est_results = {}
    for est in estimators:
        fn = _file_for_estimator(results_dir, est, sigma, mask)
        est_results[est] = _read_results_csv(fn)

    # Values row (means ± stds), bold Linear
    values_row = {}
    for est in estimators:
        cell = _fmt_sci_mean_std(est_results[est])
        values_row[est] = r"\textbf{" + cell + r"}" if est == "Linear" else cell

    # Tests row: p-values for Linear < Estimator
    tests_row = {}
    for est in estimators:
        if est == "Linear":
            continue
        _, p = ttest_rel(est_results["Linear"], est_results[est], alternative="less")
        key = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{" + est + r"})}_{\text{err}}$"
        tests_row[key] = np.format_float_scientific(p, precision=3)

    return values_row, tests_row, estimators


def build_tables_for_config(sigma: float, mask: float):
    potentials = sorted(
        d for d in os.listdir(constants.results_dir)
        if os.path.isdir(os.path.join(constants.results_dir, d))
    )

    values_table = {}
    tests_table = {}

    # Collect rows
    for pot in potentials:
        results_dir = os.path.join(constants.results_dir, pot)
        values_row, tests_row, estimators = _build_rows_for_potential(results_dir, pot, sigma, mask)
        pot_title = " ".join(w.capitalize() for w in pot.split("_"))
        values_table[pot_title] = values_row
        tests_table[pot_title] = tests_row

    # Assemble DataFrames
    df_values = pd.DataFrame.from_dict(values_table).T
    df_tests  = pd.DataFrame.from_dict(tests_table).T

    # Column orders
    # Values: errors first (FNO, UNO, DeepONet, Linear) — keep only existing
    values_order = ["FNO", "UNO", "DeepONet", "Linear"]
    df_values = df_values[[c for c in values_order if c in df_values.columns]].fillna("---")

    # Tests: p-values in same order
    p_fno  = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{FNO})}_{\text{err}}$"
    p_uno  = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{UNO})}_{\text{err}}$"
    p_onet = r"$\%^{(\text{lin})}_{\text{err}} < \%^{(\text{DeepONet})}_{\text{err}}$"
    tests_order = [p_fno, p_uno, p_onet]
    df_tests  = df_tests[[c for c in tests_order if c in df_tests.columns]].fillna("---")

    return df_values, df_tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_sigma", type=float, nargs="+", default=[0.0],
                        help="Noise levels to aggregate, e.g. --noise_sigma 0 0.01")
    parser.add_argument("--mode_zero_frac", type=float, nargs="+", default=[0.0],
                        help="Mask fractions to aggregate, e.g. --mode_zero_frac 0 0.2")
    parser.add_argument("--outfile_prefix", type=str, default="paper_table",
                        help="Base filename for LaTeX tables in results_dir.")
    args = parser.parse_args()

    for sigma, mask in product(args.noise_sigma, args.mode_zero_frac):
        df_vals, df_tests = build_tables_for_config(sigma, mask)
        noise_tag = f"noise{sigma:.3g}"
        mask_tag  = f"mask{mask:.3g}"

        # Values table
        out_vals = os.path.join(constants.results_dir, f"{args.outfile_prefix}_values_{noise_tag}_{mask_tag}.tex")
        latex_vals = df_vals.to_latex(
            column_format="c" * (1 + len(df_vals.columns)),
            na_rep="---",
            index=True,
            escape=False
        )
        with open(out_vals, "w") as f:
            f.write(latex_vals)
        print(f"[{noise_tag} {mask_tag}] wrote VALUES -> {out_vals}")
        print(latex_vals)

        # Tests table
        # out_tests = os.path.join(constants.results_dir, f"{args.outfile_prefix}_tests_{noise_tag}_{mask_tag}.tex")
        # latex_tests = df_tests.to_latex(
        #     column_format="c" * (1 + len(df_tests.columns)),
        #     na_rep="---",
        #     index=True,
        #     escape=False
        # )
        # with open(out_tests, "w") as f:
        #     f.write(latex_tests)
        # print(f"[{noise_tag} {mask_tag}] wrote TESTS  -> {out_tests}")
        # print(latex_tests)


if __name__ == "__main__":
    main()
