import argparse
import pandas as pd
import os
import numpy as np
import time
import subprocess
from functions import read_zip_to_dataframes, tab, tab1, HLApredict

DEFAULT_ZIP_URL = "https://www.dropbox.com/scl/fi/k1pz9m4jtl0gg0yc8nyg8/sampleExport.2024-02-29_20-38-59.zip?rlkey=z2ebf4c963rez4l46rvmlaw7d&st=yf24ehko&dl=1"


def maybe_download_zip(zip_path, url):
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_path} from {url} ...")
        subprocess.run(["wget", "-O", zip_path, url], check=True)
        print("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Run HLApredict pipeline on a .zip file of repertoires.")
    parser.add_argument('--zip_path', type=str, default='public/towlerton.zip',
                        help='Path to input zip file (default: public/towlerton.zip)')
    parser.add_argument('--download', action='store_true',
                        help='Download the default towlerton.zip zip file if not present in data folder')
    parser.add_argument('--zip_url', type=str, default=DEFAULT_ZIP_URL,
                        help=f'URL to download zip file (default: {DEFAULT_ZIP_URL})')
    parser.add_argument('--model_folder', type=str, default='public/XSTUDY_ALL_FEATURE_L1_v4e',
                        help='Path to model folder (default: public/XSTUDY_ALL_FEATURE_L1_v4e)')
    parser.add_argument('--model_name', type=str, default='XSTUDY_ALL_FEATURE_L1_v4e',
                        help='Model name (default: XSTUDY_ALL_FEATURE_L1_v4e)')
    parser.add_argument('--model_calibration', type=str, default='XSTUDY_ALL_FEATURE_L1_v4e_HS2',
                        help='Model calibration name (default: XSTUDY_ALL_FEATURE_L1_v4e_HS2)')
    parser.add_argument('--truth_file', type=str, default='public/XSTUDY_ALL_FEATURE_L1_v4e/sample_hla_x_towlerton.csv',
                        help='Path to truth file (default: public/XSTUDY_ALL_FEATURE_L1_v4e/sample_hla_x_towlerton.csv)')
    parser.add_argument('--output_folder', type=str, default='outputs',
                        help='Folder to write outputs (default: outputs)')
    parser.add_argument('--n', type=int, default=5,
                        help='Maximum number of files to process from zip (default: 5)')
    parser.add_argument('--run', type=str, default='towlerton_demo10',
                        help='Run name for output files (default: towleron_demo10)')
    parser.add_argument('--v_col', type=str, default='vMaxResolved',
                        help='V column name (default: vMaxResolved)')
    parser.add_argument('--cdr3_col', type=str, default='aminoAcid',
                        help='CDR3 column name (default: aminoAcid)')
    parser.add_argument('--templates_col', type=str, default='count (templates/reads)',
                        help='Templates column name (default: count (templates/reads))')
    args = parser.parse_args()

    # Download if requested or file missing
    if args.download or not os.path.exists(args.zip_path):
        os.makedirs(os.path.dirname(args.zip_path), exist_ok=True)
        maybe_download_zip(args.zip_path, args.zip_url)

    start_time = time.time()
    print(">>>> Loading and parsing repertoires from .zip file <<<<<")
    dfs = read_zip_to_dataframes(args.zip_path, args.v_col, args.cdr3_col, args.templates_col, n=args.n)

    Q = pd.read_csv(os.path.join(args.model_folder, f"{args.model_name}.query.csv"))
    Q1 = Q[Q['search'] == "edit1"]
    Q0 = Q[Q['search'] == "edit0"]
    results0 = []
    results1 = []
    keys = list(dfs.keys())
    dataframes = list(dfs.values())

    for fp, df in zip(keys, dataframes):
        print(f">>>> Tabulating exact features in : {fp}")
        result = tab(dx=df, fp=fp, query=Q0, get_col="templates", on="vfamcdr3")
        results0.append(result)
    X0 = pd.concat(results0, axis=1)

    for fp, df in zip(keys, dataframes):
        print(f">>>> Tabulating inexact matches around anchors in : {fp}")
        result = tab1(dx=df, fp=fp, query=Q1, get_col="templates", on="vfamcdr3")
        results1.append(result)
    X1 = pd.concat(results1, axis=1)

    print(f">>>> Combining subvectors x0,x1 <<<<<{fp}")
    X = pd.concat([X1, X0], axis=0, ignore_index=True)
    Xt = (X > 0).transpose().astype('int64')
    assert X.shape[0] == Q.shape[0]
    I = pd.DataFrame()
    I['U'] = pd.Series({k.replace(".tsv", "").replace(".csv", ""): x.shape[0] for k, x in dfs.items()})
    I['log10unique'] = np.log10(I['U'])
    I['log10unique2'] = np.log10(I['U']) ** 2

    print(f">>>> Loading Model <<<<<{fp}")
    h = HLApredict(Q=Q)
    h.load_fit(model_folder=args.model_folder, model_name=args.model_name)
    h.load_calibrations(model_folder=args.model_folder, model_name=args.model_calibration)
    h.predict_decisions_x(Xt)
    h.pxx(decision_scores=h.decision_scores, variables=None, covariates=I[['log10unique', 'log10unique2']])

    print(f">>>> Loading Truth File <<<<<{fp}")
    if args.truth_file is not None and os.path.exists(args.truth_file):
        truth = pd.read_csv(args.truth_file, index_col=0)
        predictions = h.output_probs_and_obs(probs=h.calibrated_prob, observations=truth.loc[h.calibrated_prob.index])
        predictions['pred'] = predictions['p'] > 0.5
        print(f">>>> Scoring Predictions <<<<<{fp}")
        performance = h.score_predictions(probs=h.calibrated_prob, observations=truth.loc[h.calibrated_prob.index].astype('float64'))
        performance = performance.rename(columns={'i': 'binary'})
        print(f">>>> Writing Outputs <<<<<{fp}")
        os.makedirs(args.output_folder, exist_ok=True)
        h.calibrated_prob.to_csv(os.path.join(args.output_folder, f"{args.run}.calibrated_probabilities.csv"), index=True)
        predictions.to_csv(os.path.join(args.output_folder, f"{args.run}.predictions.csv"), index=True)
        performance.to_csv(os.path.join(args.output_folder, f"{args.run}.performance.csv"), index=True)
        print(performance)
    else:
        print(f">>>> Writing Outputs <<<<<{fp}")
        os.makedirs(args.output_folder, exist_ok=True)
        h.calibrated_prob.to_csv(os.path.join(args.output_folder, f"{args.run}.calibrated_probabilities.csv"), index=True)
        predictions = h.output_probs_and_obs(probs=h.calibrated_prob, observations=(h.calibrated_prob > 0.5).astype('float64'))
        predictions = predictions.rename(columns={'obs': 'pred'})  # no observations
        predictions.to_csv(os.path.join(args.output_folder, f"{args.run}.predictions.csv"), index=True)

    end_time = time.time()
    print(f"HLApredict pipeline completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()