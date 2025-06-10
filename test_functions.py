import pandas as pd
import os
import numpy as np
from functions import read_zip_to_dataframes, tab, tab1
from functions import HLApredict
import scipy 
import time

def test_HLApredict():
    """
    Test the HLApredict class initialization and prediction.
    """
    start_time = time.time()  # Start timer
    # ARGUMENTS
    n = 5 # maximum number of files to run in test
    zip_path = os.path.join('data', 'towlerton.zip')
    v_col = 'vMaxResolved'
    cdr3_col = 'aminoAcid'
    templates_col = 'count (templates/reads)'
    run= "towlerton_demo10"
    model_folder      = 'XSTUDY_ALL_FEATURE_L1_v4e'
    model_name        = 'XSTUDY_ALL_FEATURE_L1_v4e'
    model_calibration = 'XSTUDY_ALL_FEATURE_L1_v4e_HS2'
    truth_file = os.path.join('data','XSTUDY_ALL_FEATURE_L1_v4e','sample_hla_x_towlerton.csv')
    output_folder = os.path.join('outputs')

    #Script
    print(">>>> Loading and parsing repertoires form .zip file <<<<<")
    dfs = read_zip_to_dataframes(zip_path, v_col, cdr3_col, templates_col, n = n)

    model_folder = os.path.join('data', 'XSTUDY_ALL_FEATURE_L1_v4e')
    model_name = 'XSTUDY_ALL_FEATURE_L1_v4e'
    Q = pd.read_csv(os.path.join(model_folder,f"{model_name}.query.csv")) 
    zip_path = os.path.join('data', 'towlerton.zip')
    v_col = 'vMaxResolved'
    cdr3_col = 'aminoAcid'
    templates_col = 'count (templates/reads)'

    Q1 = Q[Q['search'] == "edit1"]
    Q0 = Q[Q['search'] == "edit0"]
    results0 = []
    results1 = []
    keys = list(dfs.keys())
    dataframes = list(dfs.values())

    for fp, df in zip(keys, dataframes):
        print(f">>>> Tabulating exact features in : {fp}")
        result = tab(dx = df, fp = fp, query =Q0, get_col="templates", on="vfamcdr3")
        results0.append(result)
    X0 = pd.concat(results0, axis = 1)


    for fp, df in zip(keys, dataframes):
        print(f">>>> Tabulating inexact matches around anchors in : {fp}")
        result = tab1(dx = df, fp = fp, query =Q1, get_col="templates", on="vfamcdr3")
        results1.append(result)
    X1 = pd.concat(results1, axis = 1)

    print(f">>>> Combining subvectors x0,x1 <<<<<{fp}")
    X = pd.concat([X1,X0], axis = 0, ignore_index=True)
    Xt = (X>0).transpose().astype('int64')
    assert X.shape[0] == Q.shape[0]
    I = pd.DataFrame()
    I['U'] = pd.Series({k.replace(".tsv","").replace(".csv",""):x.shape[0] for k,x in dfs.items()})
    I['log10unique'] = np.log10(I['U']) 
    I['log10unique2'] = np.log10(I['U'])**2
    
    print(f">>>> Loading Model <<<<<{fp}")
    h = HLApredict(Q = Q)
    h.load_fit(model_folder = model_folder, model_name=model_name)
    h.load_calibrations(model_folder = model_folder, model_name=f"{model_name}_HS2")
    h.predict_decisions_x(Xt)
    h.pxx( decision_scores = h.decision_scores, variables=None, covariates=I[['log10unique','log10unique2']])
    
    print(f">>>> Loading Truth File <<<<<{fp}")
    if truth_file is not None:
        truth = pd.read_csv(truth_file , index_col = 0)
        predictions = h.output_probs_and_obs(probs = h.calibrated_prob, 
                           observations =truth.loc[h.calibrated_prob.index] )
        predictions['pred'] = predictions['p'] > 0.5
        print(f">>>> Scoring Predictions <<<<<{fp}")
        performance = h.score_predictions(probs =  h.calibrated_prob, observations =truth.loc[h.calibrated_prob.index].astype('float64'))
        performance = performance.rename(columns = {'i':'binary'})
        print(f">>>> Writing Outputs <<<<<{fp}")        
        print(os.path.join(output_folder,f"{run}.calibrated_probabilities.csv"))
        print(os.path.join(output_folder,f"{run}.predictions.csv"))
        print(os.path.join(output_folder,f"{run}.performance.csv"))
        h.calibrated_prob.to_csv(os.path.join(output_folder,f"{run}.calibrated_probabilities.csv"), index = True)
        
        predictions.to_csv(os.path.join(output_folder,f"{run}.predictions.csv"), index = True)
        performance.to_csv(os.path.join(output_folder,f"{run}.performance.csv"), index = True)
        print(performance)
    else:
        print(f">>>> Writing Outputs <<<<<{fp}")        
        print(os.path.join(output_folder,f"{run}.calibrated_probabilities.csv"))
        print(os.path.join(output_folder,f"{run}.predictions.csv"))
        print(os.path.join(output_folder,f"{run}.performance.csv"))
        h.calibrated_prob.to_csv(os.path.join(output_folder,f"{run}.calibrated_probabilities.csv"), index = True)
        
        predictions = h.output_probs_and_obs(probs = h.calibrated_prob, 
                           observations = (h.calibrated_prob > 0.5).astype('float64'))
        predictions = predictions.rename(columns = {'obs':'pred'}) # no observations
        predictions.to_csv(os.path.join(output_folder,f"{run}.predictions.csv"), index = True)

    end_time = time.time()  # End timer
    print(f"test_HLApredict completed in {end_time - start_time:.2f} seconds.")

def test_read_zip_to_dataframes():
    zip_path = os.path.join('data', 'towlerton.zip')
    v_col = 'vMaxResolved'
    cdr3_col = 'aminoAcid'
    templates_col = 'count (templates/reads)'
    dfs = read_zip_to_dataframes(zip_path, v_col, cdr3_col, templates_col, n = 2)
    assert isinstance(dfs, dict)
    assert all(isinstance(df, pd.DataFrame) for df in dfs.values())
    for fp,df in dfs.items():
        assert list(df.columns) == ['vfamcdr3','templates']
        assert isinstance(fp,str)
        assert not df.empty  # Optionally check that DataFrames are not empty

def test_tab_on_zip_dataframes():
    """
    Test applying the `tab` function to each DataFrame from `read_zip_to_dataframes`.
    """
    zip_path = os.path.join('data', 'towlerton.zip')
    v_col = 'vMaxResolved'
    cdr3_col = 'aminoAcid'
    templates_col = 'count (templates/reads)'
    dfs = read_zip_to_dataframes(zip_path, v_col, cdr3_col, templates_col,  n = 1)

    # dfs is a dict: keys are filenames, values are DataFrames
    keys = list(dfs.keys())
    dataframes = list(dfs.values())

    # Use the first dataframe as the query
    query_df = dataframes[0]
    results = []
    for fp, df in zip(keys, dataframes):
        result = tab(dx = df, fp = fp, query = query_df, get_col="templates", on="vfamcdr3")
        results.append(result)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == query_df.shape[0]
    results1 = []
    for fp, df in zip(keys, dataframes):
        print(f"Tabulating around anchors in {fp}")
        result = tab1(dx = df, fp = fp, query = query_df, get_col="templates", on="vfamcdr3")
        results1.append(result)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == query_df.shape[0]
    print("tab applied successfully to all dataframes in zip.")



if __name__ == "__main__":
    test_HLApredict()
    #test_tab_on_zip_dataframes()
    test_read_zip_to_dataframes()
    print("All tests passed.")