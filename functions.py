import os
import sys

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz, dok_matrix
from io import BytesIO
import itertools
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import copy
import zipfile



def parse_dataframe(df,
    v_col = 'vMaxResolved',
    cdr3_col = 'aminoAcid',
    templates_col = 'count (templates/reads)'):
    """
    Retain only the specified columns in the DataFrame and perform filtering/processing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    v_col : str
        Name of the V column.
    cdr3_col : str
        Name of the CDR3 column.
    templates_col : str
        Name of the templates column.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the specified columns, filtered and processed.
    """
    dfx = df[[v_col, cdr3_col, templates_col]].copy()
    dfx = dfx[dfx[v_col].apply(lambda x : isinstance(x, str))].reset_index(drop = True)
    dfx = dfx[dfx[cdr3_col].apply(lambda x : isinstance(x, str))].reset_index(drop = True)
    dfx = dfx[dfx[cdr3_col].apply(lambda x : x.find("*") == -1)].reset_index(drop = True)
    dfx = dfx[dfx[cdr3_col].apply(lambda x : x.find("-") == -1)].reset_index(drop = True)         
    dfx['V'] = dfx[v_col].str.split("*").str[0].str.split("-").str[0].str.replace("TCRB","").str.replace("TRB","")
    dfx['V'] = dfx['V'].apply(lambda x : f"{x[0]}0{x[1]}" if len(x) == 2 else x)
    dfx['vfamcdr3'] = dfx['V'] + dfx[cdr3_col]
    dfx['templates'] = dfx[templates_col]
    return dfx[['vfamcdr3', 'templates']]

def read_zip_to_dataframes(zip_path, 
                           v_col,
                           cdr3_col, 
                           templates_col, 
                           n=None, 
                           sep="\t", 
                           **read_csv_kwargs):
    """
    Reads each file in a zip archive into a pandas DataFrame, retains only specified columns,
    and returns a list of DataFrames.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    v_col : str
        Name of the V column.
    cdr3_col : str
        Name of the CDR3 column.
    templates_col : str
        Name of the templates column.
    n : int, optional
        Maximum number of files to read from the zip archive. If None, read all files.
    sep : str, optional
        Delimiter to use for reading the files. Default is tab.
    **read_csv_kwargs
        Additional keyword arguments to pass to `pd.read_csv`.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames, one for each file in the zip, with only specified columns.
    """
    dataframes = dict()
    cnt = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('.tsv') or filename.endswith('.csv'):
                cnt = cnt + 1
                print(filename)
                with z.open(filename) as f:
                    df = pd.read_csv(BytesIO(f.read()), sep=sep, **read_csv_kwargs)
                    parsed_df = parse_dataframe(df, v_col, cdr3_col, templates_col)
                    dataframes[filename] = parsed_df
                if n is not None:
                    if cnt > n:
                        break
    return dataframes


def tab(dx, fp, query, sep = ",", get_col = "templates", on = 'vfamcdr3', min_value = None):
    """
    Aggregate template counts based on a grouping column.

    Parameters
    ----------
    dx: data.frame 

    fp : str
        File path to a CSV or TSV file.
    query : pandas.DataFrame
        DataFrame containing the query data.
    sep : str, optional
        Delimiter used in the input file (default is ',').
    get_col : str, optional
        Column name containing the values to aggregate (default is 'templates').
    on : str, optional
        Column name to group data by (default is 'vfamcdr3').
    min_value : int, optional
        Minimum value threshold for filtering (default is None).

    Returns
    -------
    pandas.DataFrame
        Aggregated results merged with query data.
    """
    #dx = pd.read_csv(fp, sep = sep)
    if min_value is not None:
        dx = dx[dx[get_col] > min_value].reset_index(drop = True)
    
    dxt = dx.groupby(on).sum().reset_index(drop = False)
    #print(dxt.head())
    #print(query.head())
    result = query[[on]].merge(dxt, how = "left", on = on).fillna(0)
    assert result.shape[0] == query.shape[0]
    name = os.path.basename(fp).replace(".tsv","").replace(".csv","")
    rt = pd.DataFrame({name: result[get_col]})
    return rt

def tab1(dx, fp, query, sep = ",", get_col = "templates",on = 'vfamcdr3', min_value = None, enforcement = True):
    """
    Aggregate template counts based on a grouping column with optional enforcement rules.

    Parameters
    ----------
    fp : str
        File path to a CSV or TSV file.
    query : pandas.DataFrame
        DataFrame containing the query data.
    sep : str, optional
        Delimiter used in the input file (default is ',').
    get_col : str, optional
        Column name containing the values to aggregate (default is 'templates').
    on : str, optional
        Column name to group data by (default is 'vfamcdr3').
    min_value : int, optional
        Minimum value threshold for filtering (default is None).
    enforcement : bool, optional
        Whether to apply enforcement rules for Vfam consistency upon identical CDR3 (default is True).

    Returns
    -------
    pandas.DataFrame
        Aggregated results merged with query data, with enforcement applied if enabled.
    """
    # make sure we aren't overwriting anything when we apply enforcement
    query1 = query.copy()
    #dx = pd.read_csv(fp, sep = sep)
    if min_value is not None:
        dx = dx[dx[get_col] > min_value].reset_index(drop = True)
    dxt = dx.groupby(on).sum().reset_index(drop = False)
    
    # enforcement uses a simplyrick to avoid different V exact CDR by converting V02CAS to V02V02CAS
    if enforcement:
        query1[on] = query1[on].str[0:3] + query1[on] # (We add extract V03)
        dxt[on] = dxt[on].str[0:3] + dxt[on]
    
    dq = get_multimer_dictionary(random_strings = query1[on], 
    trim_left = None, trim_right = None)

    ds = get_multimer_dictionary(random_strings = dxt[on], 
        trim_left = None, trim_right = None, 
        conserve_memory = False)
    
    csr_mat1 = get_query_v_subject_npz_dok(
        dq=dq, 
        ds=ds, 
        n_rows = query1.shape[0], 
        n_cols = dxt.shape[0], 
        nn_min = 0)

    n_rows = query1.shape[0]
    n_cols = dxt.shape[0]
    dk           = dok_matrix((n_rows, n_cols))
    values = dxt[get_col]
    for (i,j),_ in csr_mat1.todok().items():
        dk[(i,j)] = values.iloc[j]
    csr_mat_val = dk.tocsr()
    result = pd.Series(csr_mat_val.sum(axis=1).A1)
    name = os.path.basename(fp).replace(".tsv","").replace(".csv","")
    rt = pd.DataFrame({name: result})
    return rt

def get_multimer_dictionary(random_strings, 
                            indels = True, 
                            conserve_memory = False, 
                            ref_d = None, 
                            trim_left = None,
                            trim_right = None, 
                            verbose = False):
    """
    Generate a dictionary mapping multi-mer sequences to their original sequence 
    indices, optionally including indels.

    This function processes a list of random string sequences (e.g., T-cell receptor sequences), creating a dictionary
    that maps each sequence and its possible one-character mismatches (and optionally, one-character indels) to the indices
    of the original sequences. If `conserve_memory` is enabled and a reference dictionary is provided, it filters
    the resulting dictionary to include only keys that are also found in the reference dictionary.

    Parameters
    ----------
    random_strings : list of str
        The list of string sequences to process.
    indels : bool, optional
        Whether to include one-character insertions and deletions in the mismatches. Default is True.
    conserve_memory : bool, optional
        If True, conserves memory by keeping only the keys that are present in both `d` and `ref_d`. Requires `ref_d`
        to be not None. Default is False.
    ref_d : dict, optional
        Reference dictionary used to filter keys when conserving memory. Must be provided if `conserve_memory` is True.
        Default is None.
    trim_left : int, optional
        Number of characters to remove from the start of each sequence before processing. Default is 2.
    trim_right : int, optional
        Number of characters to remove from the end of each sequence before processing. Default is -2
    Returns
    -------
    dict
        A dictionary where each key is a sequence or a sequence with one mismatch/indel, and the value is a list of
        indices from `random_strings` where the (mis)matched sequence originated.

    Raises
    ------
    AssertionError
        If `conserve_memory` is True but `ref_d` is None.

    Notes
    -----
    The function prints progress and timing information, indicating the number of sequences processed and the total
    processing time. Memory conservation mode prints additional information about memory optimization steps.

    Examples
    --------
    >>> random_strings = ["ABCDEFGH", "ABCGEFGH", "QRSTUVWX"]
    >>> result = get_multimer_dictionary(random_strings, trim_left = 2, trim_right = -2, indels=False)
    >>> expected = {'.DEF': [0],
                     'C.EF': [0, 1],
                     'CD.F': [0],
                     'CDE.': [0],
                     '.GEF': [1],
                     'CG.F': [1],
                     'CGE.': [1],
                     '.TUV': [2],
                     'S.UV': [2],
                     'ST.V': [2],
                     'STU.': [2]}
    >>> assert result == expected
    """
   
    # OPTIONAL TRIMMING
    if trim_left is None and trim_right is not None: 
        if trim_right > 0:
            trim_right = -1*trim_right
    
        if verbose: print(f"Right trimming input sequences by {trim_right} only.")
        random_strings = [x[:trim_right] for x in random_strings]
    
    elif trim_right is None and trim_left is not None:
        if verbose: print(f"Left trimming input sequences by {trim_left} only.")
        random_strings = [x[:trim_right] for x in random_strings]
    
    elif trim_left is None and trim_right is None:
        if verbose: print("No trimming of input sequences performed.")
        pass
    else: 
        if trim_right > 0:
            trim_right = -1*trim_right
        if verbose: print(f"Left trimming input sequences by {trim_left} and rRight trimming by {trim_right}.")
        random_strings = [x[trim_left:trim_right] for x in random_strings]


    tic = time.perf_counter()
    if verbose: print(f"Finding multi-mers from {len(random_strings)} sequences, expect 1 min per million")
    d = dict()
    for i, cdr3 in enumerate(random_strings):
        if isinstance(cdr3, str):
            mm = [cdr3[:i-1] + "." + cdr3[i:] for i in range(1, len(cdr3)+1)]
            if indels:
                indels = [cdr3[:i] + "." + cdr3[i:] for i in range(1, len(cdr3)+1)]
                mm = mm + indels
            for m in mm:
                d.setdefault(m, []).append(i)
        else:
            pass
        if i % 100000 == 0:
            if conserve_memory:
                if ref_d is None:
                    raise ValueError
                if verbose: print(f"\tprocessed {i} TCRs - conserving hash memory by dumping unmatched keys")
                # Drop unneed keys
                assert ref_d is not None
                common_keys = d.keys() & ref_d.keys()
                # Create a new dictionary from dq with only the common keys
                d = {k: d[k] for k in common_keys}
    toc = time.perf_counter()
    if verbose: print(f"\tStored 1 mismatch/indel features in {len(random_strings)/1E6} M sequences in {toc - tic:0.4f} seconds")
    return d

def get_query_v_subject_npz_dok(dq, ds, n_rows, n_cols, nn_min = 0, verbose = False):
    # Detect all shared linkages between the subject and query
    # dq : query hash
    # ds : subject hash
    from scipy.sparse import dok_matrix
    # Initialize a dok_matrix
    dok = dok_matrix((n_rows, n_cols), dtype=np.dtype('u1'))
    tic = time.perf_counter()
    # <x> We only care about keys in the query
    x = [k for k,v in dq.items() if len(v) > nn_min]
    for i in x:
        ix = dq.get(i)
        jx = ds.get(i)
        if jx is not None:
            for tup in itertools.product(ix, jx):
                dok[tup] = 1
    csr_mat = dok.tocsr()
    toc = time.perf_counter()
    if verbose:
        print(f"\tConstructed sparse matrix  {toc - tic:0.4f} seconds")
    return csr_mat


class HLApredict:
    def __init__(self, Q, X=None, Y=None, cpus = 2):
        self.cpus = cpus
        self.Q = Q 
        self.X = X
        self.Y = Y 
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None

    def copy(self):
        return copy.deepcopy(self)
  
    def load_fit(self, model_folder, model_name, use_npz = True):
        if use_npz:
            df = load_weights_from_npz(
                    weights_npz=os.path.join(model_folder, f'{model_name}.weights.npz'), #'XSTUDY_ALL_FEATURE_L1_v4.weights.npz', 
                    weights_col=os.path.join(model_folder, f'{model_name}.columns.csv')) #'XSTUDY_ALL_FEATURE_L1_v4.columns.csv')
            self.coefficients   = df
            self.intercepts     = pd.read_csv(os.path.join(model_folder, f'{model_name}.intercepts.csv'), index_col = 0)

        else:
            self.coefficients   = pd.read_csv(os.path.join(model_folder, f'{model_name}.weights.csv')   , index_col = 0)
            self.intercepts     = pd.read_csv(os.path.join(model_folder, f'{model_name}.intercepts.csv'), index_col = 0)
    
    def load_data(self, model_folder, model_name):
        self.X = pd.read_csv(os.path.join(model_folder, f'{model_name}.training_data.csv'), index_col = 0)
        self.Y = pd.read_csv(os.path.join(model_folder, f'{model_name}.observations.csv'), index_col = 0)
    
    def load_calibrations(self, model_folder, model_name):
        self.calibrations = pd.read_csv(os.path.join(model_folder, f'{model_name}.calibrations.csv'), index_col = 0)#.to_dict()

    def save_calibration(self, model_folder, model_name):
        pd.DataFrame(self.calibrations).to_csv(os.path.join(model_folder,f"{model_name}.calibrations.csv"))

    def predict_decisions_x(self, X):
        # X is a transpose
        #X = (X > 0).astype(int).values.transpose()
        #if C is not None:
        #    X = pd.concat([X, C])
        assert X.values.dtype == 'int64', "Xt values are not of type int64"
        W = self.coefficients
        i = self.intercepts
        binary_variables = W.columns
        ic_ = np.array(i['intercept'].to_list())
        ic_tile = np.tile(ic_, (X.shape[0], 1))
        z = np.dot(X,W)

        decision_scores = pd.DataFrame(z, columns = binary_variables)
        
        decision_scores.index = X.index
        
        self.decision_scores = decision_scores

        return decision_scores 

    def score_predictions(self, probs, observations, thr = .5, gate =(.5,.5), variables = None):
        from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
        from sklearn.metrics import roc_curve, roc_auc_score
        perform= list()
        if variables is None:
            variables = sorted(self.Q.binary.unique())
        for binary in variables:
            #print(binary)
            try:
                if binary in observations.columns:
                    d = pd.DataFrame({'p':probs[binary].values,'obs':observations[binary].values}, index = observations.index )
                    d = d.dropna().sort_values('p')
                    gate_ix = (d['p'] <= gate[0])| (d['p'] >= gate[1])
                    n_pre = d.shape[0]
                    d = d[gate_ix]
                    n_post = d.shape[0]
                    y_pred =  d['p'] > thr
                    y_test =  d['obs'].astype('bool')
                    auc = roc_auc_score(y_test, d['p'] )
                    f1 = f1_score(y_test, y_pred)
                    if y_pred.sum() > 0:
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        sens = tp / (tp + fn)
                        spec = tn / (tn + fp)
                        acc = (tp+tn)/(tp+fp+fn+tn)
                        mse = np.sum((y_pred-y_test.astype('float64'))**2) / y_pred.shape[0]
                        perform.append((binary,tn,fp,fn,tp,sens,spec,f1,acc,auc,mse,n_pre,n_post,gate[0],gate[1]))
            except ValueError:
                #print(f"XXXXX {binary} XXXXX")
                continue
        perform_df = pd.DataFrame(perform, columns = 'i,tn,fp,fn,tp,sens,spec,f1,acc,auc,mse,n_pre,n_post,gate0,gate1'.split(','))
        return perform_df

    def predict_raw_prob(self, X, C = None):
        X = (X > 0).astype(int).values.transpose()
        if C is not None:
            X = pd.concat([X, C])
        W = self.coefficients
        i = self.intercepts
        binary_variables = W.columns
        ic_ = np.array(i['intercept'].to_list())
        ic_tile = np.tile(ic_, (X.shape[0], 1))
        z = np.dot(X,W) + ic_tile
        P_prob = 1 / (1 + np.exp(-z))
        P_df = pd.DataFrame(P_prob, columns = binary_variables )
        return P_df

    def output_probs_and_obs(self, probs, observations, variables = None):
        store = list()
        if variables == None:
            variables =sorted(self.Q.binary.unique())
        for binary in variables:
            if binary in observations.columns:
                d = pd.DataFrame({'p':probs[binary].values,'obs':observations[binary].values}, index = observations.index )
                d['sample_id'] = observations.index
                d['binary'] = binary
                store.append(d)
        return pd.concat(store).reset_index(drop = True)
    
    def pxx(self, decision_scores, variables=None, covariates=None):
        calibrated_pred = {}
        if variables is None:
            variables = sorted(self.Q.binary.unique())

        if covariates is not None:
            assert isinstance(covariates, pd.DataFrame), "Covariates must be a pandas DataFrame."
            assert decision_scores.shape[0] == covariates.shape[0], \
            "Decision scores and covariates must have the same number of rows."
            assert np.all(decision_scores.index == covariates.index)

         # Calculate calibrated probabilities for each binary variable
        for binary in variables:
            if binary in self.calibrations.keys():
                #print(f"Calibrating for binary variable: {binary}")
                # Retrieve intercept and coefficients
                intercept = self.calibrations[binary]['intercept']
                # Retrieve coeficients
                if covariates is not None:
                        all_variables =  ['coef'] + covariates.columns.to_list()
                else:
                    all_variables = ['coef']
                coef = np.array([self.calibrations[binary][col] for col in all_variables])

                # Combine decision scores and covariates if applicable
                if covariates is None:
                    Xb = decision_scores[[binary]].values
                else:
                    Xb = np.hstack([decision_scores[[binary]].values, covariates.values])

            # Validate the shapes match
            assert Xb.shape[1] == coef.shape[0], \
            f"Shape mismatch: Xb has {Xb.shape[1]} features, but coef expects {coef.shape[0]}."
            # Compute probabilities using the logistic function
            Xb = Xb.astype(float)
            z = -1 * (intercept + np.dot(Xb, coef))
            #import pdb;pdb.set_trace()
            z=z.astype(float)
            prob = 1 / (1 + np.exp(z))
            # Store the probabilities
            calibrated_pred[binary] = prob

        # Create a DataFrame from the results
        calibrated_prob = pd.DataFrame(calibrated_pred, index = decision_scores.index)
        
        self.calibrated_prob = calibrated_prob
        return calibrated_prob

def load_weights_from_npz(
    weights_npz='XSTUDY_ALL_FEATURE_L1_v4.weights.npz', 
    weights_col='XSTUDY_ALL_FEATURE_L1_v4.columns.csv'):
    S = load_npz(weights_npz)
    w_cols = pd.read_csv(weights_col).iloc[:,0].to_list()
    df = pd.DataFrame(
        S.toarray(), 
        columns = w_cols )
    return(df)