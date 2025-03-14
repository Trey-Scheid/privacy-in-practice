"""
File: train.py
Author: Trey Scheid
Date: last modified 03/2025
Description: process data and train lasso regression models based on parameters, compute performance metrics
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score#, jaccard_score
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd
import matplotlib.cm as cm

from src.model_build import frankWolfeLASSO as FWLasso

y_name = 'power_mean'

def jaccard_similarity(set1, set2):
    """
    Intersection / Union of two sets

    Args:
        set1 (array-like): list of objects
        set2 (array-like): comparison list of objects

    Returns:
        float: Jaccard similarity
    """
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union

def trivial(feat, correct_feats=None):
    """
    Train and evaluate the trivial model for lasso regression error function, predicts mean of y for all x

    Args:
        feat (array-like): dataset (will be split for eval)
        correct_feats (array-like, optional): will compute similarity of solution features with correct_feats list. Defaults to None.

    Returns:
        mse: test set mean square error
        coef_dict: model coefficients
        r2: correlation coefficient between predictions and outcomes
        similarity: jaccard similarity between non-zero coeficients and correct_feats if any
    """
    # prep feature data and prediction array
    X, y = feat.drop(y_name, axis=1), feat[y_name]
    ones_column = np.ones((X.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(np.hstack((ones_column, X)), y, test_size=0.2, random_state=1)
    
    # create trivial model
    avg = np.mean(y_train)
    coef = np.append(np.array([avg]), np.zeros(X.shape[1]))

    # compute metrics
    print(f'Train MSE sklearn: {mean_squared_error(y_train, np.repeat(avg, y_train.shape[0])):.2f} ({100*sum(coef>0)/coef.shape[0]:.1f}% sparse)')
    
    mse = mean_squared_error(y_test, np.repeat(avg, y_test.shape[0]))
    # print(f'Test MSE sklearn: {mse:.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
    
    coef_dict = dict(zip(np.append(["Intercept"], X.columns), coef))
    
    r2 = r2_score(y_test, np.repeat(avg, y_test.shape[0]), force_finite=False)
    
    if correct_feats:
        similarity = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if not correct_feats is None else None
    else:
        similarity = None
    
    return mse, coef_dict, r2, similarity

def train(feat, correct_feats=None, method='lasso', tol=1e-8, l=1, max_iter=10000, epsilon=None, delta=1e-6, plot=False, normalize=False, clip_sd=None, triv=None, nonpriv=None):
    """
    Train linear model for power usage

    Args:
        feat (array-like): featureized data as pandas DataFrame
        method (str): type of linear model ('lstsq', 'lasso', 'fw-lasso-exp', 'fw-lasso-lap', 'compare-fw-plot')
    
    Returns: 
        mse (float): test error
        coefficient dictionary (dict): model
        r2 (float): score
        convergence trace (array-like): plot values
    """
    type_fw = False
    if method == 'lstsq':
        model = LinearRegression()
    elif method == 'lasso':
        model = Lasso(alpha=l, max_iter=max_iter, tol=tol, fit_intercept=True)
    elif method in set(["fw-lasso", 'fw-lasso-exp', 'fw-lasso-lap', 'compare-fw-plot']):
        type_fw = True
    else:
        raise ValueError('method must be "lstsq" or "lasso"')

    if type_fw:
        #print("training frank wolfe model")
        #print(X_train.to_numpy().shape, y_train.to_numpy().shape)
        X, y = feat.drop(y_name, axis=1), feat[y_name]
        ones_column = np.ones((X.shape[0], 1))
        X_train, X_test, y_train, y_test = train_test_split(np.hstack((ones_column, X)), y, test_size=0.7, random_state=1)

        should_trace = True #if plot else False

        if method == 'fw-lasso-lap':
            model = FWLasso.LaplaceNoise
            #(X_train, y_train, l=l,tol=tol, delta=delta, epsilon=epsilon, K=max_iter, trace=should_trace, normalize=normalize, clip_sd=clip_sd)
        
        elif method == "fw-lasso":
            model = FWLasso.FW_NonPrivate
            #(X_train, y_train, l=l, tol=tol, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
        
        elif method == 'fw-lasso-exp':
            model = FWLasso.ExponentialMechanism
        
        else:
            #method == 'compare-fw-plot':
            model1 = FWLasso.LaplaceNoise(X_train, y_train, l=l, delta=delta, epsilon=epsilon, tol=tol, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
            model3 = FWLasso.ExponentialMechanism(X_train, y_train, l=l, delta=delta, epsilon=epsilon, tol=tol, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
            model2 = FWLasso.FW_NonPrivate(X_train, y_train, l=l, tol=tol, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
            if not triv is None and not isinstance(triv, float):
                triv = trivial(feat)[0]

            if plot:
                trace1 = model1.get("plot")
                trace2 = model2.get("plot")
                trace3 = model3.get("plot")
                plt.clf()
                fig = plt.figure(figsize=(8, 6), facecolor='#EEEEEE', edgecolor='#EEEEEE')
                ax = fig.add_subplot(1, 1, 1)
                ax.set_facecolor("#EEEEEE")
                if not triv is None:
                    ax.plot(range(max(len(trace2), len(trace1))), np.repeat(triv, max(len(trace2), len(trace1))), "#222831", lw=1, label="Trivial Error", linestyle="--")
                ax.plot(range(len(trace2)), trace2, "#31363F", lw=1, label="Non-Private Baseline")
                plt.plot(range(len(trace1)), trace1, "#76ABAE", lw=1, label="Private Frank-Wolfe (Lap)") # / max(trace1)
                ax.plot(range(len(trace3)), trace3, "#76ABAE", lw=1, label="Private Frank-Wolfe (Exp)")
                plt.xlabel('Iterations')
                plt.ylabel('Least Square Error')
                plt.title(f'Training Loss')
                # plt.tight_layout()
                plt.legend(frameon=False, fontsize="small")
                ax.set_ylim(0, max(trace3[int(len(trace3) * 0.08):]))
                plt.savefig(plot, dpi=300, pad_inches=2)

                y_pred = X_train @ model3.get("model")
                exp_train = mean_squared_error(y_train, y_pred)

                y_pred = X_train @ model2.get("model")
                baseline_train = mean_squared_error(y_train, y_pred)


                y_pred = X_test @ model3.get("model")
                exp_test = mean_squared_error(y_test, y_pred)

                y_pred = X_test @ model2.get("model")
                baseline_test = mean_squared_error(y_test, y_pred)
                return exp_train, baseline_train, exp_test, baseline_test
        
        model = model(X_train, y_train, l=l, tol=tol, delta=delta, epsilon=epsilon, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)

        print(f'Train MSE fw: {mean_squared_error(y_train, X_train @ model.get("model")):.2f} ({100*sum(model.get("model")>0)/model.get("model").shape[0]:.1f}% sparse)')
        y_pred = X_test @ model.get("model")
        mse = mean_squared_error(y_test, y_pred)
        # print(f'Test MSE {method}: {mse:.2f} ({100*sum(model>0)/model.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(np.append(["Intercept"],X.columns), model.get("model")))
    else:
        X, y = feat.drop(y_name, axis=1), feat[y_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # SciKit Model fitting
        model.fit(X_train, y_train)
        # MSE calculation
        y_pred = model.predict(X_test)
        print(f'Train MSE sklearn: {mean_squared_error(y_train, model.predict(X_train)):.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
        mse = mean_squared_error(y_test, y_pred)
        # print(f'Test MSE sklearn: {mse:.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(X.columns, model.coef_))
        coef_dict["Intercept"] = model.intercept_

    r2 = r2_score(y_test, y_pred, force_finite=False)
    similarity = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if not correct_feats is None else "No correct feature name vector passed to find similarity"
    
    if plot:
        trace = model.get("plot")
        plt.clf()
        plt.plot(range(len(trace)), trace, color="#31363F", lw=1.33)
        plt.xlabel('Iteration')
        plt.ylabel('Lasso Training Loss')
        plt.title(f'{method} Convergence')
        plt.ylim(0, max(trace))
        plt.tight_layout()
        plt.savefig(plot, dpi=300, facecolor='#EEEEEE', edgecolor='#EEEEEE')

    return mse, coef_dict, r2, similarity


def train_run_eps(eps_vals, model, data, tol, l, K, plot_dir, baseline):

    # run model on each epsilon value
    epsresults = []
    for eps in eps_vals:
        test_mse, feat_dict, r2, sim = train.train(data, model, tol=tol, l=l, epsilon=eps, max_iter=K, plot= plot_dir / f'{model}_{eps}_convergence.png')
        
        epsresults.append(test_mse)

    # convert mse to utility
    rmses = np.sqrt(np.array(epsresults))
    max_rmse = np.max(rmses)
    # for higher values of c, punish rmse more. c in (0, inf)
    utility = 2 / (1 + np.exp(c * rmses / max_rmse)) # use sigmoid function to normalize
    utility = 1 - (rmses - baseline_mse**.5) / (50 - baseline_mse ** .5)
    
    df = pd.DataFrame({'epsilon': eps_vals,
                'task': ["Lasso Regression" for i in range(len(eps_vals))],  
                'utility': utility.tolist()})#.to_csv("lasso_results.csv", index_label="Index")

    return df


def research_plots(feat, correct_feats=None, methods='lasso', baseline=None, l=None, max_iter=10000, epsilon=None, target_eps=None, delta=None, plot=False, triv=False, nonpriv=None, normalize=False, clip_sd=None, log=False):
    """
    Convergence and performance results plots for many models

    Args:
        feat (array-like): data
        correct_feats (array-like str, optional): correct comparable feature list. Defaults to None.
        methods (array-like or str, optional): methods to test against baseline. Defaults to 'lasso'.
        baseline (str, optional): one model to use as baseline. Defaults to None.
        l (float, optional): constraint size. Defaults to None.
        max_iter (int, optional): Defaults to 10000.
        epsilon (float, optional): privacy budget. Defaults to None.
        target_eps (float, optional): eps to check sparsity with baseline. Defaults to None.
        delta (float, optional): privacy parameter. Defaults to None.
        plot (str, optional): filepath for plots if any. Defaults to False.
        triv (bool, optional): add line for trivial model to plots. Defaults to False.
        nonpriv (_type_, optional): add line for baseline to plots. Defaults to None.
        normalize (bool, optional): normalize the input feat during training. Defaults to False.
        clip_sd (float, optional): clip. Defaults to None.
        log (bool, optional): Defaults to False.

    Raises:
        ValueError: Baseline and methods must be in list

    Returns:
        super_df: results from each trained model and its configuration.
    """             
    if isinstance(methods, str):
        methods = [methods]
    if isinstance(l, (int, float)):
        l = [l]
    if isinstance(max_iter, int):
        max_iter = [max_iter]
    if isinstance(epsilon, (int, float)) or epsilon is None:
        epsilon = [epsilon]
    if target_eps is None:
        target_eps = max(epsilon)
    
    tol = 1e-4
    
    all_results = []
    all_parameters = []
    best_models = {}  # Store best models for each method
    
    if log:
        print("Initializing research plots...")
    
    X, y = feat.drop(y_name, axis=1), feat[y_name]
    ones_column = np.ones((X.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(np.hstack((ones_column, X)), y, test_size=0.7, random_state=1)

    # Store trivial model metrics if requested
    triv_metrics = {}
    if triv:
        if log:
            print("Computing trivial model metrics...")
        # create trivial model
        avg = np.mean(y_train)
        coef = np.append(np.array([avg]), np.zeros(X.shape[1]))
        triv_train_mse = mean_squared_error(y_train, np.repeat(avg, y_train.shape[0]))
        triv_test_mse = mean_squared_error(y_test, np.repeat(avg, y_test.shape[0]))
        triv_coef_dict = dict(zip(np.append(["Intercept"], X.columns), coef))
        triv_r2 = r2_score(y_test, np.repeat(avg, y_test.shape[0]), force_finite=False)
        if correct_feats is not None:
            triv_similarity = jaccard_similarity(correct_feats, [k for k, v in triv_coef_dict.items() if v > 0]) if correct_feats is not None else None
        else:
            triv_similarity = None
        
        # Store trivial metrics for plotting reference lines
        triv_metrics = {
            "mse": triv_test_mse,
            "R2": triv_r2,
            "similarity": triv_similarity,
            "training_err": triv_train_mse,
            "sparse": 0
        }
        
        # Add trivial model to results
        triv_results = {
            "method": ["trivial"],
            "l": [None],
            "iter": [None],
            "eps": [None],
            "delta": [None],
            "mse": [triv_test_mse],
            "R2": [triv_r2],
            "similarity": [triv_similarity],
            "training_err": [triv_train_mse],
            "trace_f": [None],
            "sparse": [0]
        }
        all_results.append(pd.DataFrame(triv_results))

    if (delta is None) and (not epsilon is None):
        delta = 1 / (X.shape[0]**3)

    if baseline:
        if log:
            print(f"Processing baseline: {baseline}...")
        
        # Initialize model class based on method
        type_fw = False
        if baseline == 'lstsq':
            model_class = LinearRegression()
        elif baseline == 'lasso':
            model_class = lambda X, y, l, tol, delta, epsilon, K, normalize, clip_sd, trace: {
                "model": Lasso(alpha=l, max_iter=K, tol=tol, fit_intercept=True).fit(X[:, 1:], y).coef_,
                "plot": None
            }
        if baseline == 'fw-lasso-lap':
            model_class = FWLasso.LaplaceNoise
            type_fw = True
        elif baseline == "fw-lasso":
            model_class = FWLasso.FW_NonPrivate
            type_fw = True
        elif baseline == 'fw-lasso-exp':
            model_class = FWLasso.ExponentialMechanism
            type_fw = True
        else:
            raise ValueError(f'bad baseline "{baseline}"')
        
        # Run models with different parameters
        results = {"mse":[], "R2":[], "similarity":[], "training_err":[], "trace_f":[], "sparse":[]}
        parameters = {"method":[], "l":[], "iter":[], "eps":[], "delta":[]}

        best_mse = float('inf')
        best_config = None
        best_model_data = None

        if baseline == 'fw-lasso':
            total_runs = len(l) * len(max_iter) * len([None, 1_000_000])
        else:
            total_runs = len(l) * len(max_iter) * len(epsilon)
        current_run = 0
        
        for li in l:
            for niter in max_iter:
                if baseline == 'fw-lasso':
                    epsilons = [None, 1_000_000]
                else:
                    epsilons = epsilon
                for eps in epsilons:
                    if baseline == 'fw-lasso':
                        deltas = [0]
                    else:
                        deltas = [delta] if delta is not None else [None]
                    for d in deltas:
                        current_run += 1
                        if log and current_run % max(1, total_runs // 10) == 0:
                            print(f"  Progress: {current_run}/{total_runs} runs completed ({current_run/total_runs*100:.1f}%)")
                        
                        parameters["method"].append(baseline)
                        parameters["l"].append(li)
                        parameters["iter"].append(niter)
                        parameters["eps"].append(eps)
                        parameters["delta"].append(d)

                        model = model_class(X_train, y_train, l=li, tol=0, delta=d, epsilon=eps, K=niter, normalize=normalize, clip_sd=clip_sd, trace=True)

                        train_err = mean_squared_error(y_train, X_train @ model.get("model"))
                        sparsity = 100*sum(model.get("model")>0)/model.get("model").shape[0]
                        test_mse = mean_squared_error(y_test, X_test @ model.get("model"))
                        coef_dict = dict(zip(np.append(["Intercept"],X.columns), model.get("model")))
                        r2 = r2_score(y_test, X_test @ model.get("model"), force_finite=False)
                        if correct_feats is not None:
                            sim = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if correct_feats is not None else None
                        else:
                            sim = None
                        
                        results["training_err"].append(train_err)
                        results["sparse"].append(sparsity)
                        results["mse"].append(test_mse)
                        results["R2"].append(r2)
                        results["similarity"].append(sim)
                        results["trace_f"].append(model.get("plot"))
                        
                        # Track best model for this baseline
                        if test_mse < best_mse:
                            best_mse = test_mse
                            best_config = {"l": li, "iter": niter, "eps": eps, "delta": d}
                            best_model_data = {
                                "model": model,
                                "train_err": train_err,
                                "test_mse": test_mse,
                                "r2": r2,
                                "sim": sim,
                                "sparsity": sparsity,
                                "trace": model.get("plot")
                            }
        
        best_models[baseline] = {"config": best_config, "data": best_model_data}
        
        # Create dataframe for this baseline
        baseline_df = pd.DataFrame({
            "method": parameters["method"],
            "l": parameters["l"],
            "iter": parameters["iter"],
            "eps": parameters["eps"],
            "delta": parameters["delta"],
            "mse": results["mse"],
            "R2": results["R2"],
            "similarity": results["similarity"],
            "training_err": results["training_err"],
            "sparse": results["sparse"]
        })
        # set baseline features to best non-private model solution
        if correct_feats is None:
            coef_dict = dict(zip(np.append(["Intercept"],X.columns), best_models[baseline].get("data").get("model").get("model")))
            correct_feats = [k for k, v in coef_dict.items() if v > 0]
        
        all_results.append(baseline_df)
        
        if plot:
            if log:
                print(f"  Generating plots for baseline: {baseline}...")
            
            # Create a color map for this baseline l
            unique_values = sorted(list(set([v for v in parameters["l"]])))
            cmap = cm.Blues
            min_intensity = 0.3
            colors = [cmap(min_intensity + (i/max(1, len(unique_values)-1)) * (1-min_intensity)) for i in range(len(unique_values))]
            color_map_blue = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = list(set([v for v in parameters["eps"]]))
            cmap = cm.Greens
            min_intensity = 0.3
            colors = [cmap(min_intensity + val * (1-min_intensity)) for val in (np.arange(len(parameters["eps"])) / (len(parameters["eps"])-1))]
            color_map_green = {str(val): colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            min_intensity = 0.3
            colors = [cmap(min_intensity + (i/max(1, len(unique_values)-1)) * (1-min_intensity)) for i in range(len(unique_values))]
            color_map_red = {val: colors[i] for i, val in enumerate(unique_values)}
            
            # Convergence plots
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Training Convergence for {baseline}', fontsize=16)

            maxys = []
            maxs = []
            # Training error vs iter for each epsilon
            plt.subplot(1, 2, 1)
            for i, eps_val in enumerate(sorted([e for e in set(parameters["eps"]) if e is not None])):
                eps_indices = [i for i, e in enumerate(parameters["eps"]) if e == eps_val and parameters["l"][i] == best_config["l"] and parameters["iter"][i] == best_config["iter"]]
                if eps_indices:
                    for idx in eps_indices:
                        if results["trace_f"][idx] is not None:
                            maxys.append(max(results["trace_f"][idx]))
                            maxs.append(len(results["trace_f"][idx]))
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_green[str(eps_val)], label=f'ε={eps_val}')
            plt.xlabel('Iterations')
            plt.ylabel('Training Error')
            plt.legend()

            plt.ylim(0, 50)#max(maxys))
            if maxs:
                plt.xlim(0, max(maxs))
            

            maxys = []
            maxs = []
            # Training error vs iter for each l
            plt.subplot(1, 2, 2)
            for i, l_val in enumerate(sorted(set(parameters["l"]))):
                l_indices = [i for i, e in enumerate(parameters["l"]) if e == l_val and parameters["eps"][i] == best_config["eps"] and parameters["iter"][i] == best_config["iter"]]
                if l_indices:
                    for idx in l_indices:
                        if results["trace_f"][idx] is not None:
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_blue[l_val], label=f'λ={l_val}')
            plt.xlabel('Iterations')
            plt.ylabel('Training Error')
            plt.legend()

            plt.ylim(0, 50)#max(maxys))
            if maxs:
                plt.xlim(0, max(maxs))
            
            plt.tight_layout()
            # Save convergence plots
            if isinstance(plot, str):
                convergence_plot_path = os.path.join(plot, f'{baseline}_convergence_plots.png')
                plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Results plots
            plt.figure(figsize=(15, 10))
            plt.suptitle(f'Performance vs Privacy for {baseline}', fontsize=16, y=0.96)
            
            # Filter for best iteration
            best_iter_df = baseline_df[baseline_df["iter"] == best_config["iter"]]
            
            # First row: plots per lambda
            # MSE vs eps for each l
            plt.subplot(2, 4, 1)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["mse"], marker='o', color=color_map_blue[l_val], label=f'λ={l_val}')
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('MSE')
            plt.legend()
            
            # R2 vs eps for each l
            plt.subplot(2, 4, 2)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["R2"], marker='o', color=color_map_blue[l_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('R²')
            
            # Similarity vs eps for each l
            if correct_feats is not None:
                plt.subplot(2, 4, 3)
                for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                    l_data = best_iter_df[best_iter_df["l"] == l_val]
                    plt.plot(l_data["eps"], l_data["similarity"], marker='o', color=color_map_blue[l_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                plt.xlabel('ε')
                plt.xscale('log')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each l
            plt.subplot(2, 4, 4)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["sparse"], marker='o', color=color_map_blue[l_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('Sparsity (%)')
            
            # Second row: plots per iteration
            # Create a color map for iterations
            unique_iters = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            iter_colors = [cmap(min_intensity + (i/max(1, len(unique_iters)-1)) * (1-min_intensity)) for i in range(len(unique_iters))]
            iter_color_map_iter = {val: iter_colors[i] for i, val in enumerate(unique_iters)}
            
            # Filter for best l
            best_l_df = baseline_df[baseline_df["l"] == best_config["l"]]
            
            # MSE vs eps for each niter
            plt.subplot(2, 4, 5)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["mse"], marker='o', color=iter_color_map_iter[iter_val], label=f'iter={iter_val}')
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('MSE')
            plt.legend()
            
            # R2 vs eps for each niter
            plt.subplot(2, 4, 6)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["R2"], marker='o', color=iter_color_map_iter[iter_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('R²')
            
            # Similarity vs eps for each niter
            if correct_feats is not None:
                plt.subplot(2, 4, 7)
                for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                    iter_data = best_l_df[best_l_df["iter"] == iter_val]
                    plt.plot(iter_data["eps"], iter_data["similarity"], marker='o', color=iter_color_map_iter[iter_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                plt.xlabel('ε')
                plt.xscale('log')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each niter
            plt.subplot(2, 4, 8)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["sparse"], marker='o', color=iter_color_map_iter[iter_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('Sparsity (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            
            # Save results plots
            if isinstance(plot, str):
                fig = plt.gcf()
                fig.text(0.5, 0.88, f"Various Regularization at optimal K={best_config['iter']}", ha='center', va='center', fontsize=13)#, fontweight='bold')
                fig.text(0.5, 0.43, f"Various Iterations at optimal λ={best_config['l']}", ha='center', va='center', fontsize=13)#, fontweight='bold')
                plt.subplots_adjust(top=0.85, hspace=0.4)
                results_plot_path = os.path.join(plot, f'{baseline}_results_plots.png')
                plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
                plt.close()

    for method_idx, method in enumerate(methods):
        if log:
            print(f"Processing method: {method}...")
        
        # Initialize model class based on method
        type_fw = False
        if method == 'lstsq':
            model_class = LinearRegression()
        elif method == 'lasso':
            model_class = lambda X, y, l, tol, delta, epsilon, K, normalize, clip_sd, trace: {
                "model": Lasso(alpha=l, max_iter=K, tol=tol, fit_intercept=True).fit(X[:, 1:], y).coef_,
                "plot": None
            }
        if method == 'fw-lasso-lap':
            model_class = FWLasso.LaplaceNoise
            type_fw = True
        elif method == "fw-lasso":
            model_class = FWLasso.FW_NonPrivate
            type_fw = True
        elif method == 'fw-lasso-exp':
            model_class = FWLasso.ExponentialMechanism
            type_fw = True
        else:
            raise ValueError(f'bad method "{method}"')
        
        # Run models with different parameters
        results = {"mse":[], "R2":[], "similarity":[], "training_err":[], "trace_f":[], "sparse":[]}
        parameters = {"method":[], "l":[], "iter":[], "eps":[], "delta":[]}

        best_mse = float('inf')
        best_config = None
        best_model_data = None

        if method == 'fw-lasso':
            total_runs = len(l) * len(max_iter) * len([None, 1_000_000])
        else:
            total_runs = len(l) * len(max_iter) * len(epsilon)
        current_run = 0
        
        for li in l:
            for niter in max_iter:
                if method == 'fw-lasso':
                    epsilons = [None, 1_000_000]
                else:
                    epsilons = epsilon
                for eps in epsilons:
                    if method == 'fw-lasso':
                        deltas = [0]
                    else:
                        deltas = [delta] if delta is not None else [None]
                    for d in deltas:
                        current_run += 1
                        if log and current_run % max(1, total_runs // 10) == 0:
                            print(f"  Progress: {current_run}/{total_runs} runs completed ({current_run/total_runs*100:.1f}%)")
                        
                        parameters["method"].append(method)
                        parameters["l"].append(li)
                        parameters["iter"].append(niter)
                        parameters["eps"].append(eps)
                        parameters["delta"].append(d)

                        model = model_class(X_train, y_train, l=li, tol=0, delta=d, epsilon=eps, K=niter, normalize=normalize, clip_sd=clip_sd, trace=True)

                        train_err = mean_squared_error(y_train, X_train @ model.get("model"))
                        sparsity = 100*sum(model.get("model")>0)/model.get("model").shape[0]
                        test_mse = mean_squared_error(y_test, X_test @ model.get("model"))
                        coef_dict = dict(zip(np.append(["Intercept"],X.columns), model.get("model")))
                        r2 = r2_score(y_test, X_test @ model.get("model"), force_finite=False)
                        sim = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if correct_feats is not None else None
                        
                        results["training_err"].append(train_err)
                        results["sparse"].append(sparsity)
                        results["mse"].append(test_mse)
                        results["R2"].append(r2)
                        results["similarity"].append(sim)
                        results["trace_f"].append(model.get("plot"))
                        
                        # Track best model for this method
                        if (test_mse < best_mse) and (eps <= target_eps) and (r2 > 0) and ((best_models.get(baseline) is None) or (sparsity <= 1.5 * best_models.get(baseline).get("data").get("sparsity"))):
                            best_mse = test_mse
                            best_config = {"l": li, "iter": niter, "eps": eps, "delta": d}
                            best_model_data = {
                                "model": model,
                                "train_err": train_err,
                                "test_mse": test_mse,
                                "r2": r2,
                                "sim": sim,
                                "sparsity": sparsity,
                                "trace": model.get("plot")
                            }
        
        best_models[method] = {"config": best_config, "data": best_model_data}
        
        # Create dataframe for this method
        method_df = pd.DataFrame({
            "method": parameters["method"],
            "l": parameters["l"],
            "iter": parameters["iter"],
            "eps": parameters["eps"],
            "delta": parameters["delta"],
            "mse": results["mse"],
            "R2": results["R2"],
            "similarity": results["similarity"],
            "training_err": results["training_err"],
            "sparse": results["sparse"]
        })
        
        all_results.append(method_df)
        
        if plot:
            if log:
                print(f"  Generating plots for method: {method}...")
            
            # Create a color map for this method l
            unique_values = sorted(list(set([v for v in parameters["l"]])))
            cmap = cm.Blues
            min_intensity = 0.3
            colors = [cmap(min_intensity + (i/max(1, len(unique_values)-1)) * (1-min_intensity)) for i in range(len(unique_values))]
            color_map_blue = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = (list(set([v for v in parameters["eps"]])))
            cmap = cm.Greens
            min_intensity = 0.3
            log_unique_values = np.log10(np.array(unique_values))
            min_log = min(log_unique_values)
            max_log = max(log_unique_values)
            normalized_values = [(val - min_log) / (max_log - min_log) for val in log_unique_values]
            colors = [cmap(min_intensity + val * (1-min_intensity)) for val in normalized_values]
            color_map_green = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            min_intensity = 0.3
            colors = [cmap(min_intensity + (i/max(1, len(unique_values)-1)) * (1-min_intensity)) for i in range(len(unique_values))]
            color_map_red = {val: colors[i] for i, val in enumerate(unique_values)}
            
            # Convergence plots
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Training Convergence for {method}', fontsize=16)

            maxys = []
            maxs = []
            # Training error vs iter for each epsilon @ best K / L
            plt.subplot(1, 2, 1)
            plt.title(f"Model Training Error at λ={best_config['l']}, K={best_config['iter']}")
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.8, label='Trivial Model')
            if baseline and not best_models[baseline]['data'].get("train_err") is None:
                plt.axhline(y=best_models[baseline]['data'].get("train_err"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
                plt.plot(range(len(best_models[baseline]['data'].get("trace"))), best_models[baseline]['data'].get("trace"), color='black', alpha=0.7, label=f'Non-Private Model', lw=1)
            for i, eps_val in enumerate(sorted([e for e in set(parameters["eps"]) if e is not None])):
                eps_indices = [i for i, e in enumerate(parameters["eps"]) if e == eps_val and parameters["l"][i] == best_config["l"] and parameters["iter"][i] == best_config["iter"]]
                if eps_indices:
                    for idx in eps_indices:
                        if results["trace_f"][idx] is not None:
                            maxys.append(max(results["trace_f"][idx]))
                            maxs.append(len(results["trace_f"][idx]))
                            
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_green[eps_val], label=f'ε={eps_val} Private Model', lw=1)
            plt.xlabel('Iterations')
            plt.ylabel('Training Error')
            plt.legend()

            plt.ylim(0, 20)#max(maxys))
            if maxs:
                plt.xlim(0, max(maxs))
            

            maxys = []
            maxs = []
            # Training error vs iter for each l @ best eps / K
            plt.subplot(1, 2, 2)
            plt.title(f"Model Training Error at ε={best_config['eps']}, K={best_config['iter']}")
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.8, label='Trivial Model')
            if baseline and not best_models[baseline]['data'].get("train_err") is None:
                plt.axhline(y=best_models[baseline]['data'].get("train_err"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
                plt.plot(range(len(best_models[baseline]['data'].get("trace"))), best_models[baseline]['data'].get("trace"), color='black', alpha=0.7, label=f'Non-Private Model', lw=1)

            for i, l_val in enumerate(sorted(set(parameters["l"]))):
                l_indices = [i for i, e in enumerate(parameters["l"]) if e == l_val and parameters["eps"][i] == best_config["eps"] and parameters["iter"][i] == best_config["iter"]]
                if l_indices:
                    for idx in l_indices:
                        if results["trace_f"][idx] is not None:
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_blue[l_val], label=f'λ={l_val}', lw=1)
            plt.xlabel('Iterations')
            plt.ylabel('Training Error')
            plt.legend()

            plt.ylim(0, 20)#max(maxys))
            if maxs:
                plt.xlim(0, max(maxs))
            
            plt.tight_layout()
            # Save convergence plots
            if isinstance(plot, str):
                convergence_plot_path = os.path.join(plot, f'{method}_convergence_plots.png')
                plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Results plots
            plt.figure(figsize=(15, 10))
            plt.suptitle(f'Performance vs Privacy for {method}', fontsize=16, y=0.98)
            
            # Filter for best iteration
            best_iter_df = method_df[method_df["iter"] == best_config["iter"]]
            
            # First row: plots per lambda
            # MSE vs eps for each l
            plt.subplot(2, 4, 1)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["mse"], marker='o', color=color_map_blue[l_val], label=f'λ={l_val}')
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            if baseline and not best_models[baseline]['data'].get("test_mse") is None:
                plt.axhline(y=best_models[baseline]['data'].get("test_mse"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('MSE')
            plt.legend()
            
            # R2 vs eps for each l
            plt.subplot(2, 4, 2)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["R2"], marker='o', color=color_map_blue[l_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['data'].get("r2") is None:
                plt.axhline(y=best_models[baseline]['data'].get("r2"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('R²')
            
            # Similarity vs eps for each l
            if correct_feats is not None:
                plt.subplot(2, 4, 3)
                for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                    l_data = best_iter_df[best_iter_df["l"] == l_val]
                    plt.plot(l_data["eps"], l_data["similarity"], marker='o', color=color_map_blue[l_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                if baseline and not best_models[baseline]['data'].get("sim") is None:
                    plt.axhline(y=best_models[baseline]['data'].get("sim"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
                plt.xlabel('ε')
                plt.xscale('log')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each l
            plt.subplot(2, 4, 4)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["sparse"], marker='o', color=color_map_blue[l_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['data'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['data'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('Sparsity (%)')
            
            # Second row: plots per iteration
            # Create a color map for iterations
            unique_iters = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            min_intensity = 0.3
            iter_colors = [cmap(min_intensity + (i/max(1, len(unique_iters)-1)) * (1-min_intensity)) for i in range(len(unique_iters))]
            iter_color_map_l = {val: iter_colors[i] for i, val in enumerate(unique_iters)}
            
            # Filter for best l
            best_l_df = method_df[method_df["l"] == best_config["l"]]
            
            # MSE vs eps for each niter
            plt.subplot(2, 4, 5)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["mse"], marker='o', color=iter_color_map_l[iter_val], label=f'iter={iter_val}')
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            if baseline and not best_models[baseline]['data'].get("test_mse") is None:
                plt.axhline(y=best_models[baseline]['data'].get("test_mse"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('MSE')
            plt.legend()
            
            # R2 vs eps for each niter
            plt.subplot(2, 4, 6)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["R2"], marker='o', color=iter_color_map_l[iter_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['data'].get("r2") is None:
                plt.axhline(y=best_models[baseline]['data'].get("r2"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('R²')
            
            # Similarity vs eps for each niter
            if correct_feats is not None:
                plt.subplot(2, 4, 7)
                for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                    iter_data = best_l_df[best_l_df["iter"] == iter_val]
                    plt.plot(iter_data["eps"], iter_data["similarity"], marker='o', color=iter_color_map_iter[iter_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                if baseline and not best_models[baseline]['data'].get("sim") is None:
                    plt.axhline(y=best_models[baseline]['data'].get("sim"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
                plt.xlabel('ε')
                plt.xscale('log')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each niter
            plt.subplot(2, 4, 8)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["sparse"], marker='o', color=iter_color_map_iter[iter_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['data'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['data'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('Sparsity (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            
            # Save results plots
            if isinstance(plot, str):
                fig = plt.gcf()
                fig.text(0.5, 0.88, f"Various Regularization at optimal K={best_config['iter']}", ha='center', va='center', fontsize=13)#, fontweight='bold')
                fig.text(0.5, 0.43, f"Various Iterations at optimal λ={best_config['l']}", ha='center', va='center', fontsize=13)#, fontweight='bold')
                plt.subplots_adjust(top=0.85, hspace=0.4)
                results_plot_path = os.path.join(plot, f'{method}_results_plots.png')
                plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    # Combine all results into a single dataframe
    if log:
        print("Combining results into a single dataframe...")
        
    if isinstance(all_results[0], pd.DataFrame):
        super_df = pd.concat(all_results, ignore_index=True)
    else:
        # If we only have the trivial model as a dictionary
        super_df = pd.DataFrame(all_results)
    
    if plot and len(methods) > 1:
        if log:
            print("Generating comparison plots between methods...")
            
        # Create a color map for methods
        method_cmap = cm.tab10
        method_colors = [method_cmap(i % 10) for i in range(len(methods))]
        method_color_map_l = {method: method_colors[i] for i in range(len(methods))}
        
        # Convergence plot comparing all methods
        plt.figure(figsize=(12, 6))
        plt.suptitle('Method Comparison: Convergence and Performance', fontsize=16)
        
        # Training error vs iter for each method
        plt.subplot(1, 2, 1)
        # plt.title("tbd1")
        for i, method in enumerate(methods):
            if method in best_models and best_models[method]["data"]["trace"] is not None:
                trace = best_models[method]["data"]["trace"]
                plt.plot(range(len(trace)), trace, color=method_colors[i], label=method)
        plt.xlabel('Iterations')
        plt.ylabel('Training Error')
        plt.legend()
        
        # MSE vs eps for each method
        plt.subplot(1, 2, 2)
        # plt.title("tbd2")
        for i, method in enumerate(methods):
            method_data = super_df[super_df["method"] == method]
            best_l = best_models[method]["config"]["l"] if method in best_models else None
            best_iter = best_models[method]["config"]["iter"] if method in best_models else None
            if best_l is not None and best_iter is not None:
                filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                plt.plot(filtered_data["eps"], filtered_data["mse"], marker='o', color=method_colors[i], label=method)
        if triv and "mse" in triv_metrics:
            plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
        if baseline and not best_models[baseline]['data'].get("test_mse") is None:
            plt.axhline(y=best_models[baseline]['data'].get("test_mse"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
        plt.xlabel('ε')
        plt.xscale('log')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        
        # Save comparison plots
        if isinstance(plot, str):
            comparison_plot_path = os.path.join(plot, 'methods_comparison_plots.png')
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create additional comparison plots for different metrics
        plt.figure(figsize=(15, 10))
        plt.suptitle('Method Comparison: Performance Metrics vs Privacy', fontsize=16, y=0.98)
        
        # R2 comparison
        plt.subplot(2, 2, 1)
        for i, method in enumerate(methods):
            method_data = super_df[super_df["method"] == method]
            best_l = best_models[method]["config"]["l"] if method in best_models else None
            best_iter = best_models[method]["config"]["iter"] if method in best_models else None
            if best_l is not None and best_iter is not None:
                filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                plt.plot(filtered_data["eps"], filtered_data["R2"], marker='o', color=method_colors[i], label=method)
        if triv and "R2" in triv_metrics:
            plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
        if baseline and not best_models[baseline]['data'].get("r2") is None:
            plt.axhline(y=best_models[baseline]['data'].get("r2"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
        plt.xlabel('ε')
        plt.xscale('log')
        plt.ylabel('R²')
        plt.legend()
        
        # Similarity comparison
        if correct_feats is not None:
            plt.subplot(2, 2, 2)
            for i, method in enumerate(methods):
                method_data = super_df[super_df["method"] == method]
                best_l = best_models[method]["config"]["l"] if method in best_models else None
                best_iter = best_models[method]["config"]["iter"] if method in best_models else None
                if best_l is not None and best_iter is not None:
                    filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                    plt.plot(filtered_data["eps"], filtered_data["similarity"], marker='o', color=method_colors[i], label=method)
            if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            if baseline and not best_models[baseline]['data'].get("sim") is None:
                plt.axhline(y=best_models[baseline]['data'].get("sim"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
            plt.xlabel('ε')
            plt.xscale('log')
            plt.ylabel('Jaccard Similarity')
            plt.legend()
        
        # Sparsity comparison
        plt.subplot(2, 2, 3)
        for i, method in enumerate(methods):
            method_data = super_df[super_df["method"] == method]
            best_l = best_models[method]["config"]["l"] if method in best_models else None
            best_iter = best_models[method]["config"]["iter"] if method in best_models else None
            if best_l is not None and best_iter is not None:
                filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                plt.plot(filtered_data["eps"], filtered_data["sparse"], marker='o', color=method_colors[i], label=method)
        if triv:
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Trivial')
        if baseline and not best_models[baseline]['data'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['data'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
        plt.xlabel('ε')
        plt.xscale('log')
        plt.ylabel('Sparsity (%)')
        plt.legend()
        
        # Training error comparison
        plt.subplot(2, 2, 4)
        for i, method in enumerate(methods):
            method_data = super_df[super_df["method"] == method]
            best_l = best_models[method]["config"]["l"] if method in best_models else None
            best_iter = best_models[method]["config"]["iter"] if method in best_models else None
            if best_l is not None and best_iter is not None:
                filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                plt.plot(filtered_data["eps"], filtered_data["training_err"], marker='o', color=method_colors[i], label=method)
        if triv and "training_err" in triv_metrics:
            plt.axhline(y=triv_metrics["training_err"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
        if baseline and not best_models[baseline]['data'].get("train_err") is None:
                plt.axhline(y=best_models[baseline]['data'].get("train_err"), color='black', linestyle='--', alpha=0.8, label="Non-Private Baseline")
        plt.xlabel('ε')
        plt.xscale('log')
        plt.ylabel('Training Error')
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        
        # Save additional comparison plots
        if isinstance(plot, str):
            metrics_comparison_path = os.path.join(plot, 'methods_metrics_comparison.png')
            plt.savefig(metrics_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

    if log:
        print("Research plots completed successfully!")

    return super_df
