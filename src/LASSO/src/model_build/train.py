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
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union

def trivial(feat, correct_feats=None):
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
    
    similarity = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if not correct_feats is None else None
    
    return mse, coef_dict, r2, similarity

def train(feat, correct_feats=None, method='lasso', tol=1e-8, l=1, max_iter=10000, epsilon=None, delta=1e-6, plot=False, normalize=False, clip_sd=None, triv=None, nonpriv=None):
    """
    Train linear model for power usage

    :param feat: featureized data as pandas DataFrame
    :param method: type of linear model ('lstsq' or 'lasso')
    :return: model, coefficient dictionary, r2 score, convergence trace
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
            if plot:
                trace1 = model1.get("plot")
                trace2 = model2.get("plot")
                trace3 = model3.get("plot")
                plt.clf()
                plt.figure(figsize=(6, 4))
                plt.plot(range(len(trace1)), trace1, "#76ABAE", lw=1, label="Private Frank-Wolfe (Lap)") # / max(trace1)
                plt.plot(range(len(trace3)), trace3, "darkgreen", lw=1, label="Private Frank-Wolfe (Exp)")
                plt.plot(range(len(trace2)), trace2, "#31363F", lw=1, label="Non-Private Baseline")
                if not triv is None:
                    plt.plot(range(max(len(trace2), len(trace1))), np.repeat(triv, max(len(trace2), len(trace1))), "#222831", lw=1, label="Trivial Error")
                # plt.yscale('log')
                # plt.xlabel('Number of iterations')
                # plt.ylabel('Training Lasso Loss')
                # plt.title(f'{method} Convergence')
                # plt.xlim()
                # plt.grid()
                # plt.tight_layout()
                # plt.legend(frameon=False, fontsize="small")
                plt.ylim(0, 50)
                plt.savefig(plot, dpi=300, facecolor='#EEEEEE', edgecolor='#EEEEEE', pad_inches=1)

                y_pred = X_train @ model1.get("model")
                mse11 = mean_squared_error(y_train, y_pred)

                y_pred = X_train @ model2.get("model")
                mse21 = mean_squared_error(y_train, y_pred)


                y_pred = X_test @ model1.get("model")
                mse12 = mean_squared_error(y_test, y_pred)

                y_pred = X_test @ model2.get("model")
                mse22 = mean_squared_error(y_test, y_pred)
                return mse11, mse21, mse12, mse22
        
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
        plt.plot(range(len(trace)), trace / max(trace), color="#31363F", lw=1.33)
        # plt.yscale('log')
        plt.xlabel('Number of iterations')
        plt.ylabel('Lasso Loss')
        plt.title(f'{method} Convergence')
        # plt.xlim()
        # plt.grid()
        plt.tight_layout()
        plt.savefig(plot, dpi=300, facecolor='#EEEEEE', edgecolor='#EEEEEE')
    return mse, coef_dict, r2, similarity

def research_plots(feat, correct_feats=None, methods='lasso', baseline=None, l=None, max_iter=10000, epsilon=None, delta=None, plot=False, triv=False, nonpriv=None, normalize=False, clip_sd=None, log=False):
    """Generates all the plots in the report.pdf"""
    if isinstance(methods, str):
        methods = [methods]
    if isinstance(l, (int, float)):
        l = [l]
    if isinstance(max_iter, int):
        max_iter = [max_iter]
    if isinstance(epsilon, (int, float)) or epsilon is None:
        epsilon = [epsilon]
    
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
        triv_similarity = jaccard_similarity(correct_feats, [k for k, v in triv_coef_dict.items() if v > 0]) if correct_feats is not None else None
        
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
                        sim = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0]) if correct_feats is not None else None
                        
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
        
        all_results.append(baseline_df)
        
        if plot:
            if log:
                print(f"  Generating plots for baseline: {baseline}...")
            
            # Create a color map for this baseline l
            unique_values = sorted(list(set([v for v in parameters["l"]])))
            cmap = cm.Blues
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_blue = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = (list(set([v for v in parameters["eps"]])))
            cmap = cm.Greens
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_green = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this baseline eps
            unique_values = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_red = {val: colors[i] for i, val in enumerate(unique_values)}
            
            # Convergence plots
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Training Convergence for {baseline}', fontsize=16)

            maxys = []
            maxs = []
            # Training error vs iter for each epsilon
            plt.subplot(1, 2, 1)
            for i, eps_val in enumerate(sorted([e for e in set(parameters["eps"]) if e is not None])):
                eps_indices = [i for i, e in enumerate(parameters["eps"]) if e == eps_val and parameters["l"][i] == best_config["l"]]
                if eps_indices:
                    for idx in eps_indices:
                        if results["trace_f"][idx] is not None:
                            maxys.append(max(results["trace_f"][idx]))
                            maxs.append(len(results["trace_f"][idx]))
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_green[eps_val], label=f'ε={eps_val}')
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
                l_indices = [i for i, e in enumerate(parameters["l"]) if e == l_val and parameters["eps"][i] == best_config["eps"]]
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
            plt.suptitle(f'Performance vs Privacy for {baseline}', fontsize=16, y=0.98)
            
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
            plt.ylabel('MSE')
            if i == 0:  # Only add legend to the first plot in the row
                plt.legend()
            
            # R2 vs eps for each l
            plt.subplot(2, 4, 2)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["R2"], marker='o', color=color_map_blue[l_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
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
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each l
            plt.subplot(2, 4, 4)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["sparse"], marker='o', color=color_map_blue[l_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.ylabel('Sparsity (%)')
            
            # Second row: plots per iteration
            # Create a color map for iterations
            unique_iters = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            iter_colors = [cmap(i/max(1, len(unique_iters)-1)) for i in range(len(unique_iters))]
            iter_color_map_l = {val: iter_colors[i] for i, val in enumerate(unique_iters)}
            
            # Filter for best l
            best_l_df = baseline_df[baseline_df["l"] == best_config["l"]]
            
            # MSE vs eps for each niter
            plt.subplot(2, 4, 5)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["mse"], marker='o', color=iter_color_map_l[iter_val], label=f'iter={iter_val}')
            if triv and "mse" in triv_metrics:
                plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
            plt.xlabel('ε')
            plt.ylabel('MSE')
            if i == 0:  # Only add legend to the first plot in the row
                plt.legend()
            
            # R2 vs eps for each niter
            plt.subplot(2, 4, 6)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["R2"], marker='o', color=iter_color_map_l[iter_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.ylabel('R²')
            
            # Similarity vs eps for each niter
            if correct_feats is not None:
                plt.subplot(2, 4, 7)
                for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                    iter_data = best_l_df[best_l_df["iter"] == iter_val]
                    plt.plot(iter_data["eps"], iter_data["similarity"], marker='o', color=iter_color_map_l[iter_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                plt.xlabel('ε')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each niter
            plt.subplot(2, 4, 8)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["sparse"], marker='o', color=iter_color_map_l[iter_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('ε')
            plt.ylabel('Sparsity (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            
            # Save results plots
            if isinstance(plot, str):
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
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_blue = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this method eps
            unique_values = sorted(list(set([v for v in parameters["eps"]])))
            cmap = cm.Greens
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_green = {val: colors[i] for i, val in enumerate(unique_values)}

            # Create a color map for this method eps
            unique_values = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            colors = [cmap(i/max(1, len(unique_values)-1)) for i in range(len(unique_values))]
            color_map_red = {val: colors[i] for i, val in enumerate(unique_values)}
            
            # Convergence plots
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Training Convergence for {method}', fontsize=16)

            maxys = []
            maxs = []
            # Training error vs iter for each epsilon
            plt.subplot(1, 2, 1)
            for i, eps_val in enumerate(sorted([e for e in set(parameters["eps"]) if e is not None])):
                eps_indices = [i for i, e in enumerate(parameters["eps"]) if e == eps_val and parameters["l"][i] == best_config["l"]]
                if eps_indices:
                    for idx in eps_indices:
                        if results["trace_f"][idx] is not None:
                            maxys.append(max(results["trace_f"][idx]))
                            maxs.append(len(results["trace_f"][idx]))
                            plt.plot(range(len(results["trace_f"][idx])), results["trace_f"][idx], color=color_map_green[eps_val], label=f'ε={eps_val}')
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
                l_indices = [i for i, e in enumerate(parameters["l"]) if e == l_val and parameters["eps"][i] == best_config["eps"]]
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
            if baseline and not best_models[baseline]['config'].get("test_err") is None:
                plt.axhline(y=best_models[baseline]['config'].get("test_err"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('MSE')
            if i == 0:  # Only add legend to the first plot in the row
                plt.legend()
            
            # R2 vs eps for each l
            plt.subplot(2, 4, 2)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["R2"], marker='o', color=color_map_blue[l_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['config'].get("r2") is None:
                plt.axhline(y=best_models[baseline]['config'].get("r2"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('R²')
            
            # Similarity vs eps for each l
            if correct_feats is not None:
                plt.subplot(2, 4, 3)
                for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                    l_data = best_iter_df[best_iter_df["l"] == l_val]
                    plt.plot(l_data["eps"], l_data["similarity"], marker='o', color=color_map_blue[l_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                if baseline and not best_models[baseline]['config'].get("sim") is None:
                    plt.axhline(y=best_models[baseline]['config'].get("sim"), color='black', linestyle='--', alpha=0.8, label=baseline)
                plt.xlabel('ε')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each l
            plt.subplot(2, 4, 4)
            for i, l_val in enumerate(sorted(set(best_iter_df["l"]))):
                l_data = best_iter_df[best_iter_df["l"] == l_val]
                plt.plot(l_data["eps"], l_data["sparse"], marker='o', color=color_map_blue[l_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['config'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['config'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('Sparsity (%)')
            
            # Second row: plots per iteration
            # Create a color map for iterations
            unique_iters = sorted(list(set([v for v in parameters["iter"]])))
            cmap = cm.Reds
            iter_colors = [cmap(i/max(1, len(unique_iters)-1)) for i in range(len(unique_iters))]
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
            if baseline and not best_models[baseline]['config'].get("test_err") is None:
                plt.axhline(y=best_models[baseline]['config'].get("test_err"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('MSE')
            if i == 0:  # Only add legend to the first plot in the row
                plt.legend()
            
            # R2 vs eps for each niter
            plt.subplot(2, 4, 6)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["R2"], marker='o', color=iter_color_map_l[iter_val])
            if triv and "R2" in triv_metrics:
                plt.axhline(y=triv_metrics["R2"], color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['config'].get("r2") is None:
                plt.axhline(y=best_models[baseline]['config'].get("r2"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('R²')
            
            # Similarity vs eps for each niter
            if correct_feats is not None:
                plt.subplot(2, 4, 7)
                for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                    iter_data = best_l_df[best_l_df["iter"] == iter_val]
                    plt.plot(iter_data["eps"], iter_data["similarity"], marker='o', color=iter_color_map_l[iter_val])
                if triv and "similarity" in triv_metrics and triv_metrics["similarity"] is not None:
                    plt.axhline(y=triv_metrics["similarity"], color='gray', linestyle='--', alpha=0.7)
                if baseline and not best_models[baseline]['config'].get("sim") is None:
                    plt.axhline(y=best_models[baseline]['config'].get("sim"), color='black', linestyle='--', alpha=0.8, label=baseline)
                plt.xlabel('ε')
                plt.ylabel('Jaccard Similarity')
            
            # Sparsity vs eps for each niter
            plt.subplot(2, 4, 8)
            for i, iter_val in enumerate(sorted(set(best_l_df["iter"]))):
                iter_data = best_l_df[best_l_df["iter"] == iter_val]
                plt.plot(iter_data["eps"], iter_data["sparse"], marker='o', color=iter_color_map_l[iter_val])
            if triv:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            if baseline and not best_models[baseline]['config'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['config'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
            plt.ylabel('Sparsity (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            
            # Save results plots
            if isinstance(plot, str):
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
        for i, method in enumerate(methods):
            if method in best_models and best_models[method]["data"]["trace"] is not None:
                trace = best_models[method]["data"]["trace"]
                plt.plot(range(len(trace)), trace, color=method_colors[i], label=method)
        plt.xlabel('Iterations')
        plt.ylabel('Training Error')
        plt.legend()
        
        # MSE vs eps for each method
        plt.subplot(1, 2, 2)
        for i, method in enumerate(methods):
            method_data = super_df[super_df["method"] == method]
            best_l = best_models[method]["config"]["l"] if method in best_models else None
            best_iter = best_models[method]["config"]["iter"] if method in best_models else None
            if best_l is not None and best_iter is not None:
                filtered_data = method_data[(method_data["l"] == best_l) & (method_data["iter"] == best_iter)]
                plt.plot(filtered_data["eps"], filtered_data["mse"], marker='o', color=method_colors[i], label=method)
        if triv and "mse" in triv_metrics:
            plt.axhline(y=triv_metrics["mse"], color='gray', linestyle='--', alpha=0.7, label='Trivial')
        if baseline and not best_models[baseline]['config'].get("test_err") is None:
            plt.axhline(y=best_models[baseline]['config'].get("test_err"), color='black', linestyle='--', alpha=0.8, label=baseline)
        plt.xlabel('ε')
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
        if baseline and not best_models[baseline]['config'].get("r2") is None:
            plt.axhline(y=best_models[baseline]['config'].get("r2"), color='black', linestyle='--', alpha=0.8, label=baseline)
        plt.xlabel('ε')
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
            if baseline and not best_models[baseline]['config'].get("sim") is None:
                plt.axhline(y=best_models[baseline]['config'].get("sim"), color='black', linestyle='--', alpha=0.8, label=baseline)
            plt.xlabel('ε')
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
        if baseline and not best_models[baseline]['config'].get("sparsity") is None:
                plt.axhline(y=best_models[baseline]['config'].get("sparsity"), color='black', linestyle='--', alpha=0.8, label=baseline)
        plt.xlabel('ε')
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
        if baseline and not best_models[baseline]['config'].get("train_err") is None:
                plt.axhline(y=best_models[baseline]['config'].get("train_err"), color='black', linestyle='--', alpha=0.8, label=baseline)
        plt.xlabel('ε')
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
