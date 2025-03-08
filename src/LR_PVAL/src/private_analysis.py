import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import re
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
from autodp.calibrator_zoo import eps_delta_calibrator
from utils import get_data_fps
import argparse
import concurrent.futures
from functools import partial
import glob


class NoisyGD_mech(Mechanism):
    def __init__(self, sigma, coeff, name="NoisyGD"):
        Mechanism.__init__(self)
        self.name = name
        self.params = {"sigma": sigma, "coeff": coeff}
        gm = GaussianMechanism(sigma, name="Release_gradient")
        compose = ComposeGaussian()
        mech = compose([gm], [coeff])

        self.set_all_representation(mech)


class private_analysis:
    def __init__(
        self, data_fp=None, epsilon=2, delta=1e-6, verbose=False, X=None, y=None
    ):
        if data_fp is not None:
            self.df = pd.read_csv(data_fp)
        elif X is not None and y is not None:
            self.df = pd.DataFrame({"has_corrected_error": X, "has_bugcheck": y})
        else:
            raise ValueError("X and y must be provided if data_fp is not provided")

        self.global_sensitivity = 1
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose
        self.process_data()

    def process_data(self):
        if (
            self.df.groupby(["has_corrected_error", "has_bugcheck"]).size().shape[0]
            != 4
        ):
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame({"has_corrected_error": [1], "has_bugcheck": [0]}),
                ],
                ignore_index=True,
            )
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame({"has_corrected_error": [1], "has_bugcheck": [1]}),
                ],
                ignore_index=True,
            )

        self.X = self.df[["has_corrected_error"]]
        self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        self.X = self.global_sensitivity * preprocessing.normalize(self.X, norm="l2")

        self.y = self.df["has_bugcheck"]

        self.n = self.X.shape[0]

    def cross_entropy_loss(self, score, y):
        log_phat = np.zeros_like(score)
        log_one_minus_phat = np.zeros_like(score)
        mask = score > 0
        log_phat[mask] = -np.log(1 + np.exp(-score[mask]))
        log_phat[~mask] = score[~mask] - np.log(1 + np.exp(score[~mask]))
        log_one_minus_phat[~mask] = -np.log(1 + np.exp(score[~mask]))

        return -y * log_phat - (1 - y) * log_one_minus_phat

    def loss(self, theta):
        return np.sum(self.cross_entropy_loss(self.X @ theta, self.y)) / self.n

    def err(self, theta):
        return np.sum((self.X @ theta > 0) != self.y) / self.n

    def gradient(self, theta):
        grad = np.zeros(shape=(self.X.shape[1],))

        phat = np.exp(self.X @ theta) / (1 + np.exp(self.X @ theta))
        grad = self.X[self.y == 0, :].T @ (phat[self.y == 0]) - self.X[
            self.y == 1, :
        ].T @ (1 - phat[self.y == 1].T)
        return grad

    def GS_bound(self, theta):
        """
        Calculate global sensitivity for a mini-batch.
        """
        bound = np.linalg.norm(theta)
        global_sensitivity = self.global_sensitivity / (1 + np.exp(-bound))
        return global_sensitivity

    def find_appropriate_niter(self, sigma):
        NoisyGD_fix_sigma = lambda x: NoisyGD_mech(sigma, x)
        calibrate = eps_delta_calibrator()
        mech = calibrate(NoisyGD_fix_sigma, self.epsilon, self.delta, [0, 500000])
        niter = int(np.floor(mech.params["coeff"]))
        return niter

    def theoretical_lr_choice(self, beta_L, f0_minus_fniter_bound, dim, sigma, niter):
        return np.minimum(
            1 / beta_L,
            np.sqrt(2 * f0_minus_fniter_bound / (dim * sigma**2 * beta_L * niter)),
        )

    def run_NoisyGD_step(self, theta, sigma, lr):
        global_sensitivity = self.GS_bound(theta)
        return theta - lr * (
            self.gradient(theta)
            + global_sensitivity * sigma * np.random.normal(size=theta.shape)
        )

    def run_NoisyGD(self, sigma, lr, niter, log_gap=10, mid_results=True):
        theta_GD = np.zeros(shape=(self.X.shape[1],))
        prev = theta_GD.copy()

        log_counter = 1
        results = {}
        for i in range(niter):
            theta_GD = self.run_NoisyGD_step(theta_GD, sigma, lr)
            prev = ((prev * i) / (i + 1)) + (theta_GD / (i + 1))
            if i == 0:
                continue
            if not i % log_counter or i == niter - 1:
                mech = NoisyGD_mech(sigma, i + 1)
                iteration = i
                curr_epsilon = mech.approxDP(self.delta)
                curr_loss = self.loss(theta_GD)
                curr_params = theta_GD

                if mid_results:
                    results[iteration] = {
                        "epsilon": curr_epsilon,
                        "loss": curr_loss,
                        "params": curr_params,
                    }
                if self.verbose:
                    print(
                        "iteration",
                        iteration,
                        "epsilon",
                        curr_epsilon,
                        "loss",
                        curr_loss,
                        "params",
                        curr_params,
                    )
                log_counter *= log_gap

        return theta_GD, self.loss(theta_GD), results

    def fit(
        self, sigma=300.0, log_gap=10, mid_results=True
    ) -> tuple[np.ndarray, float, list]:
        beta = 1 / 4 * self.n

        f0_minus_fniter_bound = self.n * (-np.log(0.5))

        niter = self.find_appropriate_niter(sigma)
        lr = self.theoretical_lr_choice(
            beta, f0_minus_fniter_bound, self.X.shape[1], sigma, niter
        )
        if self.verbose:
            print("niter", niter, "lr", lr)
        self.theta, loss, results = self.run_NoisyGD(
            sigma, lr, niter, log_gap, mid_results
        )

        return self.theta, loss, results


def run_single_permutation(raw_X, raw_y, epsilon, delta, verbose, log_gap, mid_results):
    """Run a single permutation"""
    shuffled_y = np.random.permutation(raw_y)
    return private_analysis(
        X=raw_X, y=shuffled_y, epsilon=epsilon, delta=delta, verbose=verbose
    ).fit(log_gap=log_gap, mid_results=mid_results)[2]


def save_progress(results, data_dir, bugcheck_id):
    """Save results to either in-progress or final directory"""
    results_df = pd.DataFrame(results)

    os.makedirs(os.path.join(data_dir, "permutation_results"), exist_ok=True)
    final_path = os.path.join(data_dir, "permutation_results", f"{bugcheck_id}.csv")
    results_df.to_csv(final_path, index=False)


def get_permutaiton_results(
    epsilon=1.5,
    delta=1e-6,
    verbose=False,
    data_fp=None,
    data_dir="private_data/",
    log_gap=10,
    mid_results=True,
    n_permutations=100,
    n_workers=None,  # New parameter for controlling parallelism
):
    data_dir = os.path.abspath(data_dir)
    if data_fp is None:
        done = glob.glob(os.path.join(data_dir, "permutation_results", "*.csv"))
        done = [int(re.findall(r"(\d+)", fp)[0]) for fp in done]
        data_fps = get_data_fps(data_dir)
        data_fps = [
            fp
            for fp in data_fps
            if int(re.findall(r"bugcheck_(\d+)", fp)[0]) not in done
        ]
        print(f"Running {len(data_fps)} permutations ", data_fps)
    else:
        data_fps = [data_fp]

    all_results = {}

    for data_fp in data_fps:
        print(f"Processing {data_fp}")
        df = pd.read_csv(data_fp)
        raw_X = df["has_corrected_error"]
        raw_y = df["has_bugcheck"]
        bugcheck_id = int(re.findall(r"bugcheck_(\d+)", data_fp)[0])

        if bugcheck_id == 0 or raw_X.shape[0] > 300000:
            continue

        # Partial function with fixed parameters
        run_permutation = partial(
            run_single_permutation,
            raw_X,
            raw_y,
            epsilon,
            delta,
            verbose,
            log_gap,
            mid_results,
        )

        results = []
        # Use ProcessPoolExecutor for CPU parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(run_permutation): i for i in range(n_permutations)
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Save progress every 10 permutations
                    if completed % 10 == 0:
                        save_progress(results, data_dir, bugcheck_id)
                        print(f"Saved progress: {completed} / {n_permutations}")
                    elif completed == n_permutations:
                        save_progress(results, data_dir, bugcheck_id)
                        print(
                            f"Finished all {n_permutations} permutations for bugcheck {bugcheck_id}"
                        )

                except Exception as e:
                    print(f"Permutation {idx} generated an exception: {e}")

        all_results[bugcheck_id] = results

    return all_results


def get_all_private_lr_results(
    epsilon=1.5,
    delta=1e-6,
    verbose=False,
    log_gap=10,
    mid_results=True,
):
    data_dir = "private_data/"
    data_dir = os.path.abspath(data_dir)
    data_fps = get_data_fps(data_dir)

    results = {}
    for data_fp in data_fps:
        bugcheck_id = int(re.findall(r"bugcheck_(\d+)", data_fp)[0])
        results[bugcheck_id] = private_analysis(
            data_fp, epsilon, delta, verbose=verbose
        ).fit(log_gap=log_gap, mid_results=mid_results)
        print(f"Finished {bugcheck_id}")

    pd.DataFrame(results).T.to_csv(os.path.join(data_dir, "private_analysis.csv"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run private analysis with various options"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "perm"],
        default="all",
        help='Mode to run: "all" for all results, "perm" for permutation results',
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Specific file to analyze (optional)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations to run (default: 1000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--log-gap", type=int, default=10, help="Log gap for results (default: 10)"
    )
    parser.add_argument(
        "--mid-results",
        action="store_true",
        default=True,
        help="Whether to store intermediate results",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: use all available cores)",
    )

    args = parser.parse_args()

    if args.mode == "all":
        get_all_private_lr_results(
            verbose=args.verbose, log_gap=args.log_gap, mid_results=args.mid_results
        )
    else:  # perm mode
        get_permutaiton_results(
            verbose=args.verbose,
            log_gap=args.log_gap,
            mid_results=args.mid_results,
            n_permutations=args.n_permutations,
            data_fp=args.file,
            n_workers=args.n_workers,
        )
