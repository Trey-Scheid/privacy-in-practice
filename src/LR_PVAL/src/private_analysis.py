import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import re
import glob
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
from autodp.calibrator_zoo import eps_delta_calibrator
from scipy.stats import logistic
from utils import get_data_fps


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
    def __init__(self, data_fp, epsilon, delta, verbose=False):
        self.global_sensitivity = 1
        self.df = pd.read_csv(data_fp)
        self.epsilon = epsilon
        proportion_fitting = 0.75
        self.ep_fitting = self.epsilon * proportion_fitting
        self.ep_pval = self.epsilon * (1 - proportion_fitting)
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
        mech = calibrate(NoisyGD_fix_sigma, self.ep_fitting, self.delta, [0, 500000])
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

    # def permutation_test(self, n_permutations=1000, alpha=0.05):
    #     if self.theta is None:
    #         raise ValueError("theta must be computed before permutation test")

    #     def compute_test_statistic(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    #         logits = X @ theta
    #         probs = logistic.cdf(logits)
    #         ll = np.sum(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))

    #         return -2 * ll

    #     observed_stat = compute_test_statistic(self.X, self.y, self.theta)

    #     sensitivity = np.sqrt(2 * np.log(1.25/self.delta)) / self.ep_pval

    #     null_distribution = []
    #     for _ in range(n_permutations):
    #         y_perm = np.random.permutation(self.y)
    #         dp_noise = np.random.normal(0, sensitivity)
    #         perm_stat = compute_test_statistic(self.X, y_perm, self.theta) + dp_noise
    #         null_distribution.append(perm_stat)

    #     null_distribution = np.array(null_distribution)
    #     p_value = np.mean(null_distribution >= observed_stat)

    #     return p_value

    # TODO: some sort of p-value calculation

    def fit(self, sigma=300.0, log_gap=10, mid_results=True):
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


def get_all_private_lr_results(
    epsilon=2,
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
    get_all_private_lr_results(verbose=True, log_gap=2, mid_results=True)
