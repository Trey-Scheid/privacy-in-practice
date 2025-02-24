from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from autodp.calibrator_zoo import eps_delta_calibrator
from sklearn.model_selection import train_test_split
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
import matplotlib.pyplot as plt
import seaborn as sns


class GD:
    def clean(self, df):
        df = pd.get_dummies(df, columns=["channel"])
        dataset = df
        self.X = dataset[[x for x in dataset.columns if x != "anomaly"]]
        self.X = 1 * preprocessing.normalize(self.X, norm="l2")
        self.y = dataset["anomaly"]
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]

    def regGD(self):
        clf = LogisticRegression(random_state=0, fit_intercept=False).fit(
            self.X, self.y
        )
        yhat = clf.predict(self.X)
        self.err_nonprivate = self.err_yhat(yhat)
        self.err_trivial = min(np.mean(self.y), 1 - np.mean(self.y))

    def __init__(self, fp, epsilons, delta):
        df = pd.read_csv(fp)
        self.clean(df)
        self.delta = delta
        self.filepath = fp
        self.epsilon = max(epsilons)
        self.GS = 1
        self.regGD()

    def CE(self, score, y):
        # numerically efficient vectorized implementation of CE loss
        log_phat = np.zeros_like(score)
        log_one_minus_phat = np.zeros_like(score)
        mask = score > 0
        log_phat[mask] = -np.log(1 + np.exp(-score[mask]))
        log_phat[~mask] = score[~mask] - np.log(1 + np.exp(score[~mask]))
        log_one_minus_phat[mask] = -score[mask] - np.log(1 + np.exp(-score[mask]))
        log_one_minus_phat[~mask] = -np.log(1 + np.exp(score[~mask]))

        return -y * log_phat - (1 - y) * log_one_minus_phat

    def loss(self, theta):
        return np.sum(self.CE(self.X @ theta, self.y)) / self.n

    def err(self, theta):
        return np.sum((self.X @ theta > 0) != self.y) / self.n

    def err_yhat(self, yhat):
        return np.sum((yhat != self.y)) / self.n

    class NoisyGD_mech(Mechanism):
        def __init__(self, sigma, coeff, name="NoisyGD"):
            Mechanism.__init__(self)
            self.name = name
            self.params = {"sigma": sigma, "coeff": coeff}
            gm = GaussianMechanism(sigma, name="Release_gradient")
            compose = ComposeGaussian()
            mech = compose([gm], [coeff])

            self.set_all_representation(mech)

    def gradient(self, theta):
        grad = np.zeros(shape=(self.dim,))

        phat = np.exp(self.X @ theta) / (1 + np.exp(self.X @ theta))
        grad = self.X[self.y == 0, :].T @ (phat[self.y == 0]) - self.X[
            self.y == 1, :
        ].T @ (1 - phat[self.y == 1].T)
        return grad

    def GS_bound(self, theta):
        """
        Calculate global sensitivity for a mini-batch.
        """
        GS = 100
        bound = np.linalg.norm(theta)
        GS = self.GS / (1 + np.exp(-bound))
        return GS

    def run_NoisyGD_step(self, theta, sigma, lr):
        GS = self.GS_bound(theta)
        return theta - lr * (
            self.gradient(theta) + GS * sigma * np.random.normal(size=theta.shape)
        )

    def run_NoisyGD(self, sigma, lr, niter, log_gap=10):
        theta_GD = np.zeros(shape=(self.dim,))
        prev = theta_GD.copy()
        err_GD = []
        eps_GD = []

        for i in range(niter):
            theta_GD = self.run_NoisyGD_step(theta_GD, sigma, lr)
            prev = ((prev * i) / (i + 1)) + (theta_GD / (i + 1))
            if i != 0:
                if not i % log_gap:
                    mech = self.NoisyGD_mech(sigma, i + 1)
                    eps_GD.append(mech.approxDP(self.delta))
                    err_GD.append(self.err(prev))
        return err_GD, eps_GD

    def run_nonprivate_GD(self, lr, niter, log_gap=10):
        theta_GD = np.zeros(shape=(self.dim,))
        prev = theta_GD.copy()
        err_GD = []
        for i in range(niter):
            theta_GD = self.run_NoisyGD_step(theta_GD, 0, lr)
            prev = ((prev * i) / (i + 1)) + (theta_GD / (i + 1))
            if i != 0:
                if not i % log_gap:
                    err_GD.append(self.err(prev))
        return err_GD

    def find_appropriate_niter(self, sigma, eps, delta):
        NoisyGD_fix_sigma = lambda x: self.NoisyGD_mech(sigma, x)
        calibrate = eps_delta_calibrator()
        mech = calibrate(NoisyGD_fix_sigma, eps, delta, [0, 500000])
        niter = int(np.floor(mech.params["coeff"]))
        return niter

    def theoretical_lr_choice(self, beta_L, f0_minus_fniter_bound, dim, sigma, niter):

        return np.minimum(
            1 / beta_L,
            np.sqrt(2 * f0_minus_fniter_bound / (dim * sigma**2 * beta_L * niter)),
        )

    def diffNoisePlot(self, epsilon):
        beta = 1 / 4 * self.n

        f0_minus_fniter_bound = self.n * (-np.log(0.5))
        sigma = 300.0
        niter = self.find_appropriate_niter(sigma, epsilon, self.delta)
        lr = self.theoretical_lr_choice(
            beta, f0_minus_fniter_bound, self.dim, sigma * self.GS, niter
        )

        err_GD1, eps_GD1 = self.run_NoisyGD(sigma, lr, niter)
        self.err_GD1 = err_GD1
        self.eps_GD1 = eps_GD1
        sigma = 30
        niter = self.find_appropriate_niter(sigma, epsilon, self.delta)
        lr = self.theoretical_lr_choice(
            beta, f0_minus_fniter_bound, self.dim, sigma * self.GS, niter
        )
        err_GD2, eps_GD2 = self.run_NoisyGD(sigma, lr, niter)

        err_GD0 = self.run_nonprivate_GD(1 / beta, niter)
        sns.set(style="whitegrid", context="talk")

        plt.figure(figsize=(8, 5))
        plt.plot(eps_GD1, err_GD1, "g.-")
        plt.plot(eps_GD2, err_GD2, "c.-")
        plt.plot(eps_GD1, self.err_nonprivate * np.ones_like(eps_GD1), "k--")
        plt.plot(eps_GD1, self.err_trivial * np.ones_like(eps_GD1), "r--")
        plt.plot(eps_GD2, err_GD0, "b--")

        plt.legend(
            [
                "NoisyGD-large-noise-more-iter",
                "NoisyGD-small-noise-fewer-iter",
                "Nonprivate-sklearn",
                "Trivial",
                "Non-private-GD",
            ],
            fontsize="x-small",
            loc="center left",  # Place it to the left of the anchor point
            bbox_to_anchor=(1, 0.5),  # Anchor the legend to the right of the plot
            ncol=1,
        )
        plt.xlabel("Epsilon")
        plt.ylabel("Error")
        plt.title("Error vs. Epsilon for Different NoisyGD Configurations")
        plt.tight_layout()
        plt.savefig("plots/GDNoise.png")

    def diffLearningRatesPlot(self, epsilon):
        sigma = 300.0
        delta = 1e-6
        beta = 1 / 4 * self.n
        GS = self.GS
        f0_minus_fniter_bound = self.n * (-np.log(0.5))
        niter = self.find_appropriate_niter(sigma, epsilon, delta)
        theoretical_lr = self.theoretical_lr_choice(
            beta, f0_minus_fniter_bound, self.dim, sigma * GS, niter
        )
        lr = 10 * theoretical_lr
        err_GD3, eps_GD3 = self.run_NoisyGD(sigma, lr, niter, log_gap=100)
        lr = 0.1 * theoretical_lr
        err_GD4, eps_GD4 = self.run_NoisyGD(sigma, lr, niter, log_gap=100)
        lr = 100 * theoretical_lr
        err_GD5, eps_GD5 = self.run_NoisyGD(sigma, lr, niter, log_gap=100)
        sns.set(style="whitegrid", context="talk")
        sns.color_palette("flare")

        plt.figure(figsize=(8, 5))
        plt.plot(self.eps_GD1, self.err_GD1, "g.-")
        plt.plot(eps_GD3, err_GD3, "c--")
        plt.plot(eps_GD4, err_GD4, "m:")
        plt.plot(eps_GD5, err_GD5, "b:")
        plt.plot(self.eps_GD1, self.err_nonprivate * np.ones_like(self.eps_GD1), "k--")
        plt.plot(self.eps_GD1, self.err_trivial * np.ones_like(self.eps_GD1), "r--")

        plt.legend(
            [
                "NoisyGD",
                "NoisyGD-lr*10",
                "NoisyGD-lr/10",
                "NoisyGD-lr*100",
                "Nonprivate-sklearn",
                "Trivial",
            ],
            fontsize="x-small",
            loc="center left",  # Place it to the left of the anchor point
            bbox_to_anchor=(1, 0.5),  # Anchor the legend to the right of the plot
            ncol=1,  # Single column layout
        )
        plt.xlabel("Epsilon")
        plt.ylabel("Error")
        plt.title("Error vs. Epsilon for Noisy GD Variants")
        plt.tight_layout()
        plt.savefig("plots/GDLearningRates.png")

    def run_all_plots(self):
        self.diffNoisePlot(self.epsilon)
        self.diffLearningRatesPlot(self.epsilon)
        return pd.DataFrame()
