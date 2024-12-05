import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from typing import List, Literal
from numpy import ndarray
from scipy.stats import norm
from scipy.optimize import root_scalar


class ObjPert:
    def __init__(self, fp, eps: List[float] = [], delta: float = None):
        self.fp = fp
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.dim,
            self.n,
            self.x_bound,
        ) = self.clean_data()
        self.eps = eps
        self.delta = delta
        self.df = None

    def clean_data(self):
        dataset = pd.read_csv("satellite.csv")
        X = dataset.iloc[:, 5:].values
        y = dataset.iloc[:, 1].values

        dim = X.shape[1]
        n = X.shape[0]

        X = X @ np.diag(
            1.0
            / np.array(
                [
                    600,
                    500,
                    1e-1,
                    1e-2,
                    1e-2,
                    1,
                    1,
                    1,
                    6,
                    3,
                    90,
                    80,
                    1e-4,
                    1e-3,
                    600,
                    500,
                    1e-4,
                    1e-4,
                ]
            )
        )  # Not supposed to use the data

        # clip data!
        x_bound = 1
        X = x_bound * preprocessing.normalize(X, norm="l2")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        return X_train, X_test, y_train, y_test, dim, n, x_bound

    def demo_data(self, trials: int = 50, repeats: int = 10):
        if self.eps == []:
            raise ValueError("Epsilon values not set")
        if self.delta is None:
            raise ValueError("Delta value not set")

        print("Starting objective perturbation")
        np.random.seed(0)

        L = self.x_bound
        beta = self.x_bound**2 / 4

        # cvxpy variables
        lamb = cp.Parameter(nonneg=True)
        sigma = cp.Parameter(nonneg=True)
        theta = cp.Variable(self.dim)

        log_likelihood = cp.sum(
            cp.multiply(self.y_train, self.X_train @ theta)
            - cp.logistic(self.X_train @ theta)
        )

        data = []

        lambda_vals = np.logspace(np.log10(beta + 1e-4), np.log10(4), trials)

        for i in range(trials):
            lamb.value = lambda_vals[i]

            for _ in range(repeats):
                unscaled_b = np.random.normal(0, 1, self.dim)

                for epsilon in self.eps:
                    try:
                        sigma.value = ObjPert.sigma_(
                            epsilon, self.delta, beta, lamb.value, L
                        )
                    except ValueError:
                        continue

                    objpert_rand = (unscaled_b * sigma.value) @ theta
                    problem = cp.Problem(
                        cp.Maximize(
                            log_likelihood
                            - lamb / 2 * cp.norm(theta, 2) ** 2
                            - objpert_rand
                        )
                    )

                    try:
                        problem.solve(solver=cp.ECOS)
                    except cp.error.SolverError:
                        continue
                    if theta.value is None:
                        continue

                    test_error = ObjPert.error((self.X_test @ theta.value), self.y_test)
                    test_loss = ObjPert.non_private_loss(
                        self.X_test, self.y_test, theta.value, lamb.value
                    )
                    data.append([epsilon, lamb.value, test_error, test_loss])

        # Nonrpivate loss
        non_private_problem = cp.Problem(
            cp.Maximize(log_likelihood - lamb / 2 * cp.norm(theta, 2) ** 2)
        )
        epsilon = "Non-private"

        for i in range(trials):
            lamb.value = lambda_vals[i]
            try:
                non_private_problem.solve(solver=cp.ECOS)
            except cp.error.SolverError:
                continue
            if theta.value is None:
                continue

            test_error = ObjPert.error((self.X_test @ theta.value), self.y_test)
            test_loss = ObjPert.non_private_loss(
                self.X_test, self.y_test, theta.value, lamb.value
            )
            data.append([epsilon, lamb.value, test_error, test_loss])

        self.df = pd.DataFrame(
            data, columns=["epsilon", "lambda", "test_error", "test_loss"]
        )

        self.df = self.df.sort_values(
            "epsilon",
            key=lambda x: x.map(
                {ep: i for i, ep in enumerate(self.eps + ["Non-private"])}
            ),
        )

        self.df["Epsilon"] = self.df["epsilon"].astype(str)

        return self.df

    def epsilon_error_df(self):
        if self.df is None:
            raise ValueError("Run the demo_data method first to run model")

        lamb = cp.Parameter(nonneg=True)
        sigma = cp.Parameter(nonneg=True)
        theta = cp.Variable(self.dim)

        best = []
        best_params = self.tune_hypterparameters()
        noise = np.random.normal(0, 1, self.dim)
        for ep in self.eps:
            lamb.value = best_params[str(ep)]
            sigma.value = ObjPert.sigma_(ep, self.delta, self.beta, lamb.value, self.L)
            objpert_rand = (noise * sigma.value) @ theta
            problem = cp.Problem(
                cp.Maximize(
                    cp.sum(
                        cp.multiply(self.y_train, self.X_train @ theta)
                        - cp.logistic(self.X_train @ theta)
                    )
                    - lamb / 2 * cp.norm(theta, 2) ** 2
                    - objpert_rand
                )
            )

            try:
                problem.solve(solver=cp.ECOS)
            except cp.error.SolverError:
                continue
            if theta.value is None:
                continue

            test_error = ObjPert.error((self.X_test @ theta.value), self.y_test)
            best.append([ep, test_error, "Objective Perturbation"])

        return pd.DataFrame(best, columns=["epsilon", "error", "method"])

    def tune_hypterparameters(self):
        return dict(
            self.df.groupby(["epsilon", "lambda"])
            .mean()
            .groupby("epsilon")
            .idxmin()
            .values
        )

    def plot(self, kind=Literal["error", "loss"]):
        if self.df is None:
            raise ValueError("Run the demo_data method first to run model")
        plt.figure(figsize=(12, 8))
        y_axis = "Error" if kind == "error" else "Loss"
        plot = sns.lineplot(
            x="Lambda",
            y=y_axis,
            hue="Epsilon",
            data=self.df,
            errorbar=("ci", 95),
            markers=True,
            palette="flare",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Lambda (log scale)")
        plt.ylabel(f"Test {y_axis} (log scale)")
        plt.title(f"Test {y_axis} vs Lambda for Different Epsilon Values")

        plt.tight_layout()
        plt.savefig(f"plots/ObjPert{y_axis}.png")
        plt.show()
        return plot

    def sigma_lambda_plot(self, max_lambda: float = 1e2, num_lambdas: int = 100):
        eps = self.eps if self.eps != [] else [0.5, 1.0, 1.5, 2.0]
        delta = self.delta if self.delta is not None else 1e-6
        lams = np.logspace(
            np.log10(self.beta + 1e-4), np.log10(max_lambda), num_lambdas
        )
        sigmas = [
            [ep, lam, ObjPert.sigma_(ep, delta, self.beta, lam, self.L)]
            for ep in eps
            for lam in lams
        ]
        df = pd.DataFrame(sigmas, columns=["epsilon", "lambda", "sigma"])

        plot = sns.lineplot(
            x="lambda", y="sigma", hue="epsilon", data=df, palette="flare"
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Lambda (log scale)")
        plt.ylabel("Sigma (log scale)")
        plt.title("Sigma vs Lambda for Different Epsilon Values")

        plt.tight_layout()
        plt.savefig(f"plots/ObjPert_SigLamb.png")
        plt.show()
        return plot

    def run_all_plots(self):
        self.demo_data()
        self.plot("error")
        self.plot("loss")
        self.sigma_lambda_plot

        return self.epsilon_error_df()

    def setEps(self, eps):
        if type(eps) == float:
            eps = [eps]
        self.eps = eps

    def setDelta(self, delta):
        self.delta = delta

    @staticmethod
    def error(scores: List[float], labels: List[float]) -> float:
        scores[scores > 0] = 1
        scores[scores <= 0] = 0
        return np.sum(np.abs(scores - labels)) / float(np.size(labels))

    @staticmethod
    def non_private_loss(X: ndarray, y: ndarray, theta: ndarray, lamb: float) -> float:
        log_likelihood = np.sum(y * (X @ theta) - np.log(1 + np.exp(X @ theta)))
        return -log_likelihood + lamb / 2 * np.linalg.norm(theta, 2) ** 2

    @staticmethod
    def delta_(
        epsilon: float, beta: float, lambd: float, L: float, sigma: float
    ) -> float:
        eps_tilde = epsilon - np.abs(np.log(1 - beta / lambd))
        eps_hat = eps_tilde - L**2 / (2 * sigma**2)

        H = lambda sub: norm.cdf(-sub * sigma / L + L / (2 * sigma)) - np.exp(
            sub
        ) * norm.cdf(-sub * sigma / L - L / (2 * sigma))

        return (
            (
                2 * H(eps_tilde)
                if eps_hat >= 0
                else (1 - np.exp(eps_hat)) + np.exp(eps_hat) * 2 * H(L**2 / sigma**2)
            ),
        )[0]

    @staticmethod
    def epsilon_(
        target_delta: float,
        beta: float,
        lambd: float,
        L: float,
        sigma: float,
        initial_guess: float = 1.0,
    ) -> float:
        def objective(epsilon):
            return ObjPert.delta_(epsilon, beta, lambd, L, sigma) - target_delta

        # Use a scalar root solver
        try:
            result = root_scalar(objective, bracket=(1e-5, 100), method="brentq")
        except ValueError:
            return np.inf
        if result.converged:
            return result.root
        else:
            raise ValueError("Solver did not converge")

    @staticmethod
    def sigma_(
        epsilon: float,
        target_delta: float,
        beta: float,
        lambd: float,
        L: float,
        initial_guess: float = 1.0,
    ) -> float:
        def objective(sigma):
            return ObjPert.delta_(epsilon, beta, lambd, L, sigma) - target_delta

        # Solve for sigma using a scalar root solver
        try:
            result = root_scalar(objective, bracket=(1e-1, 1e3), method="brentq")
        except ValueError:
            return np.inf
        if result.converged:
            return result.root
        else:
            raise ValueError("Solver did not converge for sigma")

    @staticmethod
    def lambda_(
        epsilon: float, target_delta: float, beta: float, L: float, sigma: float
    ) -> float:
        def objective(lambd):
            return ObjPert.delta_(epsilon, beta, lambd, L, sigma) - target_delta

        # Solve for lambda using a scalar root solver
        try:
            result = root_scalar(objective, bracket=(beta + 1e-4, 100), method="brentq")
        except ValueError:
            return np.inf
        if result.converged:
            return result.root
        else:
            raise ValueError("Solver did not converge for lambda")
