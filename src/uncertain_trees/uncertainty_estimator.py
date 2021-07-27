import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet, EmpiricalCovariance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from typing import Dict, List

from uncertain_trees.utils import sigmoid


def get_cov_estimator_from_string(cov_estimator_str: str):
    if cov_estimator_str.lower() in ["shrunk"]:
        return LedoitWolf
    elif cov_estimator_str.lower() in ["robust"]:
        return MinCovDet
    elif cov_estimator_str.lower() in ["empirical"]:
        return EmpiricalCovariance
    else:
        raise ValueError(
            f"Unrecognized covariance estimator string: '{cov_estimator_str}'"
        )


class SuperDummyCovEstimator(EmpiricalCovariance):
    def __init__(
        self,
        location,
        scales: np.ndarray = None,
        precision: np.ndarray = None,
    ):
        super().__init__(store_precision=True, assume_centered=False)
        self.location_ = location
        if scales is not None:
            if scales.ndim == 2:
                self.covariance_ = scales
                self.precision_ = (
                    precision if precision is not None else np.linalg.pinv(scales)
                )
            else:
                self.covariance_ = np.diag(scales)
                self.precision_ = np.diag(1 / scales)
        else:
            self.covariance_ = np.eye(location.size)
            self.precision_ = np.eye(location.size)


class DummyCovEstimator(EmpiricalCovariance):
    def __init__(self, X):
        super().__init__(store_precision=True, assume_centered=False)
        X = self._validate_data(X)
        self.location_ = X.mean(axis=0)
        if X.shape[0] > 1:
            variances = X.var(axis=0)
            self.covariance_ = np.diag(variances)
            self.precision_ = np.diag(1 / variances)
        else:
            self.covariance_ = np.eye(X.shape[1])
            self.precision_ = np.eye(X.shape[1])


class UncertaintyEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        cov_estimator: str = "shrunk",
        classes: np.ndarray = None,
        normalize: bool = False,
        original_qda: bool = False,
    ):
        self.priors = None
        self.class_covariances = None
        self.global_covariance = None
        self.classes = classes
        self.class_scalings = None
        self.global_scaling = None
        self.cov_estimator = get_cov_estimator_from_string(cov_estimator)
        self.normalize = normalize
        self.scaler = StandardScaler(copy=False)
        self.original_qda = original_qda

    @classmethod
    def create_from_splits(
        cls,
        num_path: List[int],
        splits: Dict[int, List[float]],
        location: np.ndarray = None,
        cov_estimator: str = "shrunk",
        classes: np.ndarray = None,
        normalize: bool = False,
    ):
        unc_est = cls(cov_estimator=cov_estimator, classes=classes, normalize=normalize)
        if location is None:
            location = np.array(
                [np.mean(svs) for f, svs in splits.items() if f in num_path]
            )
        feat_scales = np.array(
            [np.var(svs) for f, svs in splits.items() if f in num_path]
        )

        feat_scales[np.isclose(feat_scales, 0)] = 1  # Make sure that the variance is
        # never 0.
        if normalize:
            unc_est.scaler.mean_ = location
            unc_est.scaler.scale_ = np.sqrt(feat_scales)
            new_location = np.zeros_like(location)
            new_scales = np.ones_like(feat_scales)
        else:
            new_location = location
            new_scales = feat_scales

        unc_est.class_covariances = [
            SuperDummyCovEstimator(new_location, new_scales)
        ] * classes.size
        unc_est.class_scalings = [
            np.linalg.det(cov_est.covariance_) for cov_est in unc_est.class_covariances
        ]
        unc_est.global_covariance = SuperDummyCovEstimator(new_location, new_scales)
        unc_est.global_scaling = np.linalg.det(unc_est.global_covariance.covariance_)
        unc_est.priors = np.ones(classes.shape) / classes.size
        unc_est.cov_estimator_cls = None
        return unc_est

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        n_samples, n_features = X.shape
        check_classification_targets(y)
        if self.normalize:
            self.scaler.fit(X)
        X_scaled = self.scaler.transform(np.unique(X, axis=0)) if self.normalize else X
        if self.classes is not None:
            y = np.where(y == self.classes[1], 1, 0)
            self.priors = np.array([(y == 0).sum(), y.sum()]) / y.size
        else:
            self.classes, y = np.unique(y, return_inverse=True)
            self.priors = np.bincount(y) / float(n_samples)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than"
                " one; got %d class" % (n_classes)
            )

        if X_scaled.shape[0] <= X_scaled.shape[1]:
            self.global_covariance = DummyCovEstimator(X_scaled)
        else:
            cov_estimator = self.cov_estimator(
                store_precision=True, assume_centered=False
            )
            self.global_covariance = cov_estimator.fit(X_scaled)

        self.global_scaling = np.linalg.det(self.global_covariance.covariance_)

        class_covariances = []
        # Learn multivariate Gaussian's for each of the classes.
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            if Xg.shape[0] == 0:
                class_covariances.append(
                    SuperDummyCovEstimator(
                        location=self.global_covariance.location_,
                        scales=self.global_covariance.covariance_,
                        precision=self.global_covariance.precision_,
                    )
                )
            else:
                Xg_scaled = (
                    self.scaler.transform(np.unique(Xg, axis=0))
                    if self.normalize
                    else Xg
                )
                if (
                    Xg_scaled.shape[0] == 1
                    or np.isclose(Xg_scaled.var(axis=0), 0).sum() > 0
                ):
                    class_covariances.append(
                        SuperDummyCovEstimator(
                            location=Xg_scaled.mean(axis=0),
                            scales=self.global_covariance.covariance_,
                            precision=self.global_covariance.precision_,
                        )
                    )
                elif Xg_scaled.shape[0] <= Xg_scaled.shape[1]:
                    class_covariances.append(DummyCovEstimator(Xg_scaled))
                else:
                    cov_estimator = self.cov_estimator(
                        store_precision=True, assume_centered=False
                    )
                    class_covariances.append(cov_estimator.fit(Xg_scaled))

        self.class_covariances = class_covariances
        self.class_scalings = [
            np.linalg.det(cov_est.covariance_) for cov_est in class_covariances
        ]

        return self

    def class_squared_mahalanobis(self, X):
        X = check_array(X)
        X_scaled = self.scaler.transform(X) if self.normalize else X
        norm2 = []
        for ind in range(len(self.classes)):
            norm2.append(self.class_covariances[ind].mahalanobis(X_scaled))

        return np.array(norm2).T  # shape = [len(X), n_classes]

    def class_log_posteriors(self, X):
        p = X.shape[1]
        u = np.asarray([np.log(s) if s > 0 else 1e-4 for s in self.class_scalings])
        v = p * np.log(np.full_like(u, np.pi))
        return -0.5 * (self.class_squared_mahalanobis(X) + u + v) + np.log(
            self.priors, out=np.full(2, -1e4), where=self.priors > 0
        )

    def class_posteriors(self, X):
        return np.exp(self.class_log_posteriors(X))

    def class_uncertainties(self, X):
        return (norm.cdf(np.sqrt(self.class_squared_mahalanobis(X))) - 0.5) * 2

    def global_squared_mahalanobis(self, X):
        X = check_array(X)
        X_scaled = self.scaler.transform(X) if self.normalize else X
        return self.global_covariance.mahalanobis(X_scaled)

    def global_log_likelihood(self, X):
        norm2 = self.global_squared_mahalanobis(X)
        scaling = np.log(self.global_scaling) if self.global_scaling > 0 else 1e4

        return -0.5 * (norm2 + scaling)

    def global_likelihood(self, X):
        return np.exp(self.global_log_likelihood(X))

    def global_uncertainty(self, X):
        return (norm.cdf(np.sqrt(self.global_squared_mahalanobis(X))) - 0.5) * 2

    def decision_function(self, X):
        class_log_posteriors = self.class_log_posteriors(X)
        return class_log_posteriors[:, 1] - class_log_posteriors[:, 0]

    def predict_proba(self, X):
        if self.original_qda:
            proba = sigmoid(self.decision_function(X))
        else:
            class_posteriors = self.class_posteriors(X)
            proba = np.exp(class_posteriors[:, 1]) / (
                np.exp(class_posteriors[:, 0]) + np.exp(class_posteriors[:, 1])
            )
        return np.hstack([1 - proba.reshape(-1, 1), proba.reshape(-1, 1)])

    def predict_priors(self, X):
        return self.priors

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5
