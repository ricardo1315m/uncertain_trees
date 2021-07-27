import warnings
from typing import Union, List, Dict, Any

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

from uncertain_trees.light_gbm import (
    LightGBM,
    lgbm_tree_to_dict,
    get_lgbm_feat_id_from_feat_desc,
)
from uncertain_trees.random_forest import tree_to_dict
from uncertain_trees.uncertainty_estimator import UncertaintyEstimator
from uncertain_trees.utils import print_input_size, timeit, binary_entropy, sigmoid


def _call_tree_leaves(
    self,
    X,
    y,
    method,
    tree_ind,
    tree_leaves,
    n_nodes_ptr,
    indicator,
    out,
):
    for node_id, leaf in tree_leaves.items():
        sparse_node_id = n_nodes_ptr[tree_ind] + node_id
        node_samples_idx = indicator[:, sparse_node_id].nonzero()[0]
        if method != "priors":
            if self.only_path_features == "yes":
                feats = leaf["feats"]
            elif self.only_path_features == "model":
                feats = self.model_features
            else:
                feats = range(X.shape[1])
            data_idx = np.ix_(node_samples_idx, feats)
        if method == "fit":
            if (
                node_samples_idx.size > 1
                and np.isclose(X[data_idx].var(axis=0), 0).sum() == 0
            ):
                self.new_leaves[tree_ind][node_id]["estimator"] = UncertaintyEstimator(
                    cov_estimator=self.cov_estimator,
                    classes=self.model.classes_,
                ).fit(X[data_idx], y[node_samples_idx])
            else:
                if node_samples_idx.size >= 1:
                    location = X[data_idx].mean(axis=0)
                else:
                    location = None
                self.new_leaves[tree_ind][node_id][
                    "estimator"
                ] = UncertaintyEstimator.create_from_splits(
                    feats,
                    self.model_splits[tree_ind],
                    location=location,
                    cov_estimator=self.cov_estimator,
                    classes=self.model.classes_,
                )
        elif node_samples_idx.size > 0 and method != "fit":
            if method == "class_posteriors":
                res = leaf["estimator"].class_posteriors(X[data_idx])
            elif method == "class_log_posteriors":
                res = leaf["estimator"].class_log_posteriors(X[data_idx])
            elif method == "class_uncertainties":
                res = leaf["estimator"].class_uncertainties(X[data_idx])
            elif method == "global_likelihood":
                res = leaf["estimator"].global_likelihood(X[data_idx]).reshape(-1, 1)
            elif method == "global_uncertainty":
                res = leaf["estimator"].global_uncertainty(X[data_idx]).reshape(-1, 1)
            elif method == "aleatoric_uncertainty":
                n_estimators = self.model.n_estimators
                res = np.repeat(
                    binary_entropy(
                        (
                            sigmoid(leaf["priors"])
                            if isinstance(self.model, LightGBM)
                            else leaf["priors"]
                        ).reshape(1, -1)
                    ),
                    node_samples_idx.size,
                    axis=0,
                )
            elif method == "priors":
                res = np.repeat(
                    leaf["priors"].reshape(1, -1), node_samples_idx.size, axis=0
                )
            else:
                raise NotImplementedError(f"Method '{method}' not implemented.")
            out[node_samples_idx, :, tree_ind] = res


class TreePruner(ClassifierMixin):
    def __init__(
        self,
        tree_based_model: Union[RandomForestClassifier, LightGBM],
        max_depth: int = 6,
        min_samples: int = 10,
        verbose: bool = False,
    ):
        self.model = tree_based_model
        self.max_depth = min(max_depth, self.model.max_depth or max_depth)
        self.min_samples = self.get_min_samples(min_samples)
        self.model_depths_and_paths: List[List[Dict[str, Any]]] = []
        self.model_splits: List[Dict[int, List[float]]] = []
        self.new_leaves: List[Dict[int, Dict[str, Any]]] = []
        self.list_of_features = []
        self.model_features = []
        self.n_features = None
        self.verbose = verbose
        if self.min_samples < 2 * self.max_depth:
            warnings.warn(
                "Minimum number of samples is less than the twice the max depth."
            )
            # self.min_samples = 2 * self.max_depth

    @property
    def n_estimators(self):
        if isinstance(self.model, RandomForestClassifier):
            return self.model.n_estimators
        elif isinstance(self.model, LightGBM):
            return self.model.n_estimators

    @property
    def model_min_samples(self):
        if isinstance(self.model, RandomForestClassifier):
            return min(self.model.min_samples_split, self.model.min_samples_leaf)
        elif isinstance(self.model, LightGBM):
            return self.model.min_data_in_leaf

    def get_min_samples(self, min_samples):
        return max(min_samples, self.model_min_samples or 1)

    @property
    def model_is_fitted(self):
        attrs = [
            v for v in vars(self.model) if v.endswith("_") and not v.startswith("__")
        ]
        return len(attrs) > 0

    def fit(self, X, y, force_fit: bool = False):
        if self.verbose:
            print(
                f"Fitting TreeBasedUncertainty with model: "
                f"{self.model.__class__.__name__}"
            )
        if self.model_depths_and_paths:
            warnings.warn("Model already fitted. Re-writing fitted attributes")
            self.model_depths_and_paths = []
            self.model_splits = []
            self.new_leaves = []

        self.n_features = X.shape[1]
        self.list_of_features = list(range(self.n_features))

        if not self.model_is_fitted or force_fit:
            self.model.fit(X, y)
        self._extract_model_depths_and_paths()
        self._set_new_leaves()
        all_split_features = [
            f
            for tree in self.model_depths_and_paths
            for node in tree
            for f in node["node_path_features"]
        ]
        self.model_features = sorted(set(all_split_features))

        return self

    def _extract_model_depths_and_paths(self):
        if isinstance(self.model, RandomForestClassifier):
            for i, model_estimator in enumerate(self.model.estimators_):
                self.model_depths_and_paths.append([])
                self.model_splits.append({})
                tree_to_dict(
                    model_estimator,
                    tree_out=self.model_depths_and_paths[i],
                    splits_out=self.model_splits[i],
                )
        elif isinstance(self.model, LightGBM):
            model_df = self.model.lgbm_model_.trees_to_dataframe()
            for i, tree_df in model_df.groupby("tree_index"):
                tree_df.reset_index(inplace=True, drop=True)
                self.model_depths_and_paths.append([])
                splits_dict = (
                    tree_df.groupby("split_feature")["threshold"].apply(list).to_dict()
                )
                self.model_splits.append(
                    {
                        get_lgbm_feat_id_from_feat_desc(k): v
                        for k, v in splits_dict.items()
                    }
                )
                lgbm_tree_to_dict(tree_df, self.model_depths_and_paths[i])

    def _set_new_leaves(self):
        for tree_ind, tree in enumerate(self.model_depths_and_paths):
            tree_leaves: Dict[int, Dict[str, Any]] = dict()
            for node in tree:
                if all(
                    ancestor not in tree_leaves for ancestor in node["ancestors"]
                ) and (
                    node["depth"] == self.max_depth
                    or node.get("child_min_node_samples", node["node_samples"])
                    < self.min_samples
                    or node["leaf"]
                ):
                    tree_leaves[node["node_id"]] = {
                        "feats": node["node_path_features"]
                        or list(self.model_splits[tree_ind].keys()),
                        "priors": node["node_priors"],
                    }
            self.new_leaves.append(tree_leaves)

    def _call_leaves(
        self,
        X,
        y: np.ndarray = None,
        method: str = "class_posteriors",
        per_tree: bool = False,
    ):
        if method in [
            "class_posteriors",
            "class_log_posteriors",
            "class_uncertainties",
            "priors",
            "aleatoric_uncertainty",
        ]:
            out = np.zeros((X.shape[0], self.model.n_classes_, self.model.n_estimators))
        elif method in ["global_likelihood", "global_uncertainty"]:
            out = np.zeros((X.shape[0], 1, self.model.n_estimators))
        elif method == "fit":
            out = None
            assert y is not None, (
                "The vector of labels 'y' must be specified when calling leaves with "
                "'fit'."
            )
        else:
            raise NotImplementedError(f"Method '{method}' not implemented.")

        indicator, n_nodes_ptr = self.model.decision_path(X)
        indicator = indicator.tocsc()
        Parallel(n_jobs=-2, require="sharedmem")(
            delayed(_call_tree_leaves)(
                self,
                X,
                y,
                method,
                tree_ind,
                tree_leaves,
                n_nodes_ptr,
                indicator,
                out,
            )
            for tree_ind, tree_leaves in enumerate(self.new_leaves)
        )

        if method != "fit":
            if not per_tree:
                if method == "priors" and isinstance(self.model, LightGBM):
                    return 1 / (1 + np.exp(-np.sum(out, axis=2)))
                else:
                    return np.mean(out, axis=2)
            else:
                if method == "priors" and isinstance(self.model, LightGBM):
                    return 1 / (1 + np.exp(-out))
                else:
                    return out

    def predict_proba(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="priors", per_tree=per_tree)

    def predict(self, X, per_tree: bool = False):
        return self.predict_proba(X, per_tree=per_tree)[:, 1] > 0.5


class UncertainTrees(TreePruner, ClassifierMixin):
    def __init__(
        self,
        tree_based_model: Union[RandomForestClassifier, LightGBM],
        max_depth: int = 6,
        min_samples: int = 10,
        cov_estimator: str = "shrunk",
        only_path_features: str = "yes",
        verbose: bool = False,
    ):
        super().__init__(tree_based_model, max_depth, min_samples, verbose)
        self.cov_estimator = cov_estimator
        self.only_path_features = only_path_features

    def fit(self, X, y, force_fit: bool = False):
        super().fit(X, y, force_fit=force_fit)
        self._fit_uncertainty_leaves(X, y)
        if self.verbose:
            print(f"Fitting step done!")

        return self

    @timeit
    @print_input_size
    def _fit_uncertainty_leaves(self, X, y):
        self._call_leaves(X, y, method="fit")

    @timeit
    @print_input_size
    def class_posteriors(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="class_posteriors", per_tree=per_tree)

    @timeit
    @print_input_size
    def class_log_posteriors(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="class_log_posteriors", per_tree=per_tree)

    @timeit
    @print_input_size
    def class_uncertainties(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="class_uncertainties", per_tree=per_tree)

    @timeit
    @print_input_size
    def global_likelihood(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="global_likelihood", per_tree=per_tree)

    @timeit
    @print_input_size
    def global_uncertainty(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="global_uncertainty", per_tree=per_tree)

    @timeit
    @print_input_size
    def decision_function(self, X, per_tree: bool = False):
        class_posteriors = self.class_posteriors(X, per_tree=per_tree)
        if per_tree:
            return class_posteriors[:, 1, :] - class_posteriors[:, 0, :]
        else:
            return class_posteriors[:, 1] - class_posteriors[:, 0]

    @timeit
    @print_input_size
    def predict_proba(self, X, per_tree: bool = False):
        class_posteriors = self.class_posteriors(X, per_tree=per_tree)
        proba = np.exp(class_posteriors[:, 1]) / (
            np.exp(class_posteriors[:, 0]) + np.exp(class_posteriors[:, 1])
        )
        return np.hstack([1 - proba.reshape(-1, 1), proba.reshape(-1, 1)])

    @timeit
    @print_input_size
    def predict(self, X, per_tree: bool = False):
        return self.predict_proba(X, per_tree=per_tree)[:, 1] >= 0.5

    @timeit
    @print_input_size
    def priors(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="priors", per_tree=per_tree)

    @timeit
    @print_input_size
    def aleatoric_uncertainty(self, X, per_tree: bool = False):
        return self._call_leaves(X, method="aleatoric_uncertainty", per_tree=per_tree)

    @timeit
    @print_input_size
    def total_uncertainty(self, X, per_tree: bool = False):
        probas = self.priors(X, per_tree=per_tree)
        return binary_entropy(probas)

    @timeit
    @print_input_size
    def epistemic_uncertainty(self, X, per_tree: bool = False):
        total_uncertainty = self.total_uncertainty(X, per_tree=per_tree)
        return total_uncertainty - self.aleatoric_uncertainty(X, per_tree=per_tree)
