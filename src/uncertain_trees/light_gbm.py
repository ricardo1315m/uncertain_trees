import lightgbm as lgb
from lightgbm.basic import Booster
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix, hstack as sparse_hstack
from typing import List, Dict, Any
from copy import deepcopy

from uncertain_trees.utils import nostdout

DEFAULT_PARAMS = {
    "num_leaves": 31,
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting": "gbdt",
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "boost_from_average": True,
}


def get_lgbm_feat_id_from_feat_desc(feat_col_desc: str):
    """
    LightGBM automatically names the unnamed features as ``Column_{i}``. This function
    retrieves the i's from the feature names.

    Parameters
    ----------
    feat_col_desc : str
        The name of the feature with the ``Column_`` prefix.

    Returns
    -------
    int
        The index of the feature.

    """
    return int(feat_col_desc.split("_")[1])


def lgbm_tree_to_dict(
    tree_df,
    tree_out: List[Dict[str, Any]] = None,
    node_idx=None,
    ancestors=None,
    num_path=None,
):
    num_path = num_path or []
    ancestors = ancestors or []

    if node_idx is None:
        node_idx = tree_df.iloc[0]["node_index"]

    node_i = tree_df.index[tree_df["node_index"] == node_idx][0]
    node = tree_df.iloc[node_i]

    tree_out.append(dict())
    tree_out[node_i]["node_id"] = node_i
    tree_out[node_i]["node_samples"] = node["count"]
    tree_out[node_i]["depth"] = node["node_depth"] - 1
    tree_out[node_i]["node_priors"] = np.array([-node["value"], node["value"]])
    tree_out[node_i]["node_path_features"] = sorted(set(num_path.copy()))
    tree_out[node_i]["ancestors"] = ancestors.copy()

    if node["left_child"] is not None:
        tree_out[node_i]["leaf"] = False
        feature = int(node["split_feature"].split("_")[1])
        tree_out[node_i]["feature"] = feature

        num_path.append(feature)
        ancestors.append(node_i)
        child_left_idx = node["left_child"]
        child_right_idx = node["right_child"]
        child_left_i = tree_df.index[tree_df["node_index"] == child_left_idx][0]
        child_right_i = tree_df.index[tree_df["node_index"] == child_right_idx][0]
        lgbm_tree_to_dict(
            tree_df,
            tree_out=tree_out,
            node_idx=child_left_idx,
            ancestors=ancestors.copy(),
            num_path=num_path.copy(),
        )
        lgbm_tree_to_dict(
            tree_df,
            tree_out=tree_out,
            node_idx=child_right_idx,
            ancestors=ancestors.copy(),
            num_path=num_path.copy(),
        )
        tree_out[node_i]["child_min_node_samples"] = min(
            tree_out[child_left_i]["node_samples"],
            tree_out[child_right_i]["node_samples"],
        )

    else:
        tree_out[node_i]["leaf"] = True
        tree_out[node_i]["feature"] = -1


class LightGBM(ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        early_stopping_rounds: int = 10,
        val_prct: float = 0.2,
        verbose: bool = False,
        **params,
    ):
        self.n_estimators_input = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.params = DEFAULT_PARAMS
        self.params.update(params)
        self.val_prct = val_prct
        self.max_depth = self.params.get("max_depth")
        self.min_data_in_leaf = self.params.get("min_data_in_leaf")
        self.verbose = verbose

    def __deepcopy__(self, memodict={}):
        obj = LightGBM()
        obj.__dict__.update(deepcopy(self.__dict__))
        model_str = self.lgbm_model_.model_to_string(num_iteration=-1)
        lgbm_model = Booster(model_str=model_str)
        lgbm_model.best_score = deepcopy(self.lgbm_model_.best_score)
        lgbm_model.best_iteration = deepcopy(self.lgbm_model_.best_iteration)
        lgbm_model.params = deepcopy(self.lgbm_model_.params)
        obj.lgbm_model_ = lgbm_model
        return obj

    @nostdout
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.val_prct)
        train_data = lgb.Dataset(X_tr, label=y_tr)
        validation_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        self.lgbm_model_ = lgb.train(
            self.params,
            train_data,
            self.n_estimators_input,
            valid_sets=[validation_data],
            early_stopping_rounds=self.early_stopping_rounds,
        )
        return self

    @property
    def n_estimators(self):
        if hasattr(self, "lgbm_model_"):
            return self.lgbm_model_.best_iteration
        else:
            return self.n_estimators_input

    @staticmethod
    def _tree_decision_path(tree_df, X):
        node_indices = dict(zip(tree_df["node_index"].tolist(), tree_df.index.tolist()))
        data_indices = np.zeros((X.shape[0], tree_df.shape[0]), dtype=bool)
        data_current_node = np.full(X.shape[0], 0)
        for node_id, node in enumerate(tree_df.itertuples()):
            data_idx_in_current_node = data_current_node == node_id
            data_indices[:, node_id] = data_idx_in_current_node
            if node.left_child is not None:
                feature = get_lgbm_feat_id_from_feat_desc(node.split_feature)
                thres = node.threshold
                left_id = node_indices[node.left_child]
                right_id = node_indices[node.right_child]
                data_idx_left = X[:, feature] <= thres
                data_current_node[data_idx_in_current_node & data_idx_left] = left_id
                data_current_node[data_idx_in_current_node & ~data_idx_left] = right_id

        return csc_matrix(data_indices)

    def decision_path(self, X):
        model_df = self.lgbm_model_.trees_to_dataframe()
        indicators = []
        for i, tree_df in model_df.groupby("tree_index"):
            tree_df.reset_index(drop=True, inplace=True)
            indicators.append(self._tree_decision_path(tree_df, X))

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsc(), n_nodes_ptr

    def predict_proba(self, X):
        probas = self.lgbm_model_.predict(X)
        return np.hstack([1 - probas.reshape(-1, 1), probas.reshape(-1, 1)])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
