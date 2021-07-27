from sklearn.tree import DecisionTreeClassifier, _tree
from typing import List, Dict, Any


def tree_to_dict(
    tree: DecisionTreeClassifier,
    tree_out: List[Dict[str, Any]] = None,
    splits_out: Dict[int, List[float]] = None,
    ancestors=None,
    num_path=None,
    node_id=0,
    depth=0,
):
    num_path = num_path or []
    ancestors = ancestors or []

    tree_out.append(dict())
    d = dict()
    d["node_id"] = node_id
    d["depth"] = depth
    d["node_samples"] = tree.tree_.value[node_id][0].sum()
    d["node_priors"] = tree.tree_.value[node_id][0] / d["node_samples"]
    d["node_path_features"] = sorted(set(num_path.copy()))
    d["ancestors"] = ancestors.copy()
    depth += 1
    if tree.tree_.children_right[node_id] != _tree.TREE_LEAF:
        d["leaf"] = False
        d["feature"] = tree.tree_.feature[node_id]
        if d["feature"] not in splits_out:
            splits_out[d["feature"]] = [tree.tree_.threshold[node_id]]
        else:
            splits_out[d["feature"]].append(tree.tree_.threshold[node_id])

        num_path.append(tree.tree_.feature[node_id])
        ancestors.append(node_id)
        child_left_node_id = tree.tree_.children_left[node_id]
        child_rigth_node_id = tree.tree_.children_right[node_id]
        tree_to_dict(
            tree,
            tree_out=tree_out,
            splits_out=splits_out,
            node_id=child_left_node_id,
            depth=depth,
            ancestors=ancestors.copy(),
            num_path=num_path.copy(),
        )
        tree_to_dict(
            tree,
            tree_out=tree_out,
            splits_out=splits_out,
            node_id=child_rigth_node_id,
            depth=depth,
            ancestors=ancestors.copy(),
            num_path=num_path.copy(),
        )
        d["child_min_node_samples"] = min(
            tree_out[child_left_node_id]["node_samples"],
            tree_out[child_rigth_node_id]["node_samples"],
        )

    else:
        d["leaf"] = True
        d["feature"] = -1

    tree_out[node_id] = d
