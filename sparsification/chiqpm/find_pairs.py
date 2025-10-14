import numpy as np
import torch
from gurobipy import GRB



def get_list_of_almost_same(linear_dense, per_class_avg):
    pairwise_diff = create_pairwise_diff(linear_dense)
    n_pairs_to_create = int(linear_dense.shape[0] * per_class_avg)
    return get_list_from_pairwise_diff(pairwise_diff, n_pairs_to_create)



def create_pairwise_diff(linear):
    if isinstance(linear, torch.Tensor):
        linear = np.array(linear.to("cpu").detach())
    pairwise_diff = np.zeros((linear.shape[0], linear.shape[0]))
    for i in range(linear.shape[0]):
        for j in range(linear.shape[0]):
            pairwise_diff[i, j] = np.linalg.norm(linear[i] - linear[j])
    return pairwise_diff / np.max(pairwise_diff)


def get_list_from_pairwise_diff(pairwise_diff, n_pairs_to_create):
    pairwise_diff[np.triu_indices_from(pairwise_diff)] = np.nan

    pairs_to_enforce = []
    sorted_entries = np.argsort(pairwise_diff, axis=None)
    for i in range(n_pairs_to_create):
        index = np.unravel_index(sorted_entries[i], pairwise_diff.shape)
        pairs_to_enforce.append(index)
        print("Adding pair", i, index, "with distance", pairwise_diff[index])
    return pairs_to_enforce


def find_pairs_of_almost_same(linear_weight, min=4):
    # Find all pairs that share 4 entries, and also those that share 4 entries only with each other
    total_pairs = []
    entry_counter = torch.zeros((linear_weight.shape[0]))
    for i in range(0, len(linear_weight)):
        for j in range(i + 1, len(linear_weight)):
            if torch.sum(linear_weight[i] * linear_weight[j]) >= min:
                total_pairs.append((i, j))
                entry_counter[i] += 1
                entry_counter[j] += 1
    exclusive_pairs = []
    for x1, x2 in total_pairs:
        if entry_counter[x1] == 1 and entry_counter[x2] == 1:
            exclusive_pairs.append((x1, x2))
    return total_pairs, exclusive_pairs



def idealize_shares(existing_edges, features, forced_almost_equals, min_to_keep, prev_constraints, m, initial_val,
                      keep_ratio=1):
    if keep_ratio != 1:
        frac_of_equals_gotten = keep_ratio
    else:
        frac_of_equals_gotten = min_to_keep / len(forced_almost_equals)
    if frac_of_equals_gotten == 1:
        return initial_val # Returns since exactly the desired number of pairs exists
    for prev_constraint in prev_constraints:
        m.remove(prev_constraint)
    equality_achieved = m.addMVar(len(forced_almost_equals), vtype=GRB.BINARY, name="equality_achieved")
    m.addConstr(equality_achieved.sum() >= len(forced_almost_equals) * frac_of_equals_gotten,
                "frac_of_equals_gotten")
    for constr_idx, (i, j) in enumerate(forced_almost_equals):
        this_sum = (initial_val[:, i] * initial_val[:, j]).sum()
        constr = m.addConstr(
            (existing_edges[:, i] * existing_edges[:, j]).sum() >= this_sum * equality_achieved[constr_idx],
            "forced_almost_equals_{}_{}".format(i, j))
    features.lb = features.X
    features.ub = features.X
    m.optimize()