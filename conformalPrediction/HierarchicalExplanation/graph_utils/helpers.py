from collections import defaultdict
from random import shuffle

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from visualization.colormaps import get_default_cmaps


def draw_labels(pos, labels, font_size=8, y_offset=-.0009):
    for node, label in labels.items():
        if len(label) > 0:
            x, y = pos[node]
            label = label.replace(r"\\n", " ")
            if label.startswith("Bold"):
                plt.text(x, y + y_offset, label.replace("Bold", ""), fontsize=font_size, ha='center', va='center',
                         wrap=True, fontweight='bold')
            else:
                plt.text(x, y + y_offset, label, fontsize=font_size, ha='center', va='center',
                         wrap=True)


def get_remapped_name(names, rel_class_names = None):
    endings = [name.split("_")[-1] for name in names]
    counter = defaultdict(int)
    for end in endings:
        counter[end] += 1
    if rel_class_names is None: # Those class names that are shown in the readme on Github
        # This list ensures that their name is not omitted as a +x
        rel_class_names = ["Bronzed Cowbird", "Shiny Cowbird","Yellow Warbler", "Prothonotary Warbler" ]
    name_found = [name in rel_class_names for name in names]


    if any(name_found):
        name = names[name_found.index(True)]
    else:
        sorted_endings = sorted(names, key=lambda x: max([len(y) for y in x.split(" ")]))
        name = sorted_endings[0]
    broken_name = name.replace(' ', r'\\n')
    if len(names) == 1:
        return broken_name
    else:
        return broken_name + f", +{len(names) - 1}"



def get_smaller_v_space_pos(graph, root_idx, ranksep=0.02, second_halfer=0.0001):
    A = nx.nx_agraph.to_agraph(graph)
    A.layout(prog='dot', args=f"-Granksep={ranksep}")

    pos = {}
    max_y = 0

    for node in A:
        x, y = map(float, node.attr['pos'].split(','))
        pos[int(node)] = (x, y * second_halfer)  # Convert node name to string for NetworkX compatibility

        if y * second_halfer > max_y and int(node) != root_idx:
            max_y = y * second_halfer
    delta_y = pos[root_idx][1] - max_y
    pos[root_idx] = (pos[root_idx][0], pos[root_idx][1] - delta_y / 2)

    return pos

def get_colors_per_feature(feature, rel_features,  prior_order,
                           ):
    colors = {}
    tab20_colors = plt.cm.tab20.colors
    # First, take all the even-indexed colors (0, 2, 4, ...), which are the primary hues.
    # Then, take all the odd-indexed colors (1, 3, 5, ...), which are the darker shades.
    reordered_colors = list(tab20_colors[::2]) + list(tab20_colors[1::2])
    top_n_idx = torch.argsort(feature[rel_features], descending=True)[:20].cpu()
    for i, idx in enumerate(top_n_idx):
        index = ((i * 4) + (i * 4) // 20) % 20
        colors[rel_features[idx]] = reordered_colors[index]
    if prior_order is not None:
        for feat in prior_order:
            colors[feat] = prior_order[feat][-1] / 255
    return colors