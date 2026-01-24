from collections import defaultdict
import json
import math
import nltk
from nltk.corpus import cmudict
from nltk.corpus import brown
import random
import re


# START = "<s>"
# END = "</s>"
START = "<START>"
END = "<END>"

PHONEME_CLASSES = {
    "vowel": {
        "AA","AE","AH","AO","AW","AY",
        "EH","ER","EY",
        "IH","IY",
        "OW","OY",
        "UH","UW"
    },
    "stop": {"P","B","T","D","K","G"},
    "fricative": {"F","V","TH","DH","S","Z","SH","ZH","HH"},
    "nasal": {"M","N","NG"},
    "liquid": {"L","R"},
    "glide": {"W","Y"},
    "silence": {"<START>", "<END>"}
}

PHONEME_COLORS = {
    "vowel": "lightskyblue",
    "stop": "lightcoral",
    "fricative": "khaki",
    "nasal": "lightgreen",
    "liquid": "plum",
    "glide": "orange",
    "silence": "gray90",
    "unknown": "white"
}


class Node:
    __slots__ = ("id", "phoneme", "edges")

    def __init__(self, node_id, phoneme):
        self.id = node_id
        self.phoneme = phoneme
        self.edges = {}  # phoneme -> child Node


def strip_stress(phone):
    return re.sub(r"\d", "", phone)

def word_to_phones(word, cmu_dict):
    word = word.lower()

    # Use first pronunciation
    phones = cmu_dict[word][0]
    return [strip_stress(p) for p in phones]
    # if word not in cmu_dict:
        # raise ValueError(f"Word not in CMUdict: {word}")

def sentence_to_phonemes(sentence, cmu_dict, raise_missing_error=False):
    phones = [START]
    for word in sentence.split():
        if word in cmu_dict:
            phones.extend(word_to_phones(word, cmu_dict))
    phones.append(END)
    return phones

def build_phoneme_dag(phoneme_sentences):
    nodes = []
    node_lookup = {}

    def get_node(prefix):
        if prefix not in node_lookup:
            phoneme = prefix[-1]
            node = Node(len(nodes), phoneme)
            nodes.append(node)
            node_lookup[prefix] = node
        return node_lookup[prefix]

    root = get_node(("<START>",))

    for phones in phoneme_sentences:
        current_prefix = ("<START>",)
        current_node = root

        for phone in phones[1:]:
            next_prefix = current_prefix + (phone,)
            next_node = get_node(next_prefix)

            if phone not in current_node.edges:
                current_node.edges[phone] = next_node

            current_node = next_node
            current_prefix = next_prefix

    return root, nodes

def compute_depths(root):
    depths = {}

    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.id in depths:
            continue
        depths[node.id] = depth
        for child in node.edges.values():
            stack.append((child, depth + 1))

    return depths

def phoneme_class(phone):
    for cls, phones in PHONEME_CLASSES.items():
        if phone in phones:
            return cls
    return "unknown"

def export_dag_to_json(nodes, positions, filename="phone_dag.json"):
    incoming_connections = defaultdict(list)
    for node in nodes:
        for child in node.edges.values():
            incoming_connections[child.id].append(node.id)

    nodes_data = []
    for node in nodes:
        curr_positions = positions[node.id]
        curr_connections = incoming_connections[node.id]

        relative_coordinates = {}

        for connection_id in curr_connections:
            connection_positions = positions[connection_id]
            rel_x = float(curr_positions[0]) - float(connection_positions[0])
            rel_y = float(curr_positions[1]) - float(connection_positions[1])

            relative_coordinates[connection_id] = {
                "rel_x": rel_x,
                "rel_y": rel_y
            }

        nodes_data.append({
            "id": node.id,
            "phoneme": node.phoneme,
            "connects_from": incoming_connections[node.id],
            "relative_coordinates": relative_coordinates
        })

    with open("phone_dag_with_connections.json", "w") as f:
        json.dump(nodes_data, f, indent=2)

    print(f"Exported DAG to {filename}")

    json_nodes = []
    json_edges = []

    for node in nodes:
        phone = node.phoneme
        cls = phoneme_class(phone)

        json_nodes.append({
            "id": node.id,
            "phoneme": phone,
            "class": cls,
            "color": PHONEME_COLORS.get(cls, "white"),
            "shape": "box" if phone in ("<START>", "<END>") else "circle",
            "x": float(positions[node.id][0]),
            "y": float(positions[node.id][1])
        })

    for node in nodes:
        for child in node.edges.values():
            json_edges.append({
                "source": node.id,
                "target": child.id
            })

    graph = {
        "nodes": json_nodes,
        "edges": json_edges
    }

    with open(filename, "w") as f:
        json.dump(graph, f, indent=2)

def organic_layout(
    nodes,
    depths,
    x_spacing=200,
    base_y_spacing=35,
    attraction=0.25,
    jitter_y=30,
    jitter_x=20,
    min_spacing=12,
    repel_strength=0.5,
    iterations=4,
    seed=42,
):
    """
    Input:
        nodes  : list[Node]
        depths : dict[node_id -> int]

    Returns:
        positions : dict[node_id -> (x, y)]
    """

    random.seed(seed)

    # --------------------------------------------------
    # 1. Index nodes by ID
    # --------------------------------------------------
    node_by_id = {n.id: n for n in nodes}

    # --------------------------------------------------
    # 2. Group nodes by depth
    # --------------------------------------------------
    layers = defaultdict(list)
    for n in nodes:
        if n.id not in depths:
            raise ValueError(f"Missing depth for node {n.id}")
        layers[depths[n.id]].append(n.id)

    # --------------------------------------------------
    # 3. Initial clean layered y layout
    # --------------------------------------------------
    y_map = {}
    for depth, node_ids in layers.items():
        node_ids = sorted(node_ids)
        for i, nid in enumerate(node_ids):
            y_map[nid] = i * base_y_spacing

    # --------------------------------------------------
    # 4. Extract edge list
    # --------------------------------------------------
    edges = []
    for n in nodes:
        for child in n.edges.values():
            edges.append((n.id, child.id))

    # --------------------------------------------------
    # 5. Iterative organic relaxation
    # --------------------------------------------------
    for _ in range(iterations):

        # ---- 5a. Parent → child attraction ----
        for u, v in edges:
            y_map[v] = (1 - attraction) * y_map[v] + attraction * y_map[u]

        # ---- 5b. Repulsion ONLY within same depth ----
        for depth, node_ids in layers.items():
            node_ids = sorted(node_ids, key=lambda n: y_map[n])
            for i in range(len(node_ids) - 1):
                u = node_ids[i]
                v = node_ids[i + 1]
                dy = y_map[v] - y_map[u]
                if dy < min_spacing:
                    shift = (min_spacing - dy) * repel_strength
                    y_map[u] -= shift
                    y_map[v] += shift

        # ---- 5c. Gentle sinusoidal drift by depth ----
        for nid in y_map:
            d = depths[nid]
            y_map[nid] += math.sin(d * 0.6) * 0.3

    # --------------------------------------------------
    # 6. Final layered noise
    # --------------------------------------------------
    for nid in y_map:
        y_map[nid] += random.gauss(0, jitter_y)

    # --------------------------------------------------
    # 7. Final (x, y) positions
    # --------------------------------------------------
    positions = {}
    for n in nodes:
        d = depths[n.id]
        x = d * x_spacing + random.uniform(-jitter_x, jitter_x)
        y = y_map[n.id]
        positions[n.id] = (x, y)

    return positions

def organic_layout_with_loops(
    nodes,
    depths,
    x_spacing=200,
    base_y_spacing=35,
    attraction=0.25,
    jitter_y=30,
    jitter_x=20,
    min_spacing=12,
    repel_strength=0.5,
    iterations=4,
    seed=42,
):
    random.seed(seed)

    # ---- group by depth ----
    layers = defaultdict(list)
    for n in nodes:
        layers[depths[n.id]].append(n.id)

    # ---- initial y ----
    y_map = {}
    for depth, ids in layers.items():
        for i, nid in enumerate(sorted(ids)):
            y_map[nid] = i * base_y_spacing

    # ---- edges ----
    edges = []
    for n in nodes:
        for child in n.edges.values():
            edges.append((n.id, child.id))

    # ---- relaxation ----
    for _ in range(iterations):

        # forward-only attraction
        for u, v in edges:
            if depths[v] > depths[u]:
                y_map[v] = (1 - attraction) * y_map[v] + attraction * y_map[u]

        # same-depth repulsion
        for depth, ids in layers.items():
            ids = sorted(ids, key=lambda n: y_map[n])
            for i in range(len(ids) - 1):
                u, v = ids[i], ids[i + 1]
                dy = y_map[v] - y_map[u]
                if dy < min_spacing:
                    shift = (min_spacing - dy) * repel_strength
                    y_map[u] -= shift
                    y_map[v] += shift

        # gentle drift
        for nid in y_map:
            y_map[nid] += math.sin(depths[nid] * 0.6) * 0.3

    # ---- noise ----
    for nid in y_map:
        y_map[nid] += random.gauss(0, jitter_y)

    # ---- positions ----
    positions = {}
    for n in nodes:
        base_x = depths[n.id] * x_spacing
        x = base_x if n.id in ("<START>", "<END>") else base_x + random.uniform(-jitter_x, jitter_x)
        positions[n.id] = (x, y_map[n.id])

    return positions


def main():
    nltk.download('cmudict')
    nltk.download('brown')
    nltk.download('punkt_tab')
    # nltk.download('treebank')

    cmu = cmudict.dict()

    sentences_brown = brown.sents()

    full_sentences_brown = [" ".join(s) for s in sentences_brown]

    # n = 200  # Number of elements to sample (12271 nodes)
    n = 125 # 6941 nodes

    # Select n unique random elements
    full_sentences_brown_sample = random.sample(full_sentences_brown, n)

    # Keep only the first n_words words (if length is greater than n_words) in each sentence
    full_sentences_brown_sample_n = []

    # n_words = 400 # 12271 nodes
    n_words = 300 # 6941 nodes

    for sentence in full_sentences_brown_sample:
        if len(sentence) > n_words:
            full_sentences_brown_sample_n.append(sentence[0:n_words])
        else:
            full_sentences_brown_sample_n.append(sentence)

    phoneme_sentences_brown_sample_n = [
        sentence_to_phonemes(s, cmu_dict=cmu)
        for s in full_sentences_brown_sample_n
    ]


    root, nodes = build_phoneme_dag(phoneme_sentences_brown_sample_n)

    depths = compute_depths(root)

    positions_organic = organic_layout(
        nodes=nodes,
        depths=depths,
        x_spacing=200,
        base_y_spacing=35,
        attraction=0.25,
        # jitter_y=30,
        jitter_y=75,
        # jitter_x=20,
        jitter_x=75,
        min_spacing=12,
        repel_strength=0.5,
        iterations=4,
        seed=42,
    )


    positions_organic_with_loops = organic_layout_with_loops(
        nodes=nodes,
        depths=depths,
        x_spacing=200,
        base_y_spacing=35,
        attraction=0.25,
        # jitter_y=30,
        jitter_y=75,
        # jitter_x=20,
        jitter_x=75,
        min_spacing=12,
        repel_strength=0.5,
        iterations=4,
        seed=42,
    )

    export_dag_to_json(nodes, positions_organic_with_loops, filename="phone_dag_with_loops.json")
    export_dag_to_json(nodes, positions_organic, filename="phone_dag_no_loops.json")


if __name__ == '__main__':
    main()