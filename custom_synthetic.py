import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class CustomNode:
    def __init__(self, inputs, name, f) -> None:

        self.inputs = inputs
        self.name = name
        self.f = f


class CustomGraph:
    def __init__(self, nodes) -> None:

        self.nodes = {node.name: node for node in nodes}
        self.edges = [
            [(i, n.name) for i in n.inputs]
            for n in self.nodes.values()
            if len(n.inputs) > 0
        ]
        self.edges = [item for sublist in self.edges for item in sublist]
        self.nx_graph = nx.DiGraph(self.edges)
        self.sample_df = pd.DataFrame()

    def plot_graph(self):
        nx.draw(self.nx_graph, with_labels=True, font_weight="bold")

    def sample_node(self, node, n):

        if self.node_sampled[node.name]:
            pass

        for parent in node.inputs:
            self.sample_node(self.nodes[parent], n)

        self.sample_df[node.name] = node.f(n, self.sample_df[node.inputs])
        self.node_sampled[node.name] = True

    def sample_from_graph(self, n, sample_df=None):

        if sample_df is None:
            self.node_sampled = {name: False for name in self.nodes.keys()}
            self.sample_df = pd.DataFrame()
        else:
            self.node_sampled = {
                name: name in sample_df.columns for name in self.nodes.keys()
            }
            self.sample_df = sample_df

        for node in self.nodes.values():
            self.sample_node(node, n)

        return self.sample_df


def custom_graph_example():

    nodes = [
        # Causal Path 1
        CustomNode([], "x1", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode([], "x2", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode(["x1", "x2"], "x3", lambda n, x: x["x1"] * x["x2"]),
        CustomNode(["x1", "x2"], "x4", lambda n, x: x["x1"] * x["x2"]),
        CustomNode(["x3", "x4"], "x5", lambda n, x: x["x3"] * x["x4"]),
        # Causal Path T
        CustomNode([], "x31", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode([], "x32", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode(["x31", "x32"], "tau", lambda n, x: x["x31"] + x["x32"]),
        CustomNode([], "T", lambda n, x: np.random.binomial(1, 0.3, size=n)),
        CustomNode(["T", "tau"], "xT", lambda n, x: x["T"] * x["tau"]),
        # Causal Path 2 - child of Y (No child anymore, due to backdoor)
        CustomNode([], "x21", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode([], "x22", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode(["x21", "x22"], "x23", lambda n, x: x["x21"] * x["x22"]),
        CustomNode(["x21", "x22"], "x24", lambda n, x: x["x21"] * x["x22"]),
        CustomNode(["x23", "x24"], "x25", lambda n, x: x["x23"] * x["x24"]),
        # Y
        CustomNode(
            ["x5", "xT", "x25"],
            "Y",
            lambda n, x: x["x5"] + x["xT"] + x["x25"] + np.random.normal(0, 1, n),
        ),
    ]

    graph = CustomGraph(nodes)
    graph.plot_graph()
    df = graph.sample_from_graph(1000)

    return df


def custom_graph_example2():

    nodes = [
        # Causal Path T
        CustomNode([], "x31", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode([], "x32", lambda n, x: np.random.normal(0, 1, n)),
        CustomNode(["x31", "x32"], "tau", lambda n, x: x["x31"] + x["x32"]),
        CustomNode([], "T", lambda n, x: np.random.binomial(1, 0.3, size=n)),
        CustomNode(["T", "tau"], "xT", lambda n, x: x["T"] * x["tau"]),
        # Y
        CustomNode(
            ["x32", "xT"],
            "Y",
            lambda n, x: x["x32"] ** 2 + x["xT"] + np.random.normal(0, 1, n),
        ),
    ]

    graph = CustomGraph(nodes)
    graph.plot_graph()
    df = graph.sample_from_graph(1000)

    return df
