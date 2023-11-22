import pandas as pd
import pytest
import torch
from torch_geometric import datasets

from cool_graph.runners import HypeRunner, Runner


def test_run():
    try:
        data = datasets.Amazon(root="./data/Amazon", name="Computers").data
        runner = Runner(data)
        result = runner.run()
    except Exception as ex:
        raise pytest.fail(f"Exception in test_graphconv:\n {str(ex)}")


def test_with_edge_attr_run():
    """искусственно добавлем фичи ребер с ликом"""
    try:
        data = datasets.Amazon(root="./data/Amazon", name="Computers").data
        edges = pd.DataFrame(data.edge_index.numpy().T, columns=["index1", "index2"])
        y = pd.Series(data.y.numpy())
        edges["y1"] = edges.index1.map(y)
        edges["y2"] = edges.index2.map(y)
        edge_attr = torch.FloatTensor(
            (pd.get_dummies(edges.y1) + pd.get_dummies(edges.y2)).values
        )
        data.edge_attr = edge_attr

        runner = Runner(data, use_edge_attr=True)
        result = runner.run()
    except Exception as ex:
        raise pytest.fail(f"Exception in test_with_edge_attr_run:\n {str(ex)}")


def test_hyper_run():
    try:
        data = datasets.Amazon(root="./data/Amazon", name="Computers").data
        runner = HypeRunner(data)
        result = runner.optimize_run(n_trials=3)
    except Exception as ex:
        raise pytest.fail(f"Exception in test_graphconv:\n {str(ex)}")


def test_multi_target_run():
    try:
        data = datasets.Yelp(root="./data/Yelp").data
        data.y = data.y.long()
        runner = Runner(data)
        result = runner.run()
    except Exception as ex:
        raise pytest.fail(f"Exception in test_graphconv:\n {str(ex)}")


def test_multi_target_hyper_run():
    try:
        data = datasets.Yelp(root="./data/Yelp").data
        data.y = data.y.long()
        runner = HypeRunner(data)
        result = runner.optimize_run(n_trials=3)
    except Exception as ex:
        raise pytest.fail(f"Exception in test_graphconv:\n {str(ex)}")
