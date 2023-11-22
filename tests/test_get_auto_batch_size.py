import pytest

from cool_graph.data.batch import get_auto_batch_size


def test_get_auto_batch_size_graphconv():
    try:
        get_auto_batch_size(
            groups_num_features=[64, 32],
            conv_type="GraphConv",
            conv1_aggrs={"mean": 64, "max": 32, "add": 32},
            conv2_aggrs={"mean": 32, "max": 12, "add": 12},
            conv3_aggrs=None,
            n_hops=2,
            lin_prep_size_common=150,
            lin_prep_sizes=[75],
            edge_attr_repr_sizes=None,
            num_edge_features=None,
            device="cuda:0",
            num_neighbors=[25, 25],
        )

        get_auto_batch_size(
            groups_num_features=[64, 32],
            conv_type="GraphConv",
            conv1_aggrs={"mean": 64, "max": 32, "add": 32},
            conv2_aggrs={"mean": 32, "max": 12, "add": 12},
            conv3_aggrs={"mean": 16, "max": 8, "add": 8},
            n_hops=3,
            lin_prep_size_common=150,
            lin_prep_sizes=[75],
            edge_attr_repr_sizes=None,
            num_edge_features=None,
            device="cuda:0",
            num_neighbors=[25, 15, 10],
        )

    except Exception as ex:
        raise pytest.fail(
            f"Exception in get_auto_batch_size for GraphConv:\n {str(ex)}"
        )


def test_get_auto_batch_size_nnconv():
    try:
        get_auto_batch_size(
            groups_num_features=[64, 32],
            conv_type="NNConv",
            conv1_aggrs={"mean": 24, "max": 12, "add": 12},
            conv2_aggrs=None,
            conv3_aggrs=None,
            n_hops=1,
            lin_prep_size_common=150,
            lin_prep_sizes=[],
            edge_attr_repr_sizes=[60, 9, 2],
            num_edge_features=44,
            device="cuda:0",
            num_neighbors=[40],
        )

        get_auto_batch_size(
            groups_num_features=[64, 32],
            conv_type="NNConv",
            conv1_aggrs={"mean": 24, "max": 12, "add": 12},
            conv2_aggrs={"mean": 20, "max": 8, "add": 8},
            conv3_aggrs=None,
            n_hops=2,
            lin_prep_size_common=150,
            lin_prep_sizes=[100, 75],
            edge_attr_repr_sizes=[9, 2],
            num_edge_features=44,
            device="cuda:0",
            num_neighbors=[25, 15],
        )

        get_auto_batch_size(
            groups_num_features=[64, 32],
            conv_type="NNConv",
            conv1_aggrs={"mean": 24, "max": 12, "add": 12},
            conv2_aggrs={"mean": 24, "max": 8, "add": 8},
            conv3_aggrs={"mean": 16, "max": 4, "add": 4},
            n_hops=3,
            lin_prep_size_common=150,
            lin_prep_sizes=[100, 75],
            edge_attr_repr_sizes=[9, 1],
            num_edge_features=20,
            device="cuda:0",
            num_neighbors=[20, 15, 10],
        )

    except Exception as ex:
        raise pytest.fail(f"Exception in get_auto_batch_size for NNConv:\n {str(ex)}")
