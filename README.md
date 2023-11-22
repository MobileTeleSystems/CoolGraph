CoolGraph is an easy-to-use Python library using Graph Neural Networks for node classification

# Main advantages of CoolGraph:
 - quick start in one line of code
 - good quality of base models, comparable to state of the art
 - heterogeneous graph support
 - the best model architecture automatic search via optuna
 - tracking experiments with mlflow
 - user could define the targets count and target weights in the loss function
 - estimating batch size and neighbourhood sampling sizes for the first and second hop via graph metrics calculation

# Documentation

For more details, look at tutorials in folder Notebooks.

# Install with creating conda

`conda deactivate` <br>
`conda create -n cool_graph_env2_py38 python=3.8 cudatoolkit=11.3.1 pytorch=1.12.0=py3.8_cuda11.3_cudnn8.3.2_0 cxx-compiler=1.5.1 pyg=2.2.0=py38_torch_1.12.0_cu113 pyarrow=11.0.0 numpy=1.23.5 pandas=1.4.4 pip=22.3.1 py=1.11.0 mysqlclient=2.0.3 sqlite=3.38.2 psycopg2=2.8.6 optuna=2.10.1 -c nvidia -c pytorch -c conda-forge -c pyg`  <br>
`pip install cool-graph` 


# Install with creating conda from yml file


`conda deactivate` <br>
`conda env create -f environment.yml`  <br>
`conda activate cool_graph_env2_py38`  <br>
`pip install cool-graph`


## Install CoolGraph without creating conda

You can use CoolGraph in Google Colab without installing the conda, but make sure that the default colab environment matches the required versions for the library. 


# Usage

Look at page notebook in ./notebooks/CoolGraph_usage_examples.ipynb

or you can see the example with open fraud dataset from Yelp at ./notebooks/YelpChi_dataset_with_edge_attr.ipynb


# Benchmark

Coming soon

## Configs

In Coolgraph you can use default config file but also you can change it. See below how to copy config structure to your path, see discovery in configs and run.

## CoolGraph CLI

`coolrun --config <path/to/config>`

You can easily override config parameters: 

```coolrun --config ./cool_graph/config/full.yaml training.n_epochs=5```

To copy config structure use command:

    get_config --configs <path/to/config/where/you/need/it>`

## Jupyter notebook
### Runner without Optuna

Basic pipeline
```python
# Load Dataset
from torch_geometric import datasets
data = datasets.Amazon(root='./data/Amazon', name='Computers').data

# Train GNN model
from cool_graph.runners import Runner
runner = Runner(data)
result = runner.run()
```

You can override default parameters and/or read parameters from config file
```python
runner = Runner(data, 
                metrics=['accuracy', ...], 
                batch_size='auto', 
                train_size=0.7, 
                test_size=0.3, 
                overrides=['training.n_epochs=1', ...], 
                config_path=...)
result = runner.run()                
```
### Runner with Optuna
You can run HypeRunner for the best GNN architecture search
```python
runner = HypeRunner(data)
result = runner.optimize_run()
```
For more information look at examples

## Library Structure

The directory structure cool_graph:

```
├── config                       <- Config structure
│   ├── data                     <- Data configs (Datasets on disk)
│   ├── logging                  <- MLFlow configs
│   ├── metrics                  <- Metrics configs
│   ├── model_params             <- NN model parameters configs
│   ├── training                 <- Training  configs
│   ├── full.yaml                <- Main config for CLI 
│   ├── in_memory_data.yaml      <- Main config for notebook (GraphConv)
│   ├── in_memory_data2.yaml     <- Main config for notebook (NNConv)
│
├── cli                    <- Cli commands
├── data                   <- Data processing, data loaders, batch sizes
├── logging                <- MLflow logging experiments
├── models                 <- Cool graph models
├── parameter_search       <- Sampling model params for Optuna
├── train                  <- Trainin / eval / metrics code
├── runners.py             <- Run training (CLI + Notebook)

```

## Development

**add dev dependencies here (or switch to poetry)**

Install the package using one of the options described above adding `-e` flag to `pip install`.

Run `git checkout -b <new_branch_name>`.

Introduce changes to the branch.

Test your changes using `make test`.

Add tests if necessary (e.g. coverage fell below 80%), run `make test` again.

Ensure that codestyle is OK with `make verify_format`.

If not, run `make format`.

After that commit -> push -> PR.

## License 

`The MIT License (MIT)
Copyright 2023 MTS (Mobile Telesystems). All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`








