CoolGraph is an easy-to-use Python library using Graph Neural Networks for node classification. The CoolGraph contains several architectures that will help you train a network using two lines of code.

Thus, the parameters for training have already been selected and collected in configs, but you can change them as you wish.

Also, if for some reason the selected parameters do not ok for you, it is possible to use the search for hyperparameters with Optuna.

Moreover, your experiments can be saved in Mlflow and fully tracked.

All you need is graph-structured data.

# Main advantages of CoolGraph:
 - quick start in one line of code
 - good quality of base models, comparable to state of the art
 - the best model architecture automatic search via optuna
 - heterogeneous graph support
 - multitarget support
 - both node and edge features support
 - categorical and graph feaures usage
 - tracking experiments with mlflow

 
# Benchmarks

Dataset                   | accuracy | ROC-AUC | SOTA accuracy | SOTA ROC-AUC
--------------------------|----------|---------|---------------|-------------
**AntiFraud Amazon**      | 0.9828   | 0.9617  | -             | 0.9750
**AntiFraud YelpChi**     | 0.9050   | 0.9176  | -             | 0.9498
**Multitarget 10k**       | 0.8651   | 0.7415  | -             | -
**Multitarget 50k**       | 0.8637   | 0.7973  | -             | -
**Penn94**                | 0.7829   | 0.8714  | 0.8609        | -
**Genius**                | 0.8405   | 0.9016  | 0.9145        | -
**S_FFSD**                | 0.8961   | 0.8932  | -             | 0.8461
**OgbnProteins**          | 0.8967   | 0.8058  | -             | 0.8942


# Installation
## Install with creating conda env

`conda deactivate` <br>
`conda create -n cool_graph_env python=3.8 cudatoolkit=11.3.1 pytorch=1.12.0=py3.8_cuda11.3_cudnn8.3.2_0 cxx-compiler=1.5.1 pyg=2.2.0=py38_torch_1.12.0_cu113 pyarrow=11.0.0 numpy=1.23.5 pandas=1.4.4 pip=22.3.1 py=1.11.0 mysqlclient=2.0.3 sqlite=3.38.2 psycopg2=2.8.6 optuna=2.10.1 -c nvidia -c pytorch -c conda-forge -c pyg`  <br>
`conda activate cool_graph_env`.   <br>
`pip install cool-graph`


## Install with creating conda env from yml file


`conda deactivate` <br>
`conda env create -f environment.yml`  <br>
`conda activate cool_graph_env`  <br>
`pip install cool-graph`


## Install CoolGraph in Google Colab 

example is coming soon


# Get started


## Basic usage


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
print(result['best_loss'])       
```
## Runner with Optuna
You can run HypeRunner for the best GNN architecture search
```python
runner = HypeRunner(data)
result = runner.optimize_run()

```
For more information look at Tutorials

# Tutorials

1. [Easy start](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/Easy_start.ipynb)
2. [Main features](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/Usage_examples.ipynb)
3. [Making predictions and metric calculation](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/predict_proba_examples.ipynb)
4. [Working with categorical features](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/categorical_features_usage_examples.ipynb)
5. [Working with graph features](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/graph_features_usage_examples.ipynb)
5. [Creating your own data loaders](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/Indices_for_DataLoader.ipynb)
6. [Working with configs](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/How_to_work_with_configs.ipynb)
7. [Benchmarks](https://gitlab.services.mts.ru/bigdata_rnd/rnd_team/coolgraph/-/blob/dev/notebooks/benchmarks.ipynb)

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
├── datasets               <- CoolGraph datasets
├── logging                <- MLflow logging experiments
├── models                 <- CoolGraph models
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

## Authors
 
* [Diana Pavlikova](https://github.com/dapavlik)
* [Sergey Kuliev](https://github.com/kuliev-sd)
* [Igor Inozemtsev](https://github.com/inozemtsev)
* [Nikita Zelinskiy](https://github.com/nikita-ds)
* [Alexey Pristajko](https://github.com/qwertd105)
* [Vlad Kuznetsov](https://github.com/AnanasClassic)

## License 

`The MIT License (MIT)
Copyright 2023 MTS (Mobile Telesystems). All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`