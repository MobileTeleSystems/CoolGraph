from setuptools import setup, find_packages
import os
    
ENV = os.environ.get('COOL_GRAPH_ENV', 'PUBLIC')

NAME = 'cool-graph'
DESCRIPTION = 'The library to make GNN easy to touch'
LONG_DESCRIPTION = 'CoolGraph is an easy-to-use Python library using Graph Neural Networks for node classification.'
LICENSE = 'MIT'
VERSION = '0.0.1'

DEPENDENCIES = [
        'mlflow>=2.1.1',
        'numpy>=1.19.5',
        'omegaconf==2.3.0',
        'pandas>=1.2.4',
        'pyarrow>=6.0.1',
        'tqdm>=4.64.0',
        'urllib3>=1.26.9',
        'hydra-core>=1.3.0',
        'protobuf==3.20.0',
        'loguru==0.6.0' 
]


setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    version=VERSION,
    packages=find_packages(), 
    install_requires=DEPENDENCIES,
    package_data={
        "cool_graph.config": ["*.yaml"],
        "cool_graph.config.data": ["*.yaml"],
        "cool_graph.config.logging": ["*.yaml"],
        "cool_graph.config.metrics": ["*.yaml"],
        "cool_graph.config.model_params": ["*.yaml"],
        "cool_graph.config.training": ["*.yaml"],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'coolrun = cool_graph.cli.run:main', 
            'get_config = cool_graph.cli.config:main',
        ]
    }
)