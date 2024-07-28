from typing import List

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Visualizer:
    def __init__(self, stats: dict = None):
        if not(stats is None):
            if not (type(stats) is dict):
                raise TypeError("Expected a dict object with stats")
            self.stats = stats
        
        self.h_row_in_px = 200


    def plot_train_summary(self, stats: dict = None):
        """
        Args:
          stats (dict): train report from runner/hyperunner.
        """
        if not(stats is None):
            if not (type(stats) is dict):
                raise TypeError("Expected a dict object with stats")
            self.stats = stats
        assert(not(self.stats is None))
        
        fig = make_subplots(rows=1,
                            cols=2,
                            subplot_titles=("Loss", "Main metric"))

        fig.add_trace(
            go.Scatter(
                x=self.stats['train_loss']['epoch'],
                y=self.stats['train_loss']['total_loss'],
                name='loss',
                mode='lines+markers',
                legendgroup="train",
                legendgrouptitle_text="Train",
              ),
            row=1, col=1,
        )
        fig.update_xaxes(title_text="epoch", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=self.stats['train_metric']['epoch'],
                y=self.stats['train_metric']['main_metric'],
                name='metric',
                mode='lines+markers',
                legendgroup="train",
                legendgrouptitle_text="Train",
              ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats['test_metric']['epoch'],
                y=self.stats['test_metric']['main_metric'],
                name='metric',
                mode='lines+markers',
                legendgroup="validation",
                legendgrouptitle_text="Validation",
              ),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="epoch", row=1, col=2)

        fig.update_layout(
            height=400,
            width=800,
            title={
                'text' : 'Training progress',
                'x' : 0.5,
                'xanchor': 'center'
              }
        )

        fig.show()


    @staticmethod
    def add_trace_short(fig, stats, x_ind, y_ind, name, idx, row, col):
        fig.add_trace(
            go.Scatter(
                x=stats[x_ind],
                y=stats[y_ind],
                name=name,
                mode='lines+markers',
                legendgroup=str(idx),
                legendgrouptitle_text='scores'
            ),
            row=row, col=col,
        )

      

    def plot_metrics_results(self, stats: dict = None, category: str = 'both'):
        """
        Args:
          stats (dict): train report from runner/hyperunner,
          category (str): 'train', 'validation', or 'both', default = 'both'
        """
        if not(stats is None):
            if not (type(stats) is dict):
                raise TypeError("Expected a dict object with stats")
            self.stats = stats
        assert(not(self.stats is None))
        
        metrics_names = self.stats['test_metric'].keys()[:-4] # 4 = len(['calc_time', 'main_metric', 'tasks', 'epoch'])

        num_rows = len(metrics_names)
        num_cols = 1

        fig = make_subplots(rows=num_rows,
                            cols=num_cols,
                            subplot_titles=metrics_names)

        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                idx = (row - 1) * num_cols + col - 1

                if category == 'both' or category == 'train':
                    self.add_trace_short(fig, self.stats['train_metric'], 'epoch', metrics_names[idx], 'train', idx, row, col)

                if category == 'both' or category == 'validation':
                    self.add_trace_short(fig, self.stats['test_metric'], 'epoch', metrics_names[idx], 'validation', idx, row, col)

                fig.update_xaxes(title_text="epoch", row=row, col=col)

        fig.update_layout(
            height=(num_rows + 1) * self.h_row_in_px,
            width=800,
            title_text="Metrics results",
            legend_tracegroupgap = self.h_row_in_px * 0.9,
        )

        fig.show()

    def get_common_metrics(self):
        column_filter = {'calc_time','main_metric', 'tasks', 'epoch'}
        cols = []
        
        for key in self.stats['train_metric'].keys():
            if key in column_filter:
                continue
            cols.append(key)
        
        return cols


    @staticmethod
    def collect_metrics_for_tasks(fig, target, common_metrics, tasks, epochs, name):
        cur_metrics = [[] for i in range(len(common_metrics))]
        
        pos = dict()
        for i, metric_name in enumerate(common_metrics):
            pos[metric_name] = i

        for task in tasks:
            cur_target = task[target]
            for metric in cur_target.keys():
                cur_metrics[pos[metric]].append(cur_target[metric])
                
        for row in range(1, len(common_metrics) + 1, 1):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=cur_metrics[row - 1],
                    name=name,
                    mode='lines+markers',
                    legendgroup=common_metrics[row - 1],
                    legendgrouptitle_text='scores',
                ),
                row=row, col=1,
            )
            
            fig.update_xaxes(title_text="epoch", row=row, col=1)
    

    def plot_multitasks(self, stats: dict = None, targets: List[str] = None, category: str = 'both'):
        """
        Args:
          stats (dict): train report from runner/hyperunner.
          targets (list[str]): list of current targets.
          category (str): 'train', 'validation', or 'both', default = 'both'
        """
        if not(stats is None):
            if not (type(stats) is dict):
                raise TypeError("Expected a dict object with stats")
            self.stats = stats
        assert(not(self.stats is None))
        
        common_metrics = self.get_common_metrics()
        
        if targets is None:
            titles = ['Loss']
            titles.extend(common_metrics)
            
            num_rows = len(titles)
            num_cols = 1

            fig = make_subplots(rows=num_rows,
                                cols=num_cols,
                                subplot_titles=titles)

            fig.add_trace(
                go.Scatter(
                    x=self.stats['train_loss']['epoch'],
                    y=self.stats['train_loss']['total_loss'],
                    name='train',
                    mode='lines+markers',
                    legendgrouptitle_text='Loss',
                ),
                row=1, col=1,
            )

            scenario = []
            if category == 'train' or category == 'both':
                scenario.append('train')
            if category == 'validation' or category == 'both':
                scenario.append('test')

            for cur_row, metric in enumerate(common_metrics, start=2):
                for cur in scenario:
                    fig.add_trace(
                        go.Scatter(
                            x=self.stats[f'{cur}_metric']['epoch'],
                            y=self.stats[f'{cur}_metric'][metric],
                            name=cur,
                            mode='lines+markers',
                            legendgroup=metric,
                            legendgrouptitle_text='scores',
                        ),
                        row=cur_row, col=1,
                    )

                fig.update_xaxes(title_text="epoch", row=cur_row, col=1)

            fig.update_layout(
                height=(num_rows + 1) * self.h_row_in_px,
                width=800,
                title_text="Metrics results",
                legend_tracegroupgap = self.h_row_in_px * 0.9,
            )

            fig.show()

        else:
            train_epochs = self.stats['train_metric']['epoch']  # Ox
            val_epochs = self.stats['test_metric']['epoch']     # Ox

            for target in targets:
                fig = make_subplots(rows=len(common_metrics),
                                    cols=1,
                                    subplot_titles=common_metrics)

                if category == 'both' or category == 'train':
                    self.collect_metrics_for_tasks(fig, target, common_metrics, self.stats['train_metric']['tasks'], train_epochs, name='train')
                
                if category == 'both' or category == 'validation':
                    self.collect_metrics_for_tasks(fig, target, common_metrics, self.stats['test_metric']['tasks'], val_epochs, name='validation')
            
                fig.update_layout(
                    height=(len(common_metrics) + 1) * self.h_row_in_px,
                    width=800,
                    legend_tracegroupgap = self.h_row_in_px * 0.9,
                    title={
                        'text' : f'{target}',
                        'x' : 0.5,
                        'xanchor': 'center'
                    }
                )
            
                fig.show()
