from kedro.pipeline import Pipeline, node, pipeline
from kedro.runner import SequentialRunner

from .constants import SYMBOL_DEFAULT
from .nodes import (get_symbol, train_model, evaluate_model, plot_metric, train_test_split)

# Create a pipeline to orchestrate the steps
def create_pipeline(**kwargs) -> Pipeline:
    symbol = kwargs.get('symbol', SYMBOL_DEFAULT)
    
    pipeline_instance = pipeline([
        node(
            get_symbol,
            inputs=f'{symbol.lower()}_crypto_features_data',
            outputs='symbol',
            name='Get-Current-Symbol',
        ),
        node(
            train_test_split,
            inputs=f'{symbol.lower()}_crypto_features_data',
            outputs=['train_features', 'test_features'],
            name='Train-and-Test-Data-Split',
        ),
        node(
            train_model,
            inputs='train_features',
            outputs='model',
            name='Model-Training',
        ),
        node(
            evaluate_model,
            inputs=['model', 'test_features'],
            outputs=['y', 'y_pred', 'mse'],
            name='Model-Evaluation',
        ),
        node(
            plot_metric,
            inputs=['symbol', 'y', 'y_pred', 'mse'],
            outputs=None,
            name='Display-Model-Evaluation-Metrics',
        ),
    ])

    print(pipeline_instance.describe())
    
    return pipeline_instance

# COMMENT: ChatGPT guessed that piepline could be 'run' directly. In fact Pipeline doesn't have a run attribute.
# One must create an executor and run the pipeline through that!

# Execute the pipeline
def run_pipeline(symbol, catalog):
    runner = SequentialRunner()
    pipeline_instance = create_pipeline(**{'symbol': symbol})
    runner.run(pipeline_instance, catalog)
