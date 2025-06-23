# MosaicML
A highly configurable automl framework. Most of the available automl frameworks are not desinged to be configurable and act as a blackbox. This framework is designed to be configurable and extensible. Experiement and compare with different models, hyperparameters, data, and more.

## Usage

Define your models and their hyperparameters search space. Configure the metrics to be reported and hyperparameter optimization algorithm. Depending on the task call classifier or regressor. The model experiemnt tracker can be connected with mlflow server to visulaize the metrics and model performance. The best model can be tagged and deployed using mlflow server.
