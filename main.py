import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.metrics.base import ClassificationMetricsCalculator
from src.automl.classifier import AutoClassifier

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AutoML")
    
    # Load and split data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Experiment tracker
    tracker = hydra.utils.instantiate(cfg.tracking)
    # Metrics calculator
    metrics_calculator = ClassificationMetricsCalculator(cfg.metrics)
    # Train auto-classifier
    automl = AutoClassifier(cfg, tracker, metrics_calculator, logger)
    automl.fit(X_train, y_train)

    # Train final model on full data
    automl.train_final_model(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = automl.evaluate(X_test, y_test)
    logger.info(f"Test Metrics: {test_metrics}")
    
    # Generate report
    report = automl.generate_report()
    logger.info("\n" + report)
    
    # Save report
    with open("classification_report.txt", "w") as f:
        f.write(report)
    
    # Log report
    automl.experiment_tracker.log_artifact("classification_report.txt", "report")
if __name__ == "__main__":
    main()