import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from automl import AutoClassifier

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
    # Train auto-classifier
    automl = AutoClassifier(cfg, tracker, logger)
    automl.fit(X_train, y_train)

    # Train final model on full data
    automl.fit_best_model(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = automl.evaluate(X_test, y_test)
    logger.info(f"Test Metrics: {test_metrics}")
    
if __name__ == "__main__":
    main()