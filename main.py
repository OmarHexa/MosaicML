import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.auto_classifier import AutoClassifier

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Load and split data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train auto-classifier
    auto_clf = AutoClassifier(cfg)
    auto_clf.fit(X_train, y_train)
    
    # Evaluate
    score = auto_clf.evaluate(X_test, y_test)
    print(f"Final score: {score:.4f}")

if __name__ == "__main__":
    main()