import hydra
from omegaconf import DictConfig
from sklearn.datasets import load_iris
from auto_classifier import AutoClassifier

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    data = load_iris()
    clf = AutoClassifier(cfg)
    clf.fit(data.data, data.target)

if __name__ == "__main__":
    main()
