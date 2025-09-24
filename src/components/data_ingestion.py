import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import ConfigManager
from src.exception import CustomException
from src.logger import logger


class DataIngestion:
    def __init__(self, config):
        """
        Initialize with config dictionary from ConfigManager.
        """
        self.config = config

        self.root_dir = self.config["root_dir"]
        self.source_csv = self.config["source_csv"]
        self.image_dir = self.config["image_dir"]
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)

        # Artifact paths
        self.artifacts_dir = os.path.join("artifacts", "data_ingestion")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.train_csv = os.path.join(self.artifacts_dir, "train.csv")
        self.test_csv = os.path.join(self.artifacts_dir, "test.csv")

    def initiate_data_ingestion(self):
        """
        Reads dataset CSV, splits into train/test, and saves to artifacts.
        """
        try:
            logger.info("Starting data ingestion...")

            # Read dataset
            df = pd.read_csv(self.source_csv)
            logger.info(f"Dataset loaded with shape: {df.shape}")

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df["label"] if "label" in df.columns else None
            )

            # Save artifacts
            train_set.to_csv(self.train_csv, index=False)
            test_set.to_csv(self.test_csv, index=False)
            logger.info(f"Train/Test CSVs saved at: {self.artifacts_dir}")

            return self.train_csv, self.test_csv

        except Exception as e:
            raise CustomException(e)
