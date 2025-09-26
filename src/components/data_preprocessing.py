import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import json
from pathlib import Path

from src.config.configuration import ConfigManager
from src.exception import CustomException
from src.logger import logger
from src.utils.utils import create_directories, save_json


class data_preprocessing:
    """
    Comprehensive data preprocessing for chicken disease classification.
    Handles image loading, preprocessing, augmentation, and class imbalance.
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params


        # Directories and paths
        self.image_dir = self.config["image_dir"]
        self.artifacts_dir = Path("artifacts/data_preprocessing")
        create_directories(str[self.artifacts_dir])

        # Image parameters
        self.img_height = tuple(self.params["IMAGE_HEIGHT"])
        self.batch_size = self.params["BATCH_SIZE"]
        self.augmentation_params = self.params.get("AUGMENTATION",True)

        # Imbalance handling
        self.handle_imbalance = self.params.get("HANDLE_IMBALANCE", True)
        self.minority_class_threshold = self.params.get("MINORITY_CLASS_THRESHOLD", 0.25)
        self.minority_boost = self.params.get("MINORITY_BOOST", True)

        # Initializing label encoder
        self.label_encoder = LabelEncoder()

        # Statistics tracking
        self.preprocessing_stats = {
            "total_image_loaded": 0,
            "images_removed": 0,
            "augmented_image_created": 0,
            "preprocessing_errors": []
        }

    def create_data_generator(self,train_csv_path,test_csv_path):
        """
        Main method to create training and validation data generators.

        Args:
            train_csv_path (str): Path to training CSV file.
            test_csv_path (str): Path to testing CSV file.
        """

        try:
            logger.info("🚀 Starting data preprocessing pipeline...")
            logger.info("=" * 60)

            # Step 1: Load & validate datasets
            train_df, test_df = self._load_and_validate_data(train_csv_path, test_csv_path)

            # Step 2: Analyze class distribution
            class_analysis = self._analyze_class_distribution(train_df)

            # Step 3: Handle class imbalance
            if self.handle_imbalance and class_analysis['severe_imbalance']:
                train_df = self._handle_class_imbalance(train_df, class_analysis)
                
                # Re-analyze after balancing
                final_analysis = self._analyze_class_distribution(train_df) # Overwrite if balanced

            logger.info("📊 Final class distribution (after balancing if applied):")
            self._log_class_distribution(final_analysis) # Will always be defined

            # Step 4: Calculate class weights
            class_weights = self._calculate_class_weights(train_df)

            # Step 5: Create ImageDataGenerators
            train_generator = self._create_training_generator(train_df)
            validation_generator = self._create_validation_generator(test_df)

            # Step 6: Save preprocessing artifacts
            self._save_preprocessing_artifacts(train_df,class_analysis,class_weights)

            # prepare return info
            class_info = {
                'class_weights': class_weights,
                'num_classes': len(self.label_encoder.classes_),
                'classes_names': list(self.label_encoder.classes_),
                'class_mapping': dict(enumerate(self.label_encoder.classes_)),
                'class_analysis': class_analysis,
                'preprocessing_stats': self.preprocessing_stats
            }

            logger.info("✅ Data preprocessing completed successfully!")
            logger.info("=" * 60)\
            
            return train_generator, validation_generator, class_info
        
        except Exception as e:
            raise CustomException(e)
        
    def _load_and_validate_data(self, train_csv_path, test_csv_path):
        """
        Load datasets from CSV and validate image paths.
        Args:
            train_csv_path (str): Path to training CSV file.
            test_csv_path (str): Path to testing CSV file.
        Returns:
            train_df (pd.DataFrame): Validated training dataframe.
            test_df (pd.DataFrame): Validated testing dataframe.
        """

        try:
            logger.info("Step 1: Loading and validating datasets...")

            # Load datasets
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            logger.info(f"Initial training samples: {len(train_df)}, testing samples: {len(test_df)}")

            self.preprocessing_stats["total_image_loaded"] = len(train_df) + len(test_df)

            # fit label encoder on all data for consistency
            all_labels = pd.concat([train_df['label'], test_df['label']])
            self.label_encoder.fit(all_labels)

            # Encode labels
            train_df['label'] = self.label_encoder.transform(train_df['label'])
            test_df['label'] = self.label_encoder.transform(test_df['label'])

            logger.info(f"Classes found: {list(self.label_encoder.classes_)}")
            logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")

            # Create full image paths
            train_df['image_path'] = train_df['images'].apply(
                lambda x: str(Path(self.image_dir) / x))
            
            test_df['image_path'] = test_df['images'].apply(
                lambda x: str(Path(self.image_dir) / x))
            
            # Validate image paths
            self._validate_image_files(train_df,"training")
            self._validate_image_files(test_df,"testing")

            logger.info(f"Validated training samples: {len(train_df)}, testing samples: {len(test_df)}")
            return train_df, test_df
        
        except Exception as e:
            raise CustomException(e)
        
    def _validate_image_files(self, df, dataset_type):
        """
        Validate image file paths in the dataframe.
        Args:
            df (pd.DataFrame): Dataframe with image paths.
            dataset_type (str): 'training' or 'testing' for logging.
        """
        try:
            logger.info(f"  🔍 Validating {dataset_type} images...")
            valid_indices = []
            invalid_count = 0

            for idx, row in df.iterrows():
                image_path = row['image_path']
                # Check if file exists
                if not os.path.exists(image_path):
                    logger.warning(f"    ❌ Missing file: {image_path}")
                    invalid_count += 1
                    self.preprocessing_stats["preprocessing_errors"].append({
                        "image_path": image_path,
                        "error": "File not found"
                    })
                    continue
                # Try loading the image
                try:
                    # load image to verify it's readable
                    img = load_img(image_path, target_size=self.image_size)
                    img_array = img_to_array(img)  # Convert to array

                    # Validate image properties
                    if img_array.shape[2] != 3:  # Check for 3 channels
                        logger.warning(f"    ❌ Invalid image (not 3 channels): {image_path}")
                        invalid_count += 1
                        continue

                    # check the valid pixel values
                    if np.isnan(img_array).any() or np.isinf(img_array).any():
                        logger.warning(f"    ❌ Invalid image (NaN/Inf values): {image_path}")
                        invalid_count += 1
                        continue

                    valid_indices.append(idx)

                except Exception as img_error:
                    logger.warning(f"    ❌ Error loading image {image_path}: {img_error}")
                    invalid_count += 1
                    self.preprocessing_stats["preprocessing_errors"].append({
                        "image_path": image_path,
                        "error": str(img_error)
                    })

            # Filter dataframe to only valid images
            df_valid = df.loc[valid_indices].reset_index(drop=True)

            logger.info(f"  ✅ Valid {dataset_type} images: {len(df_valid):,}")
            if invalid_count > 0:
                logger.warning(f"  ⚠️  Removed {invalid_count} invalid images from {dataset_type} set")
                self.preprocessing_stats["images_removed"] += invalid_count
            
            return df_valid

        except Exception as e:
            raise CustomException(e)

    def _analyze_class_distribution(self, df):
        """ Analyze class distribution and identify imbalance issues
        """
        try:
            # Count samples per class
            class_counts = df['label'].value_counts.sort_index()
            total_samples = len(df)

            # Calculate imbalance metrics
            max_samples = class_counts.max()
            min_samples = class_counts.min()
            imbalance_ratio = max_samples/min_samples if min_samples > 0 else float('inf')

            # Identify minority class
            threshold = max_samples * self.minority_class_threshold
            minority_classes = class_counts[class_counts < threshold].index.tolist()

            # Determine Severity
            needs_balancing = imbalance_ratio > 2.0
            severe_imbalance = imbalance_ratio > 4.0

            class_analysis = {
                'class_counts' : class_counts.to_dict(),
                'total_samples' : total_samples,
                'max_samples' : max_samples,
                'min_samples' : min_samples,
                'imbalance_ratio' : imbalance_ratio,
                'minority_classes' : minority_classes,
                'needs_balancing' : needs_balancing,
                'severe_imbalance' : severe_imbalance
            }

            return class_analysis
        
        except Exception as e:
            CustomException(e)

    def _log_class_distribution(self, class_analysis):
        """ Log detailed class distribution analysis
        """
        try:
            logger.info(" ------ Class Distribution Analysis-------")

            total = class_analysis['total_samples']  
            for class_name, count in class_analysis['class_counts'].items():
                percentage = (count / total) * 100
                status = "MINORITY" if class_name in class_analysis['minority_class'] else "NOT A MINORITY"
                logger.info(f"   {status} {class_name} : {count : ,} samples ({percentage :.1f}%)")

            logger.info(f"  📈 Imbalance ratio: {class_analysis['imbalance_ratio']:.2f}:1")

            if class_analysis['severe_imbalance']:
                logger.warning("-------- Severe Class Imbalance !!!! ---------")
                logger.warning("-------- Will Apply aggresive balancing technique ---------")
            
            elif class_analysis['severe_imbalance']:
                logger.warning("  ⚠️  Moderate class imbalance detected")
                logger.warning("  💡 Will apply standard balancing techniques")

            else:
                logger.info("  ✅ Classes are reasonably balanced")
        
        except Exception as e:
            raise CustomException(e)
        
    def _handle_class_imbalance(self, train_df, class_analysis):
        """ Handle class imbalance through data augmentation"""
        try:
            logger.info("⚖️  Step 2: Handling class imbalance...")

            if not class_analysis['minority_classes']:
                logger.info("  ✅ No minority classes detected, skipping balancing")
                return train_df
            
            augmented_samples = []

            # Calculate target sample count (80% of majority class)
            target_samples = int(class_analysis['max_samples'] * 0.8)

            for minority_class in class_analysis['minority_classes']:
                current_count = class_analysis['class_counts'][minority_class]
                samples_needed = target_samples - current_count

                # Limit Augmentation to avoid overfitting
                max_augmentation = current_count * 2
                samples_to_create = min(max(samples_needed,0), max_augmentation)

                if samples_to_create > 0:
                    logger.info(f"  🔄 Augmenting '{minority_class}': {current_count:,} → {current_count + samples_to_create:,}")


                    # Create augmented samples for this class
                    class_df = train_df[train_df['label'] == minority_class].copy()
                    augmented_class_samples = self.create_augmented_samples(
                        class_df, samples_to_create, minority_class
                    )

                    augmented_samples.extend(augmented_class_samples)

            # Add augmented samples to training data
            if augmented_samples:
                augmented_df = pd.DataFrame(augmented_samples)
                train_df = pd.concat([train_df,augmented_df], ignore_index=True)

                self.preprocessing_stats['augmented_images_created'] = len(augmented_samples)
                logger.info(f"  ✅ Created {len(augmented_samples):,} augmented samples")
                logger.info(f"  📊 New training size: {len(train_df):,} samples")

            return train_df
        
        except Exception as e:
            raise CustomException(e)
        

    def _create_augmented_samples(self, class_df, samples_needed, class_name):
        """ Create augmented samples for a specific minority class"""
        try:
            augmented_samples = []

            # Randomly sample images to augment
            if len(class_df) >= samples_needed:
                sample_indices = np.random.choice(len(class_df), samples_needed ,replace=False)
            else:
                sample_indices = np.random.choice(len(class_df), samples_needed, replace=True)

            for i, idx in enumerate(sample_indices):
                original_row = class_df.iloc[idx]

                # Create augmented sample metadata
                augmented_sample = {
                    'images' : f"aug_{class_name}_{i:04d}_{original_row['images']}",
                    'label' : class_name,
                    'label_encoded' : original_row['label_encoded'],
                    'full_path' : original_row['full_path'],
                    'is_augmented' : True,
                    'augmented_id' : i,
                    'source_image' : original_row['images']

                }
                augmented_samples.append(augmented_sample)

            return augmented_samples
        
        except Exception as e:
            raise CustomException(e)
        
    