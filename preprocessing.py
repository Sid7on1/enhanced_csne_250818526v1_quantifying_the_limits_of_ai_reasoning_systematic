import logging
import numpy as np
import cv2
import torch
from typing import Tuple, List, Dict
from PIL import Image
from torchvision import transforms
from config import Config
from utils import load_config, get_logger

logger = get_logger(__name__)

class ImagePreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.transforms = self._create_transforms()

    def _create_transforms(self) -> transforms.Compose:
        """Create a composition of transforms for image preprocessing."""
        transforms_list = [
            transforms.Resize((self.config.image_height, self.config.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ]
        return transforms.Compose(transforms_list)

    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate the input image."""
        if not isinstance(image, np.ndarray):
            logger.error("Invalid image type. Expected numpy array.")
            return False
        if image.ndim != 3:
            logger.error("Invalid image dimensions. Expected 3D array.")
            return False
        if image.shape[2] != 3:
            logger.error("Invalid image channels. Expected 3 channels.")
            return False
        return True

    def _validate_config(self) -> bool:
        """Validate the configuration."""
        if not isinstance(self.config, Config):
            logger.error("Invalid configuration. Expected Config object.")
            return False
        if not self.config.image_height or not self.config.image_width:
            logger.error("Invalid image dimensions. Expected positive integers.")
            return False
        if not self.config.mean or not self.config.std:
            logger.error("Invalid mean and std values. Expected lists of floats.")
            return False
        return True

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the input image."""
        if not self._validate_image(image):
            raise ValueError("Invalid image")
        if not self._validate_config():
            raise ValueError("Invalid configuration")
        image = Image.fromarray(image)
        image = self.transforms(image)
        return image

class Config:
    def __init__(self, image_height: int, image_width: int, mean: List[float], std: List[float]):
        self.image_height = image_height
        self.image_width = image_width
        self.mean = mean
        self.std = std

def load_config(config_file: str) -> Config:
    """Load the configuration from a file."""
    config = Config(0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            config.image_height = config_dict['image_height']
            config.image_width = config_dict['image_width']
            config.mean = config_dict['mean']
            config.std = config_dict['std']
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError:
        logger.error(f"Invalid configuration file: {config_file}")
    return config

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('preprocessing.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    config_file = 'config.json'
    config = load_config(config_file)
    logger.info(f"Loaded configuration: {config}")
    image = np.random.rand(256, 256, 3)
    preprocessor = ImagePreprocessor(config)
    image_tensor = preprocessor.preprocess_image(image)
    logger.info(f"Preprocessed image shape: {image_tensor.shape}")