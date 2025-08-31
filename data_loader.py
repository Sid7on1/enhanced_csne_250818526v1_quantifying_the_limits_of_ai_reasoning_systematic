import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants and configuration
@dataclass
class DataLoaderConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    num_channels: int
    data_format: str

@dataclass
class DataFormat(Enum):
    PNG = 'png'
    JPEG = 'jpg'

class DataLoaderException(Exception):
    pass

class DataFormatException(DataLoaderException):
    pass

class DataLoadingError(DataLoaderException):
    pass

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.data_format = config.data_format

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            if self.data_format == DataFormat.PNG:
                image = Image.open(image_path).convert('RGB')
            elif self.data_format == DataFormat.JPEG:
                image = Image.open(image_path).convert('RGB')
            else:
                raise DataFormatException(f"Unsupported data format: {self.data_format}")
            image = np.array(image)
            image = cv2.resize(image, self.image_size)
            image = image / 255.0
            return image
        except Exception as e:
            raise DataLoadingError(f"Failed to load image: {image_path}") from e

    def _load_dataset(self) -> List[np.ndarray]:
        dataset = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(self.data_format.value):
                    image_path = os.path.join(root, file)
                    dataset.append(self._load_image(image_path))
        return dataset

    def _create_data_loader(self) -> DataLoader:
        dataset = self._load_dataset()
        dataset = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return dataset

    def load_data(self) -> DataLoader:
        try:
            return self._create_data_loader()
        except Exception as e:
            raise DataLoadingError(f"Failed to load data") from e

class ImageDataset(Dataset):
    def __init__(self, data_dir: str, image_size: Tuple[int, int], num_channels: int, data_format: str):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_channels = num_channels
        self.data_format = data_format
        self.images = self._load_dataset()

    def _load_dataset(self) -> List[np.ndarray]:
        dataset = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(self.data_format.value):
                    image_path = os.path.join(root, file)
                    dataset.append(self._load_image(image_path))
        return dataset

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            if self.data_format == DataFormat.PNG:
                image = Image.open(image_path).convert('RGB')
            elif self.data_format == DataFormat.JPEG:
                image = Image.open(image_path).convert('RGB')
            else:
                raise DataFormatException(f"Unsupported data format: {self.data_format}")
            image = np.array(image)
            image = cv2.resize(image, self.image_size)
            image = image / 255.0
            return image
        except Exception as e:
            raise DataLoadingError(f"Failed to load image: {image_path}") from e

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index]

class DataLoaderFactory:
    @staticmethod
    def create_data_loader(config: DataLoaderConfig) -> DataLoader:
        return DataLoader(config)

if __name__ == "__main__":
    config = DataLoaderConfig(
        data_dir='/path/to/data',
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        num_channels=3,
        data_format=DataFormat.PNG
    )
    data_loader = DataLoaderFactory.create_data_loader(config)
    data_loader.load_data()