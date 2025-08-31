import logging
import math
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants and configuration
VELOCITY_THRESHOLD = 0.5  # velocity threshold constant from the research paper
FLOW_THEORY_CONSTANT = 1.2  # flow theory constant from the research paper

class UtilityFunctions:
    """
    A class containing various utility functions for the computer vision project.
    """

    @staticmethod
    def validate_input(input_data: Any) -> bool:
        """
        Validate the input data.

        Args:
        input_data (Any): The input data to be validated.

        Returns:
        bool: True if the input data is valid, False otherwise.
        """
        try:
            if input_data is None:
                raise ValueError("Input data cannot be None")
            return True
        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return False

    @staticmethod
    def calculate_velocity(vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate the velocity between two vectors.

        Args:
        vector1 (List[float]): The first vector.
        vector2 (List[float]): The second vector.

        Returns:
        float: The calculated velocity.
        """
        try:
            if len(vector1) != len(vector2):
                raise ValueError("Both vectors must have the same length")
            velocity = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
            return velocity
        except Exception as e:
            logger.error(f"Error calculating velocity: {str(e)}")
            return None

    @staticmethod
    def apply_velocity_threshold(velocity: float) -> bool:
        """
        Apply the velocity threshold to determine if the velocity is within the allowed range.

        Args:
        velocity (float): The calculated velocity.

        Returns:
        bool: True if the velocity is within the allowed range, False otherwise.
        """
        try:
            if velocity is None:
                raise ValueError("Velocity cannot be None")
            return velocity <= VELOCITY_THRESHOLD
        except Exception as e:
            logger.error(f"Error applying velocity threshold: {str(e)}")
            return False

    @staticmethod
    def calculate_flow_theory(vector: List[float]) -> float:
        """
        Calculate the flow theory value for a given vector.

        Args:
        vector (List[float]): The input vector.

        Returns:
        float: The calculated flow theory value.
        """
        try:
            if len(vector) == 0:
                raise ValueError("Input vector cannot be empty")
            flow_theory_value = sum(x * FLOW_THEORY_CONSTANT for x in vector)
            return flow_theory_value
        except Exception as e:
            logger.error(f"Error calculating flow theory value: {str(e)}")
            return None

    @staticmethod
    def create_data_frame(data: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from a dictionary of data.

        Args:
        data (Dict[str, List[Any]]): The input data.

        Returns:
        pd.DataFrame: The created DataFrame.
        """
        try:
            if not data:
                raise ValueError("Input data cannot be empty")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            return None

    @staticmethod
    def convert_to_tensor(data: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.

        Args:
        data (np.ndarray): The input numpy array.

        Returns:
        torch.Tensor: The converted tensor.
        """
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
            return torch.from_numpy(data)
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            return None

class Configuration:
    """
    A class containing configuration settings for the computer vision project.
    """

    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize the configuration settings.

        Args:
        settings (Dict[str, Any]): The input settings.
        """
        self.settings = settings

    def get_setting(self, key: str) -> Any:
        """
        Get a specific setting by key.

        Args:
        key (str): The key of the setting.

        Returns:
        Any: The value of the setting.
        """
        try:
            if key not in self.settings:
                raise ValueError(f"Setting '{key}' not found")
            return self.settings[key]
        except Exception as e:
            logger.error(f"Error getting setting: {str(e)}")
            return None

class ExceptionClasses:
    """
    A class containing custom exception classes for the computer vision project.
    """

    class InvalidInputError(Exception):
        """
        A custom exception class for invalid input errors.
        """

        def __init__(self, message: str):
            """
            Initialize the exception.

            Args:
            message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

    class CalculationError(Exception):
        """
        A custom exception class for calculation errors.
        """

        def __init__(self, message: str):
            """
            Initialize the exception.

            Args:
            message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

def main():
    # Example usage of the utility functions
    vector1 = [1.0, 2.0, 3.0]
    vector2 = [4.0, 5.0, 6.0]
    velocity = UtilityFunctions.calculate_velocity(vector1, vector2)
    if velocity is not None:
        logger.info(f"Calculated velocity: {velocity}")
        if UtilityFunctions.apply_velocity_threshold(velocity):
            logger.info("Velocity is within the allowed range")
        else:
            logger.info("Velocity is outside the allowed range")
    else:
        logger.error("Failed to calculate velocity")

    flow_theory_value = UtilityFunctions.calculate_flow_theory(vector1)
    if flow_theory_value is not None:
        logger.info(f"Calculated flow theory value: {flow_theory_value}")
    else:
        logger.error("Failed to calculate flow theory value")

    data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    data_frame = UtilityFunctions.create_data_frame(data)
    if data_frame is not None:
        logger.info("Created DataFrame:")
        logger.info(data_frame)
    else:
        logger.error("Failed to create DataFrame")

    numpy_array = np.array([1.0, 2.0, 3.0])
    tensor = UtilityFunctions.convert_to_tensor(numpy_array)
    if tensor is not None:
        logger.info("Converted to tensor:")
        logger.info(tensor)
    else:
        logger.error("Failed to convert to tensor")

if __name__ == "__main__":
    main()