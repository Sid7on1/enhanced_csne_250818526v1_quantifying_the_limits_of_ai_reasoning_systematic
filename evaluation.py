import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import norm
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import norm as scipy_norm
from scipy