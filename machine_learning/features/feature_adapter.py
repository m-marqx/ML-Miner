"""
Feature Adapter Module
=====================

This module provides an adapter pattern implementation for unifying the
interface of different feature generation classes in the machine
learning pipeline.

The FeaturesAdapter class wraps either ModelFeatures or OnchainFeatures
instances to provide a consistent interface for feature generation
operations, making the code more maintainable and reducing coupling
between components.

Classes
-------
FeaturesAdapter
    Adapter class that provides a unified interface for feature generation

Example
-------
>>> from machine_learning.model_features import ModelFeatures
>>> from machine_learning.features.feature_adapter import FeaturesAdapter
>>>
>>> features = ModelFeatures()
>>> adapter = FeaturesAdapter(features)
>>> adapter.run_method('create_feature', feature_name='my_feature')

See Also
--------
machine_learning.model_features : Contains the ModelFeatures class
machine_learning.features.onchain_features : Contains the
OnchainFeatures class
"""

import pandas as pd

from machine_learning.model_features import ModelFeatures
from machine_learning.features.onchain_features import OnchainFeatures

class FeaturesAdapter:
    """
    Adapter class for unified feature creation interface.

    This class provides a common interface for working with both
    ModelFeatures and OnchainFeatures classes. It abstracts away the
    differences between these feature generators and provides consistent
    method access.

    Parameters
    ----------
    class_obj : ModelFeatures | OnchainFeatures
        The feature generator class instance to adapt

    Methods
    -------
    run_method(method_name: str, *method_args, **method_kwargs)
        Executes the named method on the underlying feature class

    get_method_docstring(method_name: str) -> str
        Retrieves and prints the docstring for the specified method

    Properties
    ----------
    dataset : pd.DataFrame
        Access the underlying feature class's dataset
    """
    def __init__(
        self,
        class_obj: ModelFeatures | OnchainFeatures,
    ):
        """
        Initialize the FeaturesAdapter with a feature generator class
        instance.

        Parameters
        ----------
        class_obj : ModelFeatures | OnchainFeatures
            The feature generator class instance to adapt. Must be
            either a ModelFeatures or OnchainFeatures instance.
        """
        self.class_obj = class_obj

    def run_method(self, method_name: str, *method_args, **method_kwargs):
        """
        Execute a method on the underlying feature class instance.

        Parameters
        ----------
        method_name : str
            Name of the method to execute
        *method_args
            Variable length argument list to pass to the method
        **method_kwargs
            Arbitrary keyword arguments to pass to the method

        Returns
        -------
        FeaturesAdapter
            Returns self for method chaining

        Raises
        ------
        AttributeError
            If the method_name doesn't exist in the underlying class
        """
        method = getattr(self.class_obj, method_name)
        method(*method_args, **method_kwargs)
        return self

