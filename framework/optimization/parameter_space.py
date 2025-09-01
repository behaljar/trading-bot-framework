"""
Parameter space definition for strategy optimization.

This module defines the structure for parameter search spaces
used in optimization algorithms.
"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ParameterRange:
    """Defines a parameter range for optimization."""
    
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Optional[Union[int, float]] = None
    param_type: str = 'float'  # 'float', 'int', or 'choice'
    choices: Optional[List[Any]] = None
    
    def __post_init__(self):
        """Validate parameter range."""
        if self.param_type == 'choice':
            if not self.choices:
                raise ValueError(f"Parameter {self.name} of type 'choice' must have choices defined")
        else:
            if self.min_value > self.max_value:
                raise ValueError(f"Parameter {self.name}: min_value must be <= max_value")
            
            if self.param_type == 'int':
                self.min_value = int(self.min_value)
                self.max_value = int(self.max_value)
                if self.step:
                    self.step = int(self.step)
    
    def get_grid_values(self) -> List[Any]:
        """Get all values for grid search."""
        if self.param_type == 'choice':
            return self.choices
        elif self.param_type == 'int':
            step = self.step or 1
            return list(range(self.min_value, self.max_value + 1, step))
        else:  # float
            if self.step:
                # Use numpy for float ranges with step
                return list(np.arange(self.min_value, self.max_value + self.step/2, self.step))
            else:
                # Default to 10 steps for float without explicit step
                return list(np.linspace(self.min_value, self.max_value, 10))
    
    def get_random_value(self) -> Any:
        """Get a random value from the parameter range."""
        if self.param_type == 'choice':
            return np.random.choice(self.choices)
        elif self.param_type == 'int':
            return np.random.randint(self.min_value, self.max_value + 1)
        else:  # float
            if self.step:
                # Respect step size for random values
                n_steps = int((self.max_value - self.min_value) / self.step) + 1
                random_step = np.random.randint(0, n_steps)
                return self.min_value + random_step * self.step
            else:
                return np.random.uniform(self.min_value, self.max_value)


class ParameterSpace:
    """
    Defines the parameter search space for optimization.
    
    Example:
        space = ParameterSpace()
        space.add_parameter('short_window', min_value=5, max_value=50, step=5, param_type='int')
        space.add_parameter('long_window', min_value=20, max_value=200, step=10, param_type='int')
        space.add_parameter('stop_loss_pct', min_value=0.01, max_value=0.05, step=0.01)
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterRange] = {}
    
    def add_parameter(self, 
                     name: str,
                     min_value: Union[int, float] = None,
                     max_value: Union[int, float] = None, 
                     step: Optional[Union[int, float]] = None,
                     param_type: str = 'float',
                     choices: Optional[List[Any]] = None) -> 'ParameterSpace':
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            min_value: Minimum value (not needed for 'choice' type)
            max_value: Maximum value (not needed for 'choice' type)
            step: Step size for grid search (optional)
            param_type: Type of parameter ('float', 'int', or 'choice')
            choices: List of choices for 'choice' type parameters
            
        Returns:
            Self for method chaining
        """
        if param_type == 'choice':
            # For choice parameters, set dummy min/max values
            param = ParameterRange(
                name=name,
                min_value=0,
                max_value=len(choices) - 1 if choices else 0,
                param_type=param_type,
                choices=choices
            )
        else:
            if min_value is None or max_value is None:
                raise ValueError(f"Parameter {name}: min_value and max_value required for type {param_type}")
            
            param = ParameterRange(
                name=name,
                min_value=min_value,
                max_value=max_value,
                step=step,
                param_type=param_type
            )
        
        self.parameters[name] = param
        return self
    
    def get_grid_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.
        
        Returns:
            List of parameter dictionaries
        """
        if not self.parameters:
            return [{}]
        
        # Get all parameter values
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name].get_grid_values() for name in param_names]
        
        # Generate all combinations using numpy meshgrid
        combinations = []
        import itertools
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)
        
        return combinations
    
    def get_random_combination(self) -> Dict[str, Any]:
        """
        Generate a random parameter combination.
        
        Returns:
            Dictionary with random parameter values
        """
        return {
            name: param.get_random_value()
            for name, param in self.parameters.items()
        }
    
    def get_total_combinations(self) -> int:
        """Get total number of combinations for grid search."""
        if not self.parameters:
            return 0
        
        total = 1
        for param in self.parameters.values():
            total *= len(param.get_grid_values())
        
        return total
    
    def __repr__(self) -> str:
        """String representation."""
        lines = ["ParameterSpace:"]
        for name, param in self.parameters.items():
            if param.param_type == 'choice':
                lines.append(f"  {name}: choices={param.choices}")
            else:
                lines.append(f"  {name}: [{param.min_value}, {param.max_value}] "
                           f"step={param.step} type={param.param_type}")
        return "\n".join(lines)