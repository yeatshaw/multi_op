task_description_outer = "You need to design an efficient autoregressive algorithm to solve a traveling salesman problem."
task_description_inner = "Now there's a class named 'Algorithm' to solve the TSP problem."

problem_info = """
In the '__init__' method, the Algorithm class can only read the following values from kwargs:
    {kwargs:
      - city_num: int, the number of cities in the TSP instance.
      - tour_evaluation_function: callable, a function returns the total tour length which is a float. Its args and return are:
        args:
            tour: list[int], the visiting order of cities, where the first city is the starting city.
            distance_matrix: np.ndarray in the shape of (city_num, city_num), the pairwise city distance matrix.
        Return:
            total_distance: float, the total tour length.}
In the 'run' method, the Algorithm class can only read the following values from kwargs:
    {kwargs:
      - distance_matrix: np.ndarray in the shape of (city_num, city_num), the pairwise city distance matrix.}
The return value of 'run' must be:
    { - tour: list[int], the city visiting order, where the first city is the starting city.}
"""

algorithm_template = """
```python
Code:
{
import ...
class Algorithm:
    def __init__(self, **kwargs):
        self.ndim_problem = ...
        ...
        
    def run(self, **kwargs):
        ...
        
    def method_name1(self, arg1, arg2):
        ...
        return result
        
    def method_name2(self, arg1):
        ... 
}
```
        
method:
{
- run: this method is used to ...
- method_name1: this method is used to ...
- method_name2: this method is used to ...
}

Class_args:
{
- ndim_problem: int, Dimension of the problem.
}

method_args:
{    
    method_name1:
        Arg:
            - arg1: int, ...
            - arg2: float, ...
        Return:
            - result: float, ...
        
    method_name2:
        Arg:
            - arg1: str, ...
}
"""