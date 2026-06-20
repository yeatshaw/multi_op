from __future__ import annotations

from typing import Any
import numpy as np
import types
import multiprocessing as mp
from LLM4AD.llm4ad.base import Evaluation
from tsp_template import *
__all__ = ['TSPEvaluation']

def evaluate_tour(tour, distance_matrix):
    total_distance = 0
    if set(tour) != set(range(distance_matrix.shape[0])):
        return float('inf')  # Invalid tour
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i + 1]]
    total_distance += distance_matrix[tour[-1], tour[0]]  # Return to the starting city
    return total_distance

class TSPEvaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=600,
                 **kwargs):
        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=kwargs.get("template_program", None),
            task_description=task_description_inner,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        
        self.instance = kwargs.get("instance", None)
        self.method_name = kwargs.get("method_name", None)
        self.algorithm_str = kwargs.get("algorithm_str", None)
        
    def evaluate_program(self, program_str: str = None, callable_func: callable = None) -> Any | None:
        return self.evaluate(program_str)

    def evaluate(self, program_str: str) -> Any | None:
        pool = mp.Pool(processes=len(self.instance))
        try:
            results = pool.starmap_async(
                self.core,
                [(instance, i, program_str) for i, instance in enumerate(self.instance)],
            ).get(timeout=self.timeout_seconds)
        except mp.TimeoutError as exc:
            pool.terminate()
            raise TimeoutError(
                f"TSP evaluation exceeded {self.timeout_seconds} seconds."
            ) from exc
        except Exception:
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            pool.join()
        return -np.mean(results)
    
    def core(self, instance, random_seed, program_str: str) -> Any | None:
        np.random.seed(random_seed)
        distance_matrix, optimum_distance = instance
        problem_size = distance_matrix.shape[0]
    
        namespace = {}
        exec(self.algorithm_str, namespace)
        algorithm = namespace['Algorithm'](city_num=problem_size, tour_evaluation_function=evaluate_tour)
    
        if program_str is not None:
            namespace = {}
            exec(program_str, namespace)
            method_callable = namespace[self.method_name]
            setattr(algorithm, self.method_name, types.MethodType(method_callable, algorithm))
            
        tour = algorithm.run(distance_matrix=distance_matrix)
        distance = evaluate_tour(tour, distance_matrix)
        return (distance - optimum_distance)/optimum_distance
