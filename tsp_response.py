response = '''
Code:
```python
import numpy as np

class Algorithm:
    def __init__(self, **kwargs):
        self.ndim_problem = None
        self.city_num = kwargs.get('city_num')
        self.tour_evaluation_function = kwargs.get('tour_evaluation_function')
        self.max_starts = 10
        self.max_2opt_passes = 30

    def run(self, **kwargs):
        distance_matrix = kwargs.get('distance_matrix')
        starts = self.select_starts(distance_matrix)
        best_tour, best_cost = None, float('inf')
        for s in starts:
            tour = self.construct_tour(distance_matrix, s)
            tour = self.local_search(tour, distance_matrix)
            cost = self.tour_evaluation_function(tour, distance_matrix)
            if cost < best_cost:
                best_cost, best_tour = cost, tour
        return best_tour

    def select_starts(self, distance_matrix):
        n = self.city_num
        k = min(self.max_starts, n)
        starts = [int(round(i * n / k)) % n for i in range(k)]
        return sorted(set(starts))

    def construct_tour(self, distance_matrix, start):
        n = self.city_num
        visited = np.zeros(n, dtype=bool)
        tour = [int(start)]
        visited[start] = True
        current = int(start)
        for _ in range(n - 1):
            dist = distance_matrix[current].astype(float).copy()
            dist[visited] = np.inf
            nxt = int(np.argmin(dist))
            tour.append(nxt)
            visited[nxt] = True
            current = nxt
        return tour

    def local_search(self, tour, distance_matrix):
        n = len(tour)
        best = list(tour)
        improved, passes = True, 0
        while improved and passes < self.max_2opt_passes:
            improved, passes = False, passes + 1
            for i in range(n - 1):
                a, b = best[i], best[i + 1]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c, d = best[j], best[(j + 1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]
                             - distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-10:
                        best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                        improved = True
                        a, b = best[i], best[i + 1]
        return best
```

method:
- run: this method is used to drive the whole pipeline: it picks several starting cities, autoregressively constructs a tour from each, refines every tour with local search, evaluates them, and returns the lowest-cost tour.
- select_starts: this method is used to choose a diverse, evenly spread set of starting cities to diversify the multi-start construction.
- construct_tour: this method is used to build a tour autoregressively: starting from one city, it repeatedly appends the nearest unvisited city, conditioning each decision on the already-built prefix.
- local_search: this method is used to improve a constructed tour with the 2-opt heuristic by reversing segments whenever it shortens the total length.

Class_args:
- ndim_problem: int, Placeholder for problem dimension (kept for template compatibility).
- city_num: int, Number of cities in the TSP problem.
- tour_evaluation_function: callable, Function returning the total distance of a tour given the distance matrix.
- max_starts: int, Maximum number of starting cities for multi-start construction.
- max_2opt_passes: int, Maximum number of full 2-opt improvement passes per tour.

method_args:
    select_starts:
        Arg:
            - distance_matrix: np.ndarray, The distance matrix, used to size the start set.
        Return:
            - starts: list, Distinct starting city indices for the multi-start construction.

    construct_tour:
        Arg:
            - distance_matrix: np.ndarray, The distance matrix guiding nearest-city selection.
            - start: int, The starting city index for this construction.
        Return:
            - tour: list, A complete tour built autoregressively from the start city.

    local_search:
        Arg:
            - tour: list, The tour to be refined.
            - distance_matrix: np.ndarray, The distance matrix used to score 2-opt moves.
        Return:
            - best: list, The locally optimized tour after 2-opt.
'''

template_program = '''
import numpy as np
from typing import Tuple, List
def mutate(self, x=None, y=None, a=None):
    """
    Arg:
        self: The instance of the class containing the evolution computation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - h: int, Length of historical memory.
            - p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - lower_boundary: float, Lower boundary of the problem.
            - upper_boundary: float, Upper boundary of the problem.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
        x: np.array, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.array, The fitness of current population of individuals, shape=(self.n_individuals,).
        a: np.array, External archive used in lshade, shape=(n, self.ndim_problem).
    Returns:
        x_mu: np.array, Population individuals after mutation, shape=(self.n_individuals, self.ndim_problem).
        f_mu: np.array, Scaling factor F used during mutation, shape=(self.n_individuals,).
        r: np.array, Index for selecting the scaling factor F and crossover rate CR, shape=(self.n_individuals,)
    """
'''

description = '''
main_stream:
def run(self, **kwargs):
    distance_matrix = kwargs.get('distance_matrix')
    starts = self.select_starts(distance_matrix)
    best_tour, best_cost = None, float('inf')
    for s in starts:
        tour = self.construct_tour(distance_matrix, s)
        tour = self.local_search(tour, distance_matrix)
        cost = self.tour_evaluation_function(tour, distance_matrix)
        if cost < best_cost:
            best_cost, best_tour = cost, tour
    return best_tour

method_introduction:
- run: this method is used to drive the whole pipeline: it picks several starting cities, autoregressively constructs a tour from each, refines every tour with local search, evaluates them, and returns the lowest-cost tour.
- select_starts: this method is used to choose a diverse, evenly spread set of starting cities to diversify the multi-start construction.
- construct_tour: this method is used to build a tour autoregressively: starting from one city, it repeatedly appends the nearest unvisited city, conditioning each decision on the already-built prefix.
- local_search: this method is used to improve a constructed tour with the 2-opt heuristic by reversing segments whenever it shortens the total length.

Class_args:
- ndim_problem: int, Placeholder for problem dimension (kept for template compatibility).
- city_num: int, Number of cities in the TSP problem.
- tour_evaluation_function: callable, Function returning the total distance of a tour given the distance matrix.
- max_starts: int, Maximum number of starting cities for multi-start construction.
- max_2opt_passes: int, Maximum number of full 2-opt improvement passes per tour.
'''

response1='''
```python
Code:
{
import numpy as np
import random

class Algorithm:
    def __init__(self, **kwargs):
        self.city_num = kwargs['city_num']
        self.tour_evaluation_function = kwargs['tour_evaluation_function']
        
    def run(self, **kwargs):
        distance_matrix = kwargs['distance_matrix']
        initial_tour = self.generate_initial_tour()
        best_tour = initial_tour
        best_distance = self.tour_evaluation_function(best_tour, distance_matrix)

        for _ in range(1000):  # Example iteration limit
            new_tour = self.local_search(best_tour)
            new_distance = self.tour_evaluation_function(new_tour, distance_matrix)
            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
        
        return best_tour
    
    def generate_initial_tour(self):
        tour = list(range(self.city_num))
        random.shuffle(tour)
        return tour
    
    def local_search(self, current_tour):
        best_tour = current_tour[:]
        for i in range(len(current_tour) - 1):
            for j in range(i + 1, len(current_tour)):
                new_tour = best_tour[:]
                new_tour[i:j+1] = reversed(best_tour[i:j+1])
                if self.tour_evaluation_function(new_tour, np.zeros((self.city_num, self.city_num))) < self.tour_evaluation_function(best_tour, np.zeros((self.city_num, self.city_num))):
                    best_tour = new_tour
        return best_tour
}
```

method:
{
- run: this method is used to execute the main algorithm, generating an initial tour and applying a local search to find the best tour.
- generate_initial_tour: this method is used to create a random initial visiting order of the cities.
- local_search: this method is used to improve the current tour by checking for better arrangements through pairwise swaps.
}

Class_args:
{
- city_num: int, the number of cities in the TSP instance.
- tour_evaluation_function: callable, a function that evaluates the total length of a given tour based on the distance matrix.
}

method_args:
{    
    generate_initial_tour:
        Arg:
            - None: Generates a random initial tour.
        Return:
            - tour: list[int], a randomly shuffled list representing the order of cities.

    local_search:
        Arg:
            - current_tour: list[int], the currently known best visiting order of cities.
        Return:
            - best_tour: list[int], a potentially improved visiting order of cities after local search.
}
'''
