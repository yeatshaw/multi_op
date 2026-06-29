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

func_response = '''
thought: The method selects diverse starting cities by strategically choosing them based on their distances to each other, ensuring a well-spread and representative set for multi-start TSP construction.

```python
code:
def select_starts(self, distance_matrix):
    import numpy as np
    
    city_num = distance_matrix.shape[0]
    max_starts = min(self.max_starts, city_num)
    
    # Create a list to store selected starting cities
    starts = []
    remaining_cities = list(range(city_num))
    
    # Randomly choose the first start city
    first_city = np.random.choice(remaining_cities)
    starts.append(first_city)
    remaining_cities.remove(first_city)
    
    while len(starts) < max_starts:
        distances = np.array([min(distance_matrix[city, start] for start in starts) for city in remaining_cities])
        next_city_index = np.argmax(distances)
        next_city = remaining_cities[next_city_index]
        starts.append(next_city)
        remaining_cities.remove(next_city)
    
    return starts
```
'''
