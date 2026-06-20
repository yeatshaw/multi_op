import numpy as np
import random
from utils import *

def evaluate_tour(tour, distance_matrix):
    total_distance = 0
    if set(tour) != set(range(distance_matrix.shape[0])):
        return float('inf')  # Invalid tour
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i + 1]]
    total_distance += distance_matrix[tour[-1], tour[0]]  # Return to the starting city
    return total_distance

class Algorithm:
    def __init__(self, **kwargs):
        self.city_num = kwargs['city_num']
        self.tour_evaluation_function = kwargs['tour_evaluation_function']
        self.best_tour = None
        self.best_distance = float('inf')
    def run(self, **kwargs):
        distance_matrix = kwargs['distance_matrix']
        self.initialize_population()
        self.evolve_population(distance_matrix)
        return self.best_tour
    def initialize_population(self):
        # Initialize with a random tour
        self.best_tour = list(range(self.city_num))
        random.shuffle(self.best_tour)
        self.best_distance = self.tour_evaluation_function(self.best_tour, None)
    def evolve_population(self, distance_matrix):
        for _ in range(1000):  # Number of iterations
            new_tour = self.mutate(self.best_tour)
            new_distance = self.tour_evaluation_function(new_tour, distance_matrix)
            if new_distance < self.best_distance:
                self.best_distance = new_distance
                self.best_tour = new_tour
    def mutate(self, tour):
        # Swap two cities to create a new tour
        new_tour = tour[:]
        idx1, idx2 = random.sample(range(self.city_num), 2)
        new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
        return new_tour
    
problem_size = 100
instance = []
distance_matrix_dict, _ = load_tsp_dictionaries()
instance.append(distance_matrix_dict['bays29'])
#for value in distance_matrix_dict.values():
#    if value[0].shape[0] <= problem_size:
#        instance.append(value)

distance_matrix, optimum_distance = instance[0]
problem_size = distance_matrix.shape[0]
    
algorithm = Algorithm(city_num=problem_size, tour_evaluation_function=evaluate_tour)            
tour = algorithm.run(distance_matrix=distance_matrix)
distance = evaluate_tour(tour, distance_matrix)
print((distance - optimum_distance)/optimum_distance)