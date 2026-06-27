from __future__ import annotations

import copy
import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation
        self._required_feasible_offspring = pop_size

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def set_required_feasible_offspring(self, required: int | None):
        if required is None:
            self._required_feasible_offspring = self._pop_size
            return
        self._required_feasible_offspring = max(1, min(required, self._pop_size))

    def feasible_next_gen_size(self) -> int:
        return sum(1 for func in self._next_gen_pop if self._is_feasible(func))

    def clear_pending_offspring(self):
        self._next_gen_pop = []

    def snapshot(self) -> dict:
        return {
            "population": copy.deepcopy(self._population),
            "next_gen_pop": copy.deepcopy(self._next_gen_pop),
            "generation": self._generation,
            "required_feasible_offspring": self._required_feasible_offspring,
        }

    def restore(self, snapshot: dict):
        self._population = snapshot["population"]
        self._next_gen_pop = snapshot["next_gen_pop"]
        self._generation = snapshot["generation"]
        self._required_feasible_offspring = snapshot["required_feasible_offspring"]

    def survival(self):
        pop = self._population + self._next_gen_pop
        pop = sorted(pop, key=lambda f: f.score, reverse=True)
        self._population = pop[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            # register to next_gen
            self._next_gen_pop.append(func)
            # Only update the population after accumulating enough feasible offspring.
            if self.feasible_next_gen_size() >= self._required_feasible_offspring:
                self.survival()
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) and func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) and func.score == f.score:
                return True
        return False

    def selection(self) -> Function:
        funcs = [f for f in self._population if not math.isinf(f.score)]
        func = sorted(funcs, key=lambda f: f.score, reverse=True)
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)
        return np.random.choice(func, p=p)

    @staticmethod
    def _is_feasible(func: Function) -> bool:
        return func.score is not None and not math.isinf(func.score)
