import numpy as np
from random import randint, random, sample
from copy import deepcopy
from math import ceil
import time
from policy import Policy

class Policy2350030(Policy):
    def __init__(self):
        self.MAX_ITER = 100
        self.pop_size = 100
        self.generations = 200
        self.mutation_rate = 0.01
        self.elite_size = 3
        self.population = []
        self.best_solution = None
        self.lengthArr = []
        self.widthArr = []
        self.demandArr = []
        self.N = 0

    def initialize_population(self, maxRepeatArr):
        return [
            [i, randint(1, maxRepeatArr[i])]
            for _ in range(self.pop_size)
            for i in np.argsort(-np.array(self.lengthArr) * np.array(self.widthArr))
        ]

    def _can_place_(self, stock, position, prod_size, rotated=False):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size if not rotated else (prod_size[1], prod_size[0])
        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

    def calculate_fitness(self, chromosome, patterns):
        return sum(
            np.sum(patterns[chromosome[i]]) * chromosome[i + 1]
            for i in range(0, len(chromosome), 2)
        )

    def generate_efficient_patterns(self, stockLength, stockWidth):
        patterns = []
        stack = [([0] * self.N, 0, 0)]
        while stack:
            current_pattern, length_used, width_used = stack.pop()
            for i in range(self.N):
                if i >= len(self.lengthArr) or i >= len(self.widthArr) or i >= len(self.demandArr):
                    continue
                max_repeat = min(
                    (stockLength - length_used) // self.lengthArr[i],
                    (stockWidth - width_used) // self.widthArr[i],
                    self.demandArr[i],
                )
                if max_repeat > 0:
                    new_pattern = current_pattern.copy()
                    new_pattern[i] += max_repeat
                    patterns.append(new_pattern)
                    stack.append(
                        (
                            new_pattern,
                            length_used + max_repeat * self.lengthArr[i],
                            width_used + max_repeat * self.widthArr[i],
                        )
                    )
        return patterns

    def max_pattern_exist(self, patterns):
        return [
            max(ceil(self.demandArr[i] / pattern[i]) for i in range(len(pattern)) if pattern[i] > 0)
            for pattern in patterns
        ]

    def crossover(self, parent1, parent2):
        crossover_point = randint(0, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, chromosome, mutation_rate):
        for i in range(len(chromosome)):
            if random() < mutation_rate:
                swap_idx = randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[swap_idx] = chromosome[swap_idx], chromosome[i]
        return chromosome

    def select_parents(self, fitness_s, population):
        total_fitness = np.sum(fitness_s)
        pick = np.random.rand() * total_fitness
        cumulative_fitness = np.cumsum(fitness_s)
        parent_index = np.searchsorted(cumulative_fitness, pick)
        return population[parent_index]

    def evolve(self, population, new_population, patterns, fitness_s, mutation_rate, max_repeat_arr):
        while len(new_population) < self.pop_size:
            parent1 = self.select_parents(fitness_s, population)
            parent2 = self.select_parents(fitness_s, population)
            while parent1 == parent2:
                parent2 = self.select_parents(fitness_s, population)
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate)
            new_population.append(child)
        return new_population

    def run_genetic_algorithm(self, patterns, population, max_repeat_arr):
        start_time = time.time()
        best_results = []
        for _ in range(self.MAX_ITER):
            fitness_pairs = [
                (ch, self.calculate_fitness(ch, patterns)) for ch in self.population
            ]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)
            new_population = deepcopy([sc[0] for sc in fitness_pairs[: self.elite_size]])
            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)
            next_gen = self.evolve(
                population,
                new_population,
                patterns,
                [sc[1] for sc in fitness_pairs],
                self.mutation_rate,
                max_repeat_arr,
            )
            self.population = deepcopy(next_gen[: self.pop_size])
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return best_solution, best_fitness, best_results

    def create_new_pop(self, population):
        return [
            self.mutate(self.crossover(*sample(population, 2)), self.mutation_rate)
            for _ in range(self.pop_size)
        ]

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        if not list_prods or not stocks:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        self.lengthArr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.widthArr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demandArr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.N = len(self.lengthArr)  # Ensure N is set correctly
        if self.N == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        first_stock = stocks[0]
        stock_Length, stock_Width = self._get_stock_size_(first_stock)
        patterns = self.generate_efficient_patterns(stock_Length, stock_Width)
        maxRepeatArr = self.max_pattern_exist(patterns)
        self.population = self.initialize_population(maxRepeatArr)
        best_solution, _, _ = self.run_genetic_algorithm(
            patterns, self.population, maxRepeatArr
        )
        for i in range(0, len(best_solution), 2):
            pattern_index = best_solution[i]
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                for x in range(stock_w):
                    for y in range(stock_h):
                        if pattern_index >= len(self.lengthArr):
                            continue
                        prod_size = (self.lengthArr[pattern_index], self.widthArr[pattern_index])
                        if self._can_place_(stock, (x, y), prod_size):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y),
                                "rotated": False,
                            }
                        elif self._can_place_(stock, (x, y), prod_size, rotated=True):
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_size[1], prod_size[0]),
                                "position": (x, y),
                                "rotated": True,
                            }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}