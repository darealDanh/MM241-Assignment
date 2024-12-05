import numpy as np
import random
from typing import List, Tuple, Dict
from copy import deepcopy
from collections import defaultdict


class Chromosome:
    def __init__(self, stock_size: Tuple[int, int], pieces: List[Dict]):
        self.stock_size = stock_size
        self.pieces = pieces  # List of piece placements
        self.fitness = 0
        self.layout = np.zeros(stock_size)

    def is_valid_placement(self, piece: Dict, pos: Tuple[int, int]) -> bool:
        x, y = pos
        w, h = piece["size"]

        if x + w > self.stock_size[0] or y + h > self.stock_size[1]:
            return False

        return not np.any(self.layout[y : y + h, x : x + w] > 0)


class Policy2350030(Chromosome):
    def __init__(self):
        self.pop_size = 50
        self.generations = 50
        self.mutation_rate = 0.1
        self.elite_size = 5
        self.population = []
        self.best_solution = None

    def initialize_population(
        self, stock_size: Tuple[int, int], pieces: List[Dict]
    ) -> List[Chromosome]:
        print("======initialize=======")
        print(pieces)
        population = []
        for _ in range(self.pop_size):
            chromosome = Chromosome(stock_size, deepcopy(pieces))
            # Randomly place pieces
            for piece in chromosome.pieces:
                placed = False
                attempts = 0
                while not placed and attempts < 100:
                    if (
                        piece["size"][0] < stock_size[0]
                        and piece["size"][1] < stock_size[1]
                    ):
                        x = random.randint(0, stock_size[0] - piece["size"][0])
                        y = random.randint(0, stock_size[1] - piece["size"][1])
                        if chromosome.is_valid_placement(piece, (x, y)):
                            piece["position"] = (x, y)
                            placed = True
                            print(x, y)
                    attempts += 1
            population.append(chromosome)
        return population

    def calculate_fitness(self, chromosome: Chromosome) -> float:
        used_area = 0
        for piece in chromosome.pieces:
            if "position" in piece:
                used_area += piece["size"][0] * piece["size"][1]
        return used_area / (chromosome.stock_size[0] * chromosome.stock_size[1])

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        child = Chromosome(parent1.stock_size, deepcopy(parent1.pieces))
        crossover_point = random.randint(0, len(parent1.pieces))
        child_pieces = list(child.pieces)
        child_pieces[:crossover_point] = deepcopy(parent1.pieces[:crossover_point])
        child_pieces[crossover_point:] = deepcopy(parent2.pieces[crossover_point:])
        child.pieces = tuple(child_pieces)
        return child

    def mutate(self, chromosome: Chromosome):
        if random.random() < self.mutation_rate:
            piece_idx = random.randint(0, len(chromosome.pieces) - 1)
            piece = chromosome.pieces[piece_idx]
            if "position" in piece:
                x = random.randint(0, chromosome.stock_size[0] - piece["size"][0])
                y = random.randint(0, chromosome.stock_size[1] - piece["size"][1])
                if chromosome.is_valid_placement(piece, (x, y)):
                    piece["position"] = (x, y)

    def select_parents(self, population: List[Chromosome]) -> List[Chromosome]:
        # Tournament selection
        tournament_size = 5
        parents = []
        for _ in range(2):
            tournament = random.sample(population, tournament_size)
            # print tournament out
            # for i in range(tournament_size):
            #     print(tournament[i].pieces)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

    def evolve(self, population: List[Chromosome]) -> List[Chromosome]:
        # Calculate fitness for all chromosomes
        for chromosome in population:
            chromosome.fitness = self.calculate_fitness(chromosome)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Keep elite solutions
        new_population = deepcopy(population[: self.elite_size])

        # Generate rest of population through crossover and mutation
        while len(new_population) < self.pop_size:
            parents = self.select_parents(population)
            child = self.crossover(parents[0], parents[1])
            self.mutate(child)
            new_population.append(child)

        return new_population

    def get_action(self, observation, info):
        if not observation["products"]:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        if len(observation["products"]) <= 1:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        stock = observation["stocks"][1]
        # print("stock: ", stock)
        stock_width = np.sum(np.any(stock != 2, axis=1))
        stock_height = np.sum(np.any(stock != 2, axis=0))
        print("stock_width: ", stock_width)
        print("stock_height: ", stock_height)
        # Initialize population if first time
        if not self.population:
            self.population = self.initialize_population(
                (
                    stock_width,
                    stock_height,
                ),
                observation["products"],
            )

        # Evolve for specified generations
        for _ in range(self.generations):
            self.population = self.evolve(self.population)

        # Get best solution
        best = max(self.population, key=lambda x: x.fitness)

        for piece in best.pieces:
            print(piece)

        # Return next piece placement from best solution
        if len(observation["products"]) > 1:
            next_piece = observation["products"][1]
            for piece in best.pieces:
                if (
                    "size" in piece
                    and piece["size"][0] == next_piece["size"][0]
                    and piece["size"][1] == next_piece["size"][1]
                    and "position" in piece
                ):
                    print(piece)
                    return {
                        "stock_idx": 0,
                        "size": piece["size"],
                        "position": piece["position"],
                    }
        print("stock_idx: ", -1)
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


# class Particle:
#     def __init__(self, stock_size: Tuple[int, int], pieces: List[Dict]):
#         self.stock_size = stock_size
#         self.pieces = deepcopy(pieces)
#         self.velocity = np.zeros((len(pieces), 2))  # 2D velocity for each piece
#         self.position = self.initialize_position()
#         self.pbest_position = self.position.copy()
#         self.pbest_score = float("-inf")
#         self.fitness = 0

#     def initialize_position(self) -> np.ndarray:
#         positions = np.zeros((len(self.pieces), 2))
#         for i, piece in enumerate(self.pieces):
#             x = random.randint(0, self.stock_size[0] - piece["size"][0])
#             y = random.randint(0, self.stock_size[1] - piece["size"][1])
#             positions[i] = [x, y]
#         return positions


# class Policy2350030:
#     def __init__(self):
#         self.num_particles = 30
#         self.w = 0.7  # inertia weight
#         self.c1 = 1.5  # cognitive weight
#         self.c2 = 1.5  # social weight
#         self.max_iter = 50
#         self.particles = []
#         self.gbest_position = None
#         self.gbest_score = float("-inf")

#     def initialize_swarm(self, stock_size: Tuple[int, int], pieces: List[Dict]):
#         self.particles = [
#             Particle(stock_size, pieces) for _ in range(self.num_particles)
#         ]

#     def calculate_fitness(self, particle: Particle) -> float:
#         used_area = 0
#         overlap = 0

#         for i, piece in enumerate(particle.pieces):
#             x, y = particle.position[i]
#             w, h = piece["size"]

#             # Check boundaries
#             if (
#                 x < 0
#                 or y < 0
#                 or x + w > particle.stock_size[0]
#                 or y + h > particle.stock_size[1]
#             ):
#                 return float("-inf")

#             used_area += w * h

#             # Check overlap with other pieces
#             for j, other_piece in enumerate(particle.pieces):
#                 if i != j:
#                     ox, oy = particle.position[j]
#                     ow, oh = other_piece["size"]

#                     if x < ox + ow and x + w > ox and y < oy + oh and y + h > oy:
#                         overlap += 1

#         if overlap > 0:
#             return float("-inf")

#         return used_area / (particle.stock_size[0] * particle.stock_size[1])

#     def update_particle(self, particle: Particle):
#         r1, r2 = random.random(), random.random()

#         # Update velocity
#         particle.velocity = (
#             self.w * particle.velocity
#             + self.c1 * r1 * (particle.pbest_position - particle.position)
#             + self.c2 * r2 * (self.gbest_position - particle.position)
#         )

#         # Update position
#         particle.position += particle.velocity

#         # Clamp positions within bounds
#         for i, piece in enumerate(particle.pieces):
#             particle.position[i][0] = max(
#                 0,
#                 min(particle.position[i][0], particle.stock_size[0] - piece["size"][0]),
#             )
#             particle.position[i][1] = max(
#                 0,
#                 min(particle.position[i][1], particle.stock_size[1] - piece["size"][1]),
#             )

#     def get_action(self, observation, info):
#         if not observation["products"]:
#             return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

#         stock = observation["products"][0]
#         if len(observation["products"]) <= 1:
#             return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

#         # Initialize swarm if first time
#         if not self.particles:
#             self.initialize_swarm((stock[0], stock[1]), observation["products"][1:])

#         # Run PSO iterations
#         for _ in range(self.max_iter):
#             for particle in self.particles:
#                 # Calculate fitness
#                 particle.fitness = self.calculate_fitness(particle)

#                 # Update personal best
#                 if particle.fitness > particle.pbest_score:
#                     particle.pbest_score = particle.fitness
#                     particle.pbest_position = particle.position.copy()

#                 # Update global best
#                 if particle.fitness > self.gbest_score:
#                     self.gbest_score = particle.fitness
#                     self.gbest_position = particle.position.copy()

#             # Update particles
#             for particle in self.particles:
#                 self.update_particle(particle)

#         # Return best position found
#         next_piece = observation["products"][1]
#         if self.gbest_position is not None:
#             pos = self.gbest_position[0].astype(int)
#             return {"stock_idx": 0, "size": next_piece["size"], "position": tuple(pos)}

#         return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


# class DPPolicy:
#     def __init__(self):
#         self.memo = {}
#         self.stock_size = None
#         self.placement_map = {}

#     def solve_dp(
#         self, width: int, height: int, pieces: List[Dict]
#     ) -> Tuple[float, Dict]:
#         # Create state key
#         state = (
#             width,
#             height,
#             tuple(sorted((p["size"][0], p["size"][1]) for p in pieces)),
#         )

#         # Check memoized result
#         if state in self.memo:
#             return self.memo[state]

#         # Base cases
#         if not pieces:
#             return 0, {}
#         if width <= 0 or height <= 0:
#             return float("-inf"), {}

#         best_value = 0
#         best_cut = {}

#         # Try each piece at current position
#         for i, piece in enumerate(pieces):
#             w, h = piece["size"]
#             remaining_pieces = pieces[:i] + pieces[i + 1 :]

#             # Try horizontal cut
#             if w <= width:
#                 # Cut right
#                 right_value, right_cut = self.solve_dp(
#                     width - w, height, remaining_pieces
#                 )
#                 if right_value != float("-inf"):
#                     value = w * h + right_value
#                     if value > best_value:
#                         best_value = value
#                         best_cut = {
#                             "piece": piece,
#                             "position": (0, 0),
#                             "next": right_cut,
#                         }

#             # Try vertical cut
#             if h <= height:
#                 # Cut down
#                 down_value, down_cut = self.solve_dp(
#                     width, height - h, remaining_pieces
#                 )
#                 if down_value != float("-inf"):
#                     value = w * h + down_value
#                     if value > best_value:
#                         best_value = value
#                         best_cut = {
#                             "piece": piece,
#                             "position": (0, 0),
#                             "next": down_cut,
#                         }

#         self.memo[state] = (best_value, best_cut)
#         return best_value, best_cut

#     def reconstruct_solution(self, cut_info: Dict, offset: Tuple[int, int] = (0, 0)):
#         if not cut_info:
#             return []

#         piece = cut_info["piece"]
#         x, y = offset
#         piece["position"] = (x, y)

#         solution = [piece]

#         # Recursively reconstruct next cuts
#         if "next" in cut_info:
#             next_x = x + piece["size"][0] if "right" in cut_info else x
#             next_y = y + piece["size"][1] if "down" in cut_info else y
#             solution.extend(
#                 self.reconstruct_solution(cut_info["next"], (next_x, next_y))
#             )

#         return solution

#     def get_action(self, observation, info):
#         if not observation["products"]:
#             return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

#         stock = observation["products"][0]
#         if len(observation["products"]) <= 1:
#             return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

#         # Clear memoization for new problem
#         self.memo = {}
#         self.stock_size = (stock[0], stock[1])

#         # Solve using DP
#         _, cut_info = self.solve_dp(stock[0], stock[1], observation["products"][1:])

#         # Reconstruct solution
#         solution = self.reconstruct_solution(cut_info)

#         if solution:
#             next_piece = observation["products"][1]
#             for piece in solution:
#                 if piece["size"] == next_piece["size"]:
#                     return {
#                         "stock_idx": 0,
#                         "size": piece["size"],
#                         "position": piece["position"],
#                     }

#         return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
