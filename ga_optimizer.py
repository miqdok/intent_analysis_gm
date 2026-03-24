import random
import copy
import time
import json
import numpy as np


class Individual:
    def __init__(self, search_space, genes=None):
        self.search_space = search_space
        self.genes = genes or self._random_genes()
        self.fitness = None
        self.metrics = None

    def _random_genes(self):
        genes = {}
        for param, values in self.search_space.items():
            genes[param] = random.choice(values)
        return genes

    def __repr__(self):
        return f"Individual(fitness={self.fitness}, genes={self.genes})"


class GeneticOptimizer:
    def __init__(self, search_space, fitness_fn, population_size=20,
                 n_generations=15, crossover_rate=0.8, mutation_rate=0.2,
                 tournament_size=3, elitism_count=2, seed=42):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.seed = seed

        self.history = []
        self.all_evaluations = []
        self.best_individual = None

        random.seed(seed)
        np.random.seed(seed)

    def _init_population(self):
        return [Individual(self.search_space) for _ in range(self.population_size)]

    def _evaluate(self, individual):
        if individual.fitness is not None:
            return
        fitness, metrics = self.fitness_fn(individual.genes)
        individual.fitness = fitness
        individual.metrics = metrics
        self.all_evaluations.append({
            "genes": copy.deepcopy(individual.genes),
            "fitness": fitness,
            "metrics": metrics
        })

    def _tournament_select(self, population):
        candidates = random.sample(population, self.tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1_genes, child2_genes = {}, {}
        params = list(self.search_space.keys())

        for param in params:
            if random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]

        return (Individual(self.search_space, child1_genes),
                Individual(self.search_space, child2_genes))

    def _mutate(self, individual):
        changed = False
        for param, values in self.search_space.items():
            if random.random() < self.mutation_rate:
                new_val = random.choice(values)
                if new_val != individual.genes[param]:
                    individual.genes[param] = new_val
                    changed = True
        if changed:
            individual.fitness = None
            individual.metrics = None

    def run(self):
        start_time = time.time()
        population = self._init_population()

        for ind in population:
            self._evaluate(ind)

        for gen in range(self.n_generations):
            gen_start = time.time()

            population.sort(key=lambda x: x.fitness, reverse=True)
            gen_best = population[0]

            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(gen_best)

            gen_fitnesses = [ind.fitness for ind in population]
            gen_info = {
                "generation": gen + 1,
                "best_fitness": float(gen_best.fitness),
                "min_fitness": float(min(gen_fitnesses)),
                "avg_fitness": float(np.mean(gen_fitnesses)),
                "std_fitness": float(np.std(gen_fitnesses)),
                "best_genes": copy.deepcopy(gen_best.genes),
                "global_best_fitness": float(self.best_individual.fitness),
                "time_seconds": round(time.time() - gen_start, 2)
            }
            self.history.append(gen_info)

            print(f"Gen {gen+1:2d}/{self.n_generations}  "
                  f"best={gen_best.fitness:.4f}  "
                  f"avg={np.mean(gen_fitnesses):.4f}  "
                  f"global_best={self.best_individual.fitness:.4f}")

            new_population = [copy.deepcopy(ind) for ind in population[:self.elitism_count]]

            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            for ind in new_population:
                self._evaluate(ind)

            population = new_population

        population.sort(key=lambda x: x.fitness, reverse=True)
        if population[0].fitness >= self.best_individual.fitness:
            self.best_individual = copy.deepcopy(population[0])

        total_time = time.time() - start_time

        return {
            "best_genes": copy.deepcopy(self.best_individual.genes),
            "best_fitness": self.best_individual.fitness,
            "best_metrics": self.best_individual.metrics,
            "total_evaluations": len(self.all_evaluations),
            "total_time_seconds": round(total_time, 2),
            "history": self.history
        }

    def save_results(self, filepath):
        results = {
            "best_genes": self.best_individual.genes,
            "best_fitness": self.best_individual.fitness,
            "best_metrics": self.best_individual.metrics,
            "total_evaluations": len(self.all_evaluations),
            "ga_params": {
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "tournament_size": self.tournament_size,
                "elitism_count": self.elitism_count,
                "seed": self.seed
            },
            "history": self.history,
            "all_evaluations": self.all_evaluations
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
