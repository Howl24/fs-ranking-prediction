import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from project.utils.metrics import evaluate_metric
import matplotlib.pyplot as plt

def max_acc_fitness_function(dna, X, y, features, model, metric, cv):
    X_selected = X[:, features[dna]]
    if X_selected.shape[1] == 0:
        return 0
    
    y_pred = cross_val_predict(model, X_selected, y, cv=cv)
    acc = evaluate_metric(metric, y, y_pred)
    return acc

def tan_fitness_function(dna, X, y, features, model, metric, cv, acc_weight):
    X_selected = X[:, features[dna]]
    if X_selected.shape[1] == 0:
        return 0
    
    y_pred = cross_val_predict(model, X_selected, y, cv=cv)
    acc = evaluate_metric(metric, y, y_pred)
    
    n_feats = len(features[dna])
    return acc_weight * acc + (1- acc_weight)* (1/n_feats)

class Individual(object):
    def __init__(self, dna):
        self.dna = dna
        self.fitness = self.fitness_function(dna)

    @classmethod
    def Random(cls):
        dna = np.random.choice([True, False], size=cls.dna_size)
        return cls(dna)
    
    def print(self):
        print("G-mean: ", self.fitness)

    @classmethod
    def Crossover(cls, ind1, ind2):
        mask = np.random.choice([True, False], size=cls.dna_size)
        new_dna1 = np.logical_or(np.logical_and(ind1.dna, mask),
                                 np.logical_and(ind2.dna, ~mask))

        new_dna2 = np.logical_or(np.logical_and(ind2.dna, mask),
                                 np.logical_and(ind1.dna, ~mask))

        return Individual(new_dna1), Individual(new_dna2)

    def mutate(self):
        pos = np.random.randint(self.dna_size)
        self.dna[pos] = not self.dna[pos]
        self.fitness = self.fitness_function(self.dna)

    @classmethod
    def Configure(cls, dna_size, fitness_function):
        cls.dna_size = dna_size
        cls.fitness_function = staticmethod(fitness_function)
        
class GeneticAlgorithm(object):

    def __init__(self, population_size=30, mutation_rate=0.5,
                 tournament_size=5, generations=20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.generations = generations
        self.current_generation = 0
        self.ranking = {}

    def random_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(Individual.Random())

    def selection(self):
        """Tournament parent selection."""
        participants = np.random.choice(self.population,
                                        size=self.tournament_size,
                                        replace=False)  # no repeat

        return sorted(participants, key=lambda i: i.fitness, reverse=True)[:2]

    def mutation(self, individual):
        if np.random.choice([True, False], size=1,
                            p=[self.mutation_rate, 1-self.mutation_rate]):
            individual.mutate()
        return individual

    def new_gen_selection(self, new_population):
        all_pop = self.population + new_population
        all_pop = sorted(all_pop, key=lambda i: i.fitness, reverse=True)
        return all_pop[: self.population_size]

    def stop_condition(self):
        return self.current_generation >= self.generations
    
    def _update_ranking(self, population):
        for ind in population:
            # Count True values in dna
            cnt = np.sum(ind.dna)
            if cnt not in self.ranking:
                self.ranking[cnt] = ind
            else:
                self.ranking[cnt] = max(ind, self.ranking[cnt],
                                        key = lambda i: i.fitness)
                
    def run(self, plot=True):
        results_by_gen = []

        self.random_population()
        while not self.stop_condition():
            self.current_generation += 1

            new_population = []
            for i in range(self.population_size//2):
                ind1, ind2 = self.selection()
                offspring1, offspring2 = Individual.Crossover(ind1, ind2)

                self.mutation(offspring1)
                self.mutation(offspring2)

                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = self.new_gen_selection(new_population)
            self._update_ranking(self.population)
            
            results_by_gen.append(self.population[0])

        if plot:
            plt.plot(range(1, self.generations+1),
                     [r.fitness for r in results_by_gen])
            
            
            