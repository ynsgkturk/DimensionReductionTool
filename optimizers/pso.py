import numpy as np


class Particle:

    def __init__(self):
        self.position = []             # particle position
        self.velocity = []             # particle velocity
        self.best_position = []        # best position individual
        self.best_fitness = 10         # best fitness individual
        self.fitness = 10              # fitness individual

        # I use 10 because knn returns values between 0-1

    def update_particle(self, position, velocity, fitness):

        self.position = position
        self.velocity = velocity
        self.fitness = fitness

        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position