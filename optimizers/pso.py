import numpy as np


class Particle:

    def __init__(self):
        self.position = []  # particle position
        self.velocity = []  # particle velocity
        self.best_position = []  # best position individual
        self.best_fitness = 10  # best fitness individual
        self.fitness = 10  # fitness individual

        # I use 10 because knn returns values between 0-1

    def update_particle(self, position, velocity, fitness):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness

        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position


def pso(k, problem, problem_terminate, X_train, y_train, X_test, y_test):
    """
    Particle Swarm Optimization (PSO) algorithm.

    :return: g_best: the best population a dict contains the best population and its fitness,
    history: a list of the best solutions found at each iteration
    """
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    dim = X_train.shape[1]
    num_particles, max_iter, lb, ub = problem_terminate(dim)
    particles = np.tile(Particle(), num_particles)
    velocities = np.zeros((num_particles, dim))

    # store the best candidate
    g_best = {"position": np.array([]), "fitness": 10}
    history = []

    # Initialize the particles and p_best
    for i in range(num_particles):
        position = np.random.rand(1, dim)

        fitness = problem(position, X_train, y_train, X_test, y_test, k)

        particles[i].update_particle(position, velocities[i, :], fitness)

        if fitness < g_best["fitness"]:
            g_best["position"] = position
            g_best["fitness"] = fitness

    history.append(g_best["fitness"])

    max_iter = (max_iter // num_particles) + 1
    # PSO iterations
    for i in range(max_iter):
        for j in range(num_particles):
            # Update the velocity
            new_velocity = w * particles[j].velocity \
                           + c1 * np.random.rand() * (particles[j].best_position - particles[j].position) \
                           + c2 * np.random.rand() * (g_best["position"] - particles[j].position)
            # Update the position
            new_particle = particles[j].position + new_velocity
            # Ensure the particles stay within the bounds
            new_particle = np.clip(new_particle, 0, 1)

            # Evaluate the cost of the new position
            cost = problem(new_particle, X_train, y_train, X_test, y_test, k)

            # update the jth particle with new particle
            particles[j].update_particle(new_particle, new_velocity, cost)

            # Update the p_best and g_best if necessary
            if cost < g_best["fitness"]:
                g_best["position"] = new_particle
                g_best["fitness"] = cost
        # Keep track of the best solution found at each iteration
        history.append(g_best["fitness"])

    return g_best, history
