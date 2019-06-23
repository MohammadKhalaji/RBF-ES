import numpy as np
from deap import algorithms
import random
from .RBF import RBFNet
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools



# Individual generator
def generateES(icls, scls, imin, imax, smin, smax, circle_dim, rmin, rmax, CIRCLES):
    ind = (np.random.rand(CIRCLES, circle_dim) * (imax-imin)) + imin # circle centers
    ind2 = (np.random.rand(CIRCLES, 1) * (rmax-rmin)) + rmin # circle radiuses
    ind = np.append(ind, ind2, axis=1)
    ind = icls(ind)
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(CIRCLES))
    return ind


def fitness(rbf, problem, individual):
    circle_dim = individual.shape[1] - 1
    V = individual[:, :circle_dim]
    gamma = individual[:, circle_dim:]
    rbf.set_characteristics(V, gamma)
    return rbf.update_network().reshape(1, 1)


def crossover(problem, ind1, ind2):
    if problem == 'classification':
        res1, res2 = tools.cxESBlend(ind1, ind2, alpha=0.1)
    else: 
        res1, res2 = tools.cxOnePoint(ind1, ind2)
    return res1, res2

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


class ES:
    def __init__(self, X, y, MU, LAMBDA, ngen, circles, problem='regression'):
        self.rbf = RBFNet(problem)
        self.X = X
        self.y = y
        self.rbf.set_inputs(X, y)
        self.problem = problem
        self.circle_dim = X.shape[1]
        self.MU = MU
        self.LAMBDA = LAMBDA
        self.ngen = ngen
        self.circles = circles
        if problem == 'classification':
            self.min_value = np.min(X)
            self.max_value = np.max(X)
        else:
            self.min_value = np.min(X)
            self.max_value = np.max(X)

        self.min_strategy = 0.5
        self.max_strategy = 3
        self.min_radius = 0.01
        self.max_radius = 0.25

    def train(self):
        print('start training')
        creator.create("FitnessMin", base.Fitness, weights=[-1])
        creator.create("Individual", np.ndarray, typecode="d", fitness=creator.FitnessMin, strategy=None)
        creator.create("Strategy", np.ndarray, typecode="d")

        toolbox = base.Toolbox()
        toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                         self.min_value, self.max_value, self.min_strategy, self.max_strategy, self.circle_dim, self.min_radius, self.max_radius, self.circles)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover, self.problem)
        toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("evaluate", fitness, self.rbf, self.problem)

        toolbox.decorate("mate", checkStrategy(self.min_strategy))
        toolbox.decorate("mutate", checkStrategy(self.max_strategy))

        random.seed()
        pop = toolbox.population(n=self.MU)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        if self.problem == 'classification':
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=self.MU, lambda_=self.LAMBDA, cxpb=0.7, mutpb=0.3, ngen=self.ngen, stats=stats, halloffame=hof)

        else: 
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=self.MU, lambda_=self.LAMBDA, cxpb=0.7, mutpb=0.3, ngen=self.ngen, stats=stats, halloffame=hof)

        V = hof[0][:, :self.circle_dim]
        gamma = hof[0][:, self.circle_dim:]
        self.rbf.set_characteristics(V, gamma)
        print('best individual\'s error:', self.rbf.update_network())
        return self.rbf





