import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
from Environment import Lake
import pickle
from copy import deepcopy
import argparse
import multiprocessing
from time import time
import math

maps = []
importance_maps = []

# Load the different resolutions maps #
for i in range(4):
    maps.append(np.genfromtxt('map_{}.csv'.format(i+1), delimiter=','))
    importance_maps.append(np.genfromtxt('importance_map_{}.csv'.format(i+1), delimiter=','))

#init_points = np.array([[5, 6], [11, 12], [17, 19], [23, 25]])
init_points = np.array([[11, 12], [15, 14], [15, 3]])

# parser = argparse.ArgumentParser(description='Evolutionary computation of the .')
# parser.add_argument('-R', metavar='R', type=int,
#                     help='Resolution of the map', default=2)
# parser.add_argument('--cxpb', metavar='cxpb', type=float,
#                     help='Cross breed prob.', default=0.7)
# parser.add_argument('--mutpb', metavar='mutpb', type=float,
#                     help='Mut prob.', default=0.3)
# args = parser.parse_args()

# r = args.R
# cxpb = args.cxpb
# mutpb = args.mutpb

r = 2
cxpb = 0.7
mutpb = 0.3

NUM_OF_AGENTS = 1

# Creation of the environment #

print(" ---- OPTIMIZING MAP NUMBER {} ----".format(r))

env = Lake(filepath='map_{}.csv'.format(r),
           number_of_agents= NUM_OF_AGENTS,
           action_type="complete",
           init_pos=init_points, # para el caso multi agente
           #init_pos=init_points[r - 1][np.newaxis],
           importance_map_path='importance_map_{}.csv'.format(r),
           num_of_moves=50)

IND_SIZE = 8  # Number of actions #

# Creation of the multi-objective problem #
# First objective: NHPP
# Second objective: distance
creator.create('FitnessMax', base.Fitness, weights=(1.0,-1.0))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# used in single-AVS, 50 maximum number of steps
def crea_inividuo():
    pasos = random.randint(5, 50)
    return [random.randint(0,7) for i in range(pasos)]

#used in multi-ASV
def crea_inividuo_multiagente():
    pasos = random.randint(5, 50) * NUM_OF_AGENTS
    return [random.randint(0,7) for i in range(pasos)]


# random chromosomes
#toolbox.register("indices", crea_inividuo_multiagente)
toolbox.register("indices", crea_inividuo)


# Generación de inviduos y población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# mutation adapted for messy GA representation
def mutShuffleIndexes(individual, indpb):
    size = len(individual)
    if size > 1:
        for i in range(size):
            if random.random() < indpb:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                individual[i], individual[swap_indx] = \
                    individual[swap_indx], individual[i]

    return individual,


# genetic operators
toolbox.register("mate", tools.cxMessyOnePoint)
toolbox.register("mutate", mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

#%% single objective fitness function, no used in the paper

def evalTrajectory(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""

    # distancia entre el último elemento y el primero
    env.reset()
    R = 0
    if len(individual) < 3: # penalizamos soluciones muy cortas
        return -1,
    for t in range(len(individual)):
        _, reward = env.step([individual[t]])

        R += np.sum(reward)
    
    return R,


#%% single-ASV fitness function

def evalTrajectory_monoagente(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""

    # distancia entre el último elemento y el primero
    env.reset()
    R = 0
    distance = 0
    #print(individual)
    if len(individual) < 3: # penalizamos soluciones muy cortas
        return -1, 100000
    
    tam_individuo = len(individual)
    
    
    if tam_individuo > 50:
        return -1, 100000
    
    pasos_extra = tam_individuo - 50
    
    if pasos_extra > 0:
        R = pasos_extra * -3    
    
    for t in range(len(individual)):
        _, reward = env.step([individual[t]])

        R += np.sum(reward)
        if individual[t] > 3:
            distance += math.sqrt(2)
        else:
            distance += 1

    return R, distance

#%% multi-ASV fitness function

def evalTrajectory_multiagente(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""

    # distancia entre el último elemento y el primero
    env.reset()
    R = 0
    distance = 0
    
    tam_individuo = len(individual)
    
    pasos_extra = tam_individuo - 150
    
    if pasos_extra > 0:
        R = pasos_extra * -3
    
    if tam_individuo < 6: # penalizamos soluciones muy cortas
        return -1, 100000
    
    if tam_individuo > 150:
        return -1, 100000
    
    while len(individual) % 3 != 0: # comprobar divisible entre tres
        individual.pop()
        
    for t in range(0,len(individual), NUM_OF_AGENTS):
        _, reward = env.step(individual[t: t+NUM_OF_AGENTS])

        R += np.sum(reward)
        
        for paso in individual[t: t+NUM_OF_AGENTS]:
            if paso > 3:
                distance += math.sqrt(2)
            else:
                distance += 1
    
    return R, distance

toolbox.register("evaluate", evalTrajectory_monoagente)


#%% not used in the paper

def plot_evolucion(log, r):

    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    plt.plot(gen, fit_mins, "b")
    plt.plot(gen, fit_maxs, "r")
    plt.plot(gen, fit_ave, "--k")
    plt.fill_between(gen, fit_mins, fit_maxs,
                     facecolor="g", alpha=0.2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend(["Min", "Max", "Avg"])
    plt.grid()
    plt.savefig("EvolucionYpacarai_{}.png".format(r), dpi=300)

#%%

def plot_frente():
    """
    Representación del frente de Pareto que hemos obtenido
    """
    #replace .txt with the one to be represented    
    datos_pareto = np.loadtxt("fitnessmonoagente.txt", delimiter=",")
    plt.scatter(datos_pareto[:, 1], datos_pareto[:, 0], s=20)    
    plt.xlabel("Distance")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("Pareto.png", dpi=300, bbox_inches="tight")    

#%% not used in the paper

def main_unico_objetivo(c, m):
    #CXPB, MUTPB, NGEN = cxpb, mutpb, 100 + 50 * (r-1)
    CXPB, MUTPB, NGEN = c, m, 4
    pop = toolbox.population()
    MU, LAMBDA = len(pop), len(pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    t0 = time()
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats=stats,
                                             halloffame=hof)
    print("He tardado {} segundos".format(time()-t0))
    return hof, logbook

#%% multi-objective configuration

def main_multi_objetivo(c, m):
    #CXPB, MUTPB, NGEN = cxpb, mutpb, 100 + 50 * (r-1)
    CXPB, MUTPB, NGEN = c, m, 400
    pop = toolbox.population(2000)
    MU, LAMBDA = len(pop), len(pop)
    pareto = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    t0 = time()
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU,
                                             LAMBDA, CXPB, MUTPB,
                                             NGEN, stats=stats,
                                             halloffame=pareto)
    print("He tardado {} segundos".format(time()-t0))
    return pareto, logbook


if __name__ == "__main__":
    random.seed(0)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    multi = True
    if multi == True:
        pareto_list = list()
        c = [0.8, 0.7, 0.6, 0.5] # crossover probabilities
        m = [0.2, 0.3, 0.4, 0.5] # mutation probabilities
        for popc, popm in zip(c, m):
            pareto, logbook = main_multi_objetivo(popc, popm)
            pareto_list.append(pareto)
        pareto_conjunto = deepcopy(pareto_list[0])
        pareto_conjunto.update(pareto_list[1])
        pareto_conjunto.update(pareto_list[2])
        pareto_conjunto.update(pareto_list[3])
        res_individuos = open("individuosmonoagente_NSGA.txt", "w")
        res_fitness = open("fitnessmonoagente_NSGA.txt", "w")
        for ind in pareto_conjunto:
            res_individuos.write(str(ind))
            res_individuos.write("\n")
            res_fitness.write(str(ind.fitness.values[0]))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[1]))
            res_fitness.write("\n")
        res_fitness.close()
        res_individuos.close()
    else:
        c = [0.8, 0.7, 0.6]
        m = [0.2, 0.3, 0.4]
        for cp, mp in zip(c, m):
            res_individuos = open("individuos_NSGA.txt", "a")
            res_fitness = open("fitness_NSGA.txt", "a")
            for i in range(2):
                hof, logbook = main_unico_objetivo(cp, mp)
                res_individuos.write(str(i)+","+str(cp)+","+str(mp)+",")
                res_individuos.write(str(hof[0]))
                res_individuos.write("\n")
                res_fitness.write(str(i)+","+str(cp)+","+str(mp)+",")
                res_fitness.write(str(hof[0].fitness.values[0]))
                res_fitness.write("\n")
            res_fitness.close()
            res_individuos.close()

