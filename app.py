import streamlit as st
import numpy as np
import pandas as pd
import math

# import tkinter as tk
# from tkinter import filedialog
import easygui
import random
import random as rn
from numpy.random import choice as np_choice
import time
from os import listdir
from os.path import isfile, join
import time

st.set_page_config(
    page_title="knapsack problem solver app",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
max_it = 500
max_n = 10
N = 2500
NI = 100
Pc = 0.6
Pm = 0.4
stagnation = True
n_ants = 100
n_best = 10
n_iterations = 10
decay = 0.8
alpha = 1
beta = 2
param1=1000
param2=5
param3=0.9
param4=5
######################### Recuit simulé ########################
# Mise a jour de la temperature
def cool(temprature, coolingFactor):
    return temprature * coolingFactor

# Definir le voisinage d'une solution
def getNeighbour(solution, taille, tab_poids_new, capacity):
    np.random.seed()
    sol = solution.copy()
    i = 0
    # generer un indice aleatoire
    x = np.random.randint(taille)
    # alterer la valeur de la solution correspondante a l'indice de 1 a 0
    if sol[x] == 1:
        sol[x] = 0
    # essayer d'alterer une des valeurs de la solution de 0 a 1
    else:
        capacityRest = capacity - get_poids_total(sol, tab_poids_new)
        listItemCanEnter = []
        for i in range(len(sol)):
            if capacityRest > tab_poids_new[i] and sol[i] == 0:
                listItemCanEnter.append(i)
        if len(listItemCanEnter) != 0:
            ind = np.random.randint(len(listItemCanEnter))
            sol[listItemCanEnter[ind]] = 1
        # essayer d'alterer une des valeurs de la solution de 1 a 0
        else:
            listItemPris = []
            for i in range(len(sol)):
                if sol[i] == 1:
                    listItemPris.append(i)
            if len(listItemPris) != 0:
                ind = np.random.randint(len(listItemPris))
                sol[listItemPris[ind]] = 0
            # essayer d'alterer une des valeurs de la solution de 0 a 1
            capacityRest = capacity - get_poids_total(sol, tab_poids_new)
            listItemCanEnter = []
            for i in range(len(sol)):
                if capacityRest > tab_poids_new[i] and sol[i] == 0:
                    listItemCanEnter.append(i)
            if len(listItemCanEnter) != 0:
                ind = np.random.randint(len(listItemCanEnter))
                sol[listItemCanEnter[ind]] = 1
    return sol

# passage d'une solution a une autre
def getNextState(solution, taille, tab_poids_new, tab_gain_new, capacity, temperature):

    # generer le voisin
    newSolution = getNeighbour(solution, taille, tab_poids_new, capacity)
    # evaluer le voisin
    evalNewSol = eval_solution(newSolution, tab_gain_new)
    # evaluer l'ancienne solution
    evalOldSol = eval_solution(solution, tab_gain_new)
    # calculer delta
    delta = evalNewSol - evalOldSol

    if delta > 0:
        return newSolution  # solution meilleur => accepée
    else:
        x = np.random.rand()  # generer un nombre aleatoire

        # critere d'acceptation de la solution
        if x < math.exp(delta / temperature):
            return newSolution  # verifié => accepter nouvelle solution
        else:
            return solution  # non verifié => garder l'ancienne solution

# evaluation d'une solution obtenue
def eval_solution(solution, tab_gain_new):
    gain_total = 0
    for i in range(len(solution)):
        gain_total = gain_total + solution[i] * tab_gain_new[i]

    return gain_total

# trier par utilité
def trier_objet_utility(items):
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    return items

# le nombre maximal que peut contenir le sac de chaque objet
def get_max_number_item(items, capacity=0):
    tab_number = [capacity // item[0] for item in items]
    return tab_number, sum(tab_number)

# definir une table de gain correspondante a l'écriture binaire d'une solution
def get_tab_gain_new(items_sorted, tab_max_nb):
    tab_gain = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][1]] * tab_max_nb[i]
        tab_gain = tab_gain + tab

    return tab_gain

# definir une table de poids  correspondante a l'ecriture binaire d'une solution
def get_tab_poid_new(items_sorted, tab_max_nb):
    tab_poid = []
    for i in range(len(tab_max_nb)):
        tab = [items_sorted[i][0]] * tab_max_nb[i]
        tab_poid = tab_poid + tab
    return tab_poid

# le poids obtenue par une solution ecrite sous sa forme binaire
def get_poids_total(bsol, tab_poid_new):
    poid_total = 0
    for i in range(len(bsol)):
        poid_total = poid_total + bsol[i] * tab_poid_new[i]
    return poid_total

# convertir une solution en n en une forme binaire
def ntobinary(nsol, max_num_tab):
    bsol = []
    for i in range(len(max_num_tab)):
        for p in range(nsol[i]):
            bsol.append(1)
        for p in range(nsol[i], max_num_tab[i]):
            bsol.append(0)
    return bsol

# generer une solution aleatoire
def gen_random_sol(tab, n, capacity):
    weight = []
    profits = []
    capacityleft = capacity
    sol = []
    # initialiser la solution avec des 0
    for k in range(0, n):
        sol.append(0)
    for i in range(0, n):
        weight.append(tab[i][0])
        profits.append(tab[i][1])
    j = 0
    # TQ capacité max non atteinte
    while j < n and capacityleft > 0:
        # generer un indice aleatoire
        index = np.random.randint(0, n - 1)
        # calculer le nombre maximale d'exemplaires qu'on peut rajouter
        maxQuantity = int(capacityleft / weight[index]) + 1
        if maxQuantity == 0:
            nbItems = 0
        else:  # si maxQuantity>0 generer un nombre aleatoire d'exemplaires inferieurs a maxQuantity
            nbItems = np.random.randint(0, maxQuantity)
        sol[index] = nbItems
        capacityleft = capacityleft - weight[index] * sol[index]
        j = j + 1

    gain_out = 0  # calculer le gain obtenu
    for i in range(n):
        gain_out = gain_out + profits[i] * sol[i]

    return gain_out, capacityleft, sol

# convertir une solution binaire en une solution en n
def binaryToNsolution(solution, tab_max_nb):
    solN = []
    indMin = 0
    for i in range(len(tab_max_nb)):
        indMax = indMin + tab_max_nb[i]
        solN.append(sum(solution[indMin:indMax]))
        indMin = indMax
    return solN

# la fonction principale du recuit simulé
def simulatedAnnealing(
    itemsIn,
    capacity,
    solinit,
    samplingSize,
    temperatureInit,
    coolingFactor,
    endingTemperature,
):
    items = itemsIn.copy()
    for i in range(len(items)):
        items[i].append(solinit[i])
    # trier objets par utilité
    items_sorted = trier_objet_utility(items)
    # reordonner la solution
    solinitsorted = []
    for i in range(len(items_sorted)):
        solinitsorted.append(items_sorted[i][2])
    # recupere le tabeau contenant le nombre max d'exemplaires de chaque objet
    tab_max_nb, taille = get_max_number_item(items_sorted, capacity)
    tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb)
    tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb)
    # convertir la solution en une solution binaire
    solCurrent = ntobinary(solinitsorted, tab_max_nb)
    # evaluer la solution
    evalsol = eval_solution(solCurrent, tab_gain_new)
    # recuperer la temperature initaile
    temperature = temperatureInit
    # initialiser la meilleur solution
    bestSol = solCurrent.copy()
    bestEval = evalsol
    while temperature > endingTemperature:

        for i in range(samplingSize):
            # passage a une nouvelle configuration
            solCurrent = getNextState(
                solCurrent, taille, tab_poids_new, tab_gain_new, capacity, temperature
            )
            # evaluer la nouvelle configuation
            evalCurrent = eval_solution(solCurrent, tab_gain_new)
            # si meilleur MAJ de la meilleur solution
            if evalCurrent > bestEval:
                bestSol = solCurrent.copy()
                bestEval = evalCurrent
        # MAJ la temperature
        temperature = cool(temperature, coolingFactor)

    objects = []
    solution = []
    # convertir la solution binaire trouver en une solution en n
    Nsol = binaryToNsolution(bestSol, tab_max_nb)
    for i, item in enumerate(Nsol):
        if item != 0:
            objects.append(items[i])
            solution.append(item)
    poids = 0
    for i, obj in enumerate(objects):
        poids += obj[0] * solution[i]
    # retourne la solution son gain et son poids
    return objects, solution, Nsol, bestEval, poids

############################################################################################################

######################### Ant colony ########################

class AntColony:
    def __init__(
        self,
        benifices,
        poids,
        utilites,
        n_objets,
        W,
        densitySol,
        n_ants,
        n_best,
        n_iterations,
        decay,
        alpha=1,
        beta=1,
    ):
        """
        Args:
            benifices (1D numpy.array): Les benifices de chaque objet.
            poids (1D numpy.array): Les poids de chaque objet.
            poids (1D numpy.array): L'utilité d'un objet de chaque objet.
            n_objets (int): Nombre d'objets
            W (int): La capacité du sac
            densitySol (liste): Solution generé par heuristique spécifique
            n_ants (int): Nombre de fourmis par iterations
            n_best (int): Nombre de meilleures fourmis qui déposent le pheromone
            n_iteration (int): nombre d'iteration
            decay (float): 1-Taux d'evaporation de pheromone
            alpha (int or float): Exposant dans le pheromone, Alpha grand donne plus de poid au pheromone
            beta (int or float): Exposant sur l'utilité, Beta grand donne plus de poid a l'utilité
        Example:
            ant_colony = AntColony(benifices,poids, utilites,n_objets, W,densitySol,n_ants, n_best, n_iterations, decay, alpha=1, beta=2)
        """
        self.utilites = utilites
        self.W = W
        self.n_objets = n_objets
        self.poids = poids
        self.benifices = benifices
        self.pheromone = np.ones(n_objets)
        # ajouter du pheromone au objets generé par heuristique
        for i, s in enumerate(densitySol):
            if s > 0:
                self.pheromone[i] += s * 0.1
        self.all_inds = range(len(utilites))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self, n_candidats, densitySol):
        """
        Args:
            n_candidats (int): Nombre de candidats pour construire les solutions
            densitySol (Gain:int,sol:liste,Poid:int): Solution generé par heuristique spécifique
        Example:
            best_sol = ant_colony.run(n_candidats,densitySol)
        """
        best_solution = (densitySol[1], densitySol[0], densitySol[2])
        best_solution_all_time = (densitySol[1], densitySol[0], densitySol[2])
        for i in range(self.n_iterations):
            # generer toutes les solutions par les fourmis
            all_solutions = self.gen_all_solutions(n_candidats)
            # mise a jours des pistes pheromones
            self.spread_pheronome(
                all_solutions, self.n_best, best_solution=best_solution
            )
            # Choisir meilleure solution dans l'iteration actuelle
            best_solution = max(all_solutions, key=lambda x: x[1])
            # print (best_solution)
            # Mettre a jour la meilleure solution globale
            if best_solution[1] > best_solution_all_time[1]:
                best_solution_all_time = best_solution

            # evaporation de pheromone
            self.pheromone = self.pheromone * self.decay
            self.pheromone[self.pheromone < 1] = 1

        # print(self.gen_sol_gain(best_solution_all_time[0]))
        # print(self.gen_path_poid(best_solution_all_time[0]))
        return best_solution_all_time

    def spread_pheronome(self, solutions, n_best, best_solution):
        """
        Dépose le pheromone sur les n_best meilleures solutions

        """
        sorted_solution = sorted(solutions, key=lambda x: x[1], reverse=True)
        for sol, gain, poid in sorted_solution[:n_best]:
            for i in sol:
                self.pheromone += 0.00001 * gain

    def gen_sol_gain(self, sol):
        """
        Calcul le gain d'une solution.
        Pas necessaire mais peut servir à verifier les résultats (test unitaire)

        """
        total_fitness = 0
        for i, ele in enumerate(sol):
            total_fitness += self.benifices[i] * ele
        return total_fitness

    def gen_path_poid(self, sol):
        """
        Calcul le poid d'une solutions.
        Pas necessaire mais peut servir à verifier les résultats (test unitaire)

        """
        total_fitness = 0
        for i, ele in enumerate(sol):
            total_fitness += self.poids[i] * ele
        return total_fitness

    def gen_all_solutions(self, n_candidats):
        """
        Generer toutes les solutions par les fourmis

        """
        all_solutions = []
        for i in range(self.n_ants):
            # Positionner la fourmis sur un objets de départ aleatoirement
            n = rn.randint(0, self.n_objets - 1)
            # generation de la solution par la fourmis en utilisant n_candidats
            solution = self.gen_sol(n, n_candidats)

            # ajouter la solution a la liste de toute les solutions
            all_solutions.append((solution[0], solution[1], solution[2]))
        return all_solutions

    def listeCandidate(self, phero, visited, n_candidats):
        """
        retourne La liste des candidats pour une solution

        """
        pheromone = phero.copy()

        pheromone[list(visited)] = 0
        # rn.choices returns a list with the randomly selected element from the list.
        # weights to affect a probability for each element

        c = rn.choices(self.all_inds, weights=[p for p in pheromone], k=n_candidats)
        c = list(set(c))
        i = len(c)
        """while i<n_candidats:
          n=rn.randint(0,self.n_objets-1)
          if n not in visited:
            c.append(n)
            i+=1"""
        nb_candidats = len(c)

        return c, pheromone

    # generer solution c'est bon
    def gen_sol(self, start, n_candidats):
        """
        Construit la solution avec n_candidats et start comme premier objet

        """
        sol = np.zeros(self.n_objets)
        poidrestant = self.W
        visited = set()  # liste des objets visité
        # ajouter le premier objet
        r = rn.randint(1, poidrestant // self.poids[start])
        sol[start] = r
        poidrestant -= self.poids[start] * r
        gain = r * self.benifices[start]
        visited.add(start)  # ajouter le debut a la liste civité

        # la liste candidates avec les pheromones mis a jours localement (0 sur les visited)
        candidats, pheromones = self.listeCandidate(
            self.pheromone, visited, n_candidats
        )

        for i in candidats:
            # Choisir le prochain objets parmi les candidats ainsi que le nombre
            move, nb = self.pick_move(
                pheromones, candidats, n_candidats, self.utilites, visited
            )
            toPop = candidats.index(move)
            candidats.pop(toPop)
            n_candidats -= 1
            np.delete(
                pheromones, toPop
            )  # rendre le pheromone à 0 pour indiquer qu'il a été visité

            # Mise a jour poidRestant et gain de la solution
            poidrestant -= self.poids[move] * nb
            while poidrestant < 0:
                nb -= 1
                poidrestant += self.poids[move]

            sol[move] = nb
            gain += nb * self.benifices[move]

            # ajouter l'objet a visited
            visited.add(move)
        # print("s",i,sol,gain)
        return sol, gain, self.W - poidrestant

    def pick_move(self, pheromone, liste_cand, n_candidats, utilite, visited):

        pheromone = pheromone.copy()[liste_cand]
        # generer le regle de déplacement sur les candidat
        numerateurs = (pheromone ** self.alpha) * (
            (1.0 / (utilite[liste_cand])) ** self.beta
        )

        # formule vu en cours

        P = numerateurs / numerateurs.sum()

        # choisir l'objet suivant en utilisant les probabilité P
        move = np_choice(liste_cand, 1, p=P)[0]
        # nombre d'objet a prendre
        nb = self.W // self.poids[move]
        # nb=rn.randint(0,self.W//self.poids[move])

        return (move, nb)


def density_ordered_greedy_ukp(b, v, w):
    d = [(b[i] / v[i], i) for i in range(len(v))]
    d.sort(key=lambda x: x[0], reverse=True)
    M = 0
    res = [0 for _ in range(len(d))]
    for i in range(len(d)):
        if w == 0:
            break
        nb = int(w / v[d[i][1]])
        M += nb * b[d[i][1]]
        w -= nb * v[d[i][1]]
        res[d[i][1]] = nb
    return M, res, w


############################################################################################################


######################### Heuristic By Rounding ########################


def ratios(b, v, V_t):
    ratios = [(b[i] / v[i], i) for i in range(len(b))]
    ratios.sort(reverse=True)
    return ratios


def intermediate_solution(b, v, V_t):
    x = [0 for i in range(len(b))]
    ratio = ratios(b, v, V_t)
    cap_act = 0
    capacite = V_t
    cap = cap_act
    i = 0
    benefice = 0
    while capacite >= cap_act + v[ratio[i][1]]:
        if i == len(b):
            i = 0
        benefice += b[ratio[i][1]]
        cap_act += v[ratio[i][1]]
        x[ratio[i][1]] = x[ratio[i][1]] + 1
        i += 1
    return x, cap_act, benefice, i


def heuristic_arrondi(b, v, V_t):
    x, capacite, benefice, i_max = intermediate_solution(b, v, V_t)
    ratio = ratios(b, v, V_t)
    difference = V_t - capacite
    difference_min = 100000000
    i = 0
    while difference_min != difference:
        if i == i_max:
            i = 0
        if x[ratio[i][1]] == 0:
            i += 1
        else:
            if capacite + b[ratio[i][1]] <= V_t:
                benefice += b[ratio[i][1]]
                capacite += v[ratio[i][1]]
                x[ratio[i][1]] = x[ratio[i][1]] + 1
            i += 1
            difference = V_t - capacite
            difference_min = min(difference, difference_min)
    return capacite, x, benefice


############################################################################################################

######################### AG ########################


class chromosome:
    def __init__(self, gain, poids, solution):
        self.gain = gain
        self.poids = poids
        self.solution = solution


def generation_population(n, w, b, v, N):
    population = []
    gain, sol, poids = density_ordered_greedy_ukp(b, v, w)
    population.append(chromosome(gain, w - poids, sol))
    for _ in range(N - 1):
        sol = []
        gain = 0
        poids = w
        for i in range(n):
            nb = random.randint(0, int(poids / v[i]))
            gain += nb * b[i]
            poids -= nb * v[i]
            sol.append(nb)
        population.append(chromosome(gain, w - poids, sol))
    return population


def recherche_dico(liste, v):
    i = 0
    j = len(liste) - 1
    while i <= j:
        m = (i + j) // 2
        if liste[m] == v:
            return m
        elif liste[m] < v:
            i = m + 1
        else:
            j = m - 1
    return i


def selection_elitiste(population, NI):
    pool = []
    population.sort(key=lambda x: x.gain, reverse=True)
    for i in range(min(len(population), NI)):
        pool.append(population[i])
    return pool


def selection_roue_loterie(population, NI):
    pop = list(population)
    pool = []
    for _ in range(min(len(pop), NI)):
        gain_t = 0
        probas = []
        for chromosome in pop:
            gain_t += chromosome.gain
            probas.append(gain_t)
        probas = np.array(probas) / gain_t
        rand = random.uniform(0, 1)
        ind = recherche_dico(probas, rand)
        pool.append(pop[ind])
        pop.pop(ind)
    return pool


def corriger(liste, n, w, b, v):
    poids = 0
    gain = 0
    for i in range(n):
        poids += liste[i] * v[i]
        gain += liste[i] * b[i]
    while poids > w:
        r = random.randint(0, n - 1)
        if liste[r] > 0:
            liste[r] -= 1
            poids -= v[r]
            gain -= b[r]
    return chromosome(gain, poids, liste)


def croisement_1_point(parent1, parent2, n, w, b, v):
    k = random.randint(1, n - 1)
    enfant1 = corriger(parent1.solution[0:k] + parent2.solution[k::], n, w, b, v)
    enfant2 = corriger(parent2.solution[0:k] + parent1.solution[k::], n, w, b, v)
    return enfant1, enfant2


def croisement(pool, Pc, n, w, b, v):
    selected = []
    for chromosome in pool:
        r = random.uniform(0, 1)
        if r <= Pc:
            selected.append(chromosome)
    enfants = []
    for i in range(0, len(selected) - 1, 2):
        enfant1, enfant2 = croisement_1_point(selected[i], selected[i + 1], n, w, b, v)
        enfants.append(enfant1)
        enfants.append(enfant2)
    return enfants


def mutation(pool, Pm, n, w, b, v):
    enfants = []
    for chromosome in pool:
        r = random.uniform(0, 1)
        if r <= Pm:
            liste = list(chromosome.solution)
            r = random.randint(0, n - 1)
            val_s = str(bin(liste[r]))
            rand = random.randint(2, len(val_s) - 1)
            if val_s[rand] == "0":
                val_s = val_s[0:rand] + "1" + val_s[rand + 1 :]
            else:
                val_s = val_s[0:rand] + "0" + val_s[rand + 1 :]
            liste[r] = int(val_s, 2)
            enfants.append(corriger(liste, n, w, b, v))
    return enfants


def maj_population(population, enfants1, enfants2, N):
    population = population + enfants1 + enfants2
    population.sort(key=lambda x: x.gain, reverse=True)
    return population[0:N]


def arreter(t, max_it, mem, max_n, stagnation):
    if t == max_it:
        return True
    if stagnation:
        if len(mem) == max_n:
            val = mem[0]
            for elt in mem:
                if elt != val:
                    return False
            return True
    return False


def AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation):
    population = generation_population(n, w, b, v, N)
    t = 1
    mem = []
    while not arreter(t, max_it, mem, max_n, stagnation):
        pool = selection_elitiste(population, NI)
        enfants1 = croisement(pool, Pc, n, w, b, v)
        enfants2 = mutation(pool, Pm, n, w, b, v)
        population = maj_population(population, enfants1, enfants2, N)
        if stagnation:
            if len(mem) == max_n:
                mem.pop(0)
            mem.append(population[0].gain)
        t += 1
    return population[0].gain, population[0].poids, population[0].solution


############################################################################################################

################## Branch and bound #################
class noeud:
    def __init__(self, objet, val, pere, poids, m):
        self.objet = objet
        self.val = val
        self.pere = pere
        self.poids = poids
        self.m = m


def first_solution(d, b, v, w, racine):
    M = 0
    n = racine
    for i in range(len(d)):
        if w == 0:
            break
        nb = int(w / v[d[i][1]])
        M += nb * b[d[i][1]]
        w -= nb * v[d[i][1]]
        n = noeud(i + 1, nb, n, w, M)
    return M, n


def diviser(n, b, v, d):
    ind = d[n.objet][1]
    nb = int(n.poids / v[ind])
    return [
        noeud(n.objet + 1, i, n, n.poids - i * v[ind], n.m + i * b[ind])
        for i in range(nb, -1, -1)
    ]


def evaluer(n, d):
    if n.objet == len(d):
        return n.m
    return n.m + n.poids * d[n.objet][0]


def branch_and_bound_ukp(b, v, w):
    racine = noeud(0, -1, None, w, 0)
    d = [(b[i] / v[i], i) for i in range(len(v))]  # Densité
    d.sort(key=lambda x: x[0], reverse=True)
    mini = v[d[-1][1]]
    min_w = []
    for i in range(len(d) - 1, -1, -1):
        mini = min(mini, v[d[i][1]])
        min_w.insert(0, mini)
    M, res = first_solution(d, b, v, w, racine)
    na = diviser(racine, b, v, d)
    while len(na) != 0:
        n = na.pop(0)
        if n.poids < min_w[n.objet - 1]:
            if n.m > M:
                M = n.m
                res = n
        elif int(evaluer(n, d)) > M:
            fils = diviser(n, b, v, d)
            fils_retenus = []
            for f in fils:
                evaluation = evaluer(f, d)
                if int(evaluation) > M:
                    if f.objet == len(d):
                        M = evaluation
                        res = f
                    else:
                        fils_retenus.append(f)
            na = fils_retenus + na
    sol = [0 for _ in range(len(b))]
    M = res.m
    while res.val != -1:
        sol[d[res.objet - 1][1]] = res.val
        res = res.pere
    return M, sol


#######################################################


################# Dynamic programing  #################
def dp_ukp(w, n, b, v):
    # k contient le gain maximal associé aux poids allant de 0 à w (poids max)
    k = [0 for i in range(w + 1)]

    # items contient la liste des objets choisis pour obtenir le gain maximal associé aux poids allant de 0 à w
    items = [[] for i in range(w + 1)]

    for wi in range(w + 1):
        for j in range(n):
            if v[j] <= wi:  # si le poids de l'objet est inférieur au poids considéré
                tmp = k[
                    wi
                ]  # tmp sera utilise pour savoir si k[wi] a été modifié (pour modifier items en conséquences)
                k[wi] = max(k[wi], k[wi - v[j]] + b[j])
                if (
                    k[wi] > tmp
                ):  # si k[wi] a changé (donc on a trouvé une val supérieur à la valeur précedente sauvgardée dans tmp), on met à jour les objets pris
                    items[wi] = []
                    for l in range(len(items[wi - v[j]])):
                        items[wi].append(items[wi - v[j]][l])
                    items[wi].append(j + 1)
                    # donc la liste des objets pris est la liste des objets de items[wi-wt[j] en plus de l'objet j ajouté
    return k[w], items[w]


#######################################################

################ Density Ordered Greedy ###############


def density_ordered_greedy_ukp(b, v, w):
    d = [(b[i] / v[i], i) for i in range(len(v))]
    d.sort(key=lambda x: x[0], reverse=True)
    M = 0
    res = [0 for _ in range(len(d))]
    for i in range(len(d)):
        if w == 0:
            break
        nb = int(w / v[d[i][1]])
        M += nb * b[d[i][1]]
        w -= nb * v[d[i][1]]
        res[d[i][1]] = nb
    return M, res, w


#######################################################

############### Weighted Ordered Greedy ###############


def weighted_ordered_greedy_ukp(b, v, w, ordre_croissant=False):
    d = [(v[i], i) for i in range(len(v))]
    d.sort(key=lambda x: x[0], reverse=not ordre_croissant)
    M = 0
    res = [0 for _ in range(len(d))]
    for i in range(len(d)):
        if w == 0:
            break
        nb = int(w / v[d[i][1]])
        M += nb * b[d[i][1]]
        w -= nb * v[d[i][1]]
        res[d[i][1]] = nb
    return M, res, w


#######################################################

################## Reading intances ###################


def read_data_3_single(path):  # Ex : instance = 10, 15, 20, 25, ..., 205.
    file = open(path)
    v, b = [], []
    n = int(file.readline().strip())
    w = int(file.readline().strip())

    for _ in range(n):
        line = file.readline().strip().split()
        v.append(int(line[0]))
        b.append(int(line[1]))
    return n, w, v, b


def dir_files(dir_path):
    return [dir_path + "/" + f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]


#######################################################

################ Statistics functions #################


def stats(dir_path, meth):
    nbObj = []
    time_bb = []
    gain = []

    files = dir_files(dir_path)
    for file in files:

        n, w, v, b = read_data_3_single(file)
        nbObj.append(n)
        #############################################
        if meth == "rs":
            start_time = time.time()
            data= pd.DataFrame(list(zip(v,b)))
            items_init=data.values.tolist()
            sol = density_ordered_greedy_ukp(b, v, w)[1]
            gain_out, capacityleft, sol = gen_random_sol(items_init, n, w)
            objects, solution, Nsol, bestEval, poids = simulatedAnnealing(
                items_init, w, sol, param1, param2, param3, param4
            )
            gain.append(bestEval)
        if meth == "bb":
            start_time = time.time()
            gain.append(branch_and_bound_ukp(b, v, w)[0])

        elif meth == "dp":
            start_time = time.time()
            gain.append(dp_ukp(w, n, b, v)[0])
        elif meth == "dog":
            start_time = time.time()
            gain.append(density_ordered_greedy_ukp(b, v, w)[0])
        elif meth == "wdogT":
            start_time = time.time()
            gain.append(weighted_ordered_greedy_ukp(b, v, w, True)[0])
        elif meth == "wdogF":
            start_time = time.time()
            gain.append(weighted_ordered_greedy_ukp(b, v, w, False)[0])
        elif meth == "ag":
            start_time = time.time()
            gain.append(AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation)[0])
        elif meth == "hr":
            start_time = time.time()
            gain.append(heuristic_arrondi(b, v, w)[2])
        elif meth == "ac":
            start_time = time.time()
            densitySol = density_ordered_greedy_ukp(b, v, w)
            benifices = np.array(b)
            poids = np.array(v)
            utilites = poids / benifices         

            colony = AntColony(
                benifices,
                poids,
                utilites,
                n,
                w,
                densitySol[1],
                n_ants,
                n_best,
                n_iterations,
                decay,
                alpha=1,
                beta=1,
            )
            gain.append(colony.run(30, densitySol)[1])
            # gain_bb.append()
        time_bb.append(time.time() - start_time)
        #############################################

    df = pd.DataFrame(
        {
            "ojt": nbObj,
            "Time": time_bb,
        }
    )

    df = df.rename(columns={"ojt": "index"}).set_index("index")

    chart = st.line_chart(df)
    st.text("")
    st.subheader("Gain results")
    st.text("")
    df1 = pd.DataFrame(
        {
            "ojt": nbObj,
            "Gain": gain,
        }
    )

    df1 = df1.rename(columns={"ojt": "index"}).set_index("index")

    chart = st.line_chart(df1)


##############################################################################################################


##############################################################################################################
def statsComp(
    dir_path,
    bbcheck,
    dpcheck,
    dogcheck,
    wdogTcheck,
    wdogFcheck,
    hrcheck,
    agcheck,
    accheck,
    rscheck,
):
    nbObj = []
    (
        time_bb,
        time_dp,
        time_dog,
        time_wdogT,
        time_wdogF,
        time_hr,
        time_ag,
        time_ac,
        time_rs,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )

    (
        gain_bb,
        gain_dp,
        gain_dog,
        gain_wdogT,
        gain_wdogF,
        gain_hr,
        gain_ag,
        gain_ac,
        gain_rs,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )
    files = dir_files(dir_path)
    for file in files:

        n, w, v, b = read_data_3_single(file)
        nbObj.append(n)
        #############################################
        if rscheck:
            start_time = time.time()
            data= pd.DataFrame(list(zip(v,b)))
            items_init=data.values.tolist()
            sol = density_ordered_greedy_ukp(b, v, w)[1]
            gain_out, capacityleft, sol = gen_random_sol(items_init, n, w)
            objects, solution, Nsol, bestEval, poids = simulatedAnnealing(
                items_init, w, sol, param1, param2, param3, param4
            )
            time_rs.append(time.time() - start_time)
            gain_rs.append(bestEval)
        if bbcheck:
            start_time = time.time()
            gain = branch_and_bound_ukp(b, v, w)[0]
            time_bb.append(time.time() - start_time)
            gain_bb.append(gain)
        if dpcheck:
            start_time = time.time()
            gain = dp_ukp(w, n, b, v)[0]
            time_dp.append(time.time() - start_time)
            gain_dp.append(gain)
        if dogcheck:
            start_time = time.time()
            gain = density_ordered_greedy_ukp(b, v, w)[0]
            time_dog.append(time.time() - start_time)
            gain_dog.append(gain)
        if wdogTcheck:
            start_time = time.time()
            gain = weighted_ordered_greedy_ukp(b, v, w, True)[0]
            time_wdogT.append(time.time() - start_time)
            gain_wdogT.append(gain)
        if wdogFcheck:
            start_time = time.time()
            gain = weighted_ordered_greedy_ukp(b, v, w, False)[0]
            time_wdogF.append(time.time() - start_time)
            gain_wdogF.append(gain)
        if hrcheck:
            start_time = time.time()
            gain = heuristic_arrondi(b, v, w)[2]
            time_hr.append(time.time() - start_time)
            gain_hr.append(gain)
        if agcheck:
            random.seed(1)
            start_time = time.time()
            gain = AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation)[0]
            time_ag.append(time.time() - start_time)
            gain_ag.append(gain)
        if accheck:
            start_time = time.time()
            densitySol = density_ordered_greedy_ukp(b, v, w)
            benifices = np.array(b)
            poids = np.array(v)
            utilites = poids / benifices

            colony = AntColony(
                benifices,
                poids,
                utilites,
                n,
                w,
                densitySol[1],
                n_ants,
                n_best,
                n_iterations,
                decay,
                alpha=1,
                beta=1,
            )
            gain = colony.run(30, densitySol)[1]
            time_ac.append(time.time() - start_time)
            gain_ac.append(gain)

        #############################################
    df = pd.DataFrame(
        {
            "ojt": nbObj,
        }
    )
    if bbcheck:
        df["Branch and Bound"] = time_bb
    if dpcheck:
        df["Dynamic Programing"] = time_dp
    if dogcheck:
        df["Density Ordered Greedy"] = time_dog
    if wdogTcheck:
        df["Weighted Ordered Greedy (asc)"] = time_wdogT
    if wdogFcheck:
        df["Weighted Ordered Greedy (desc)"] = time_wdogF
    if hrcheck:
        df["Heuristic By Rounding"] = time_hr
    if agcheck:
        df["Genetic Algorithm"] = time_ag
    if accheck:
        df["Ant colony"] = time_ac
    if rscheck:
        df["Recuit simulé"] = time_rs

    df = df.rename(columns={"ojt": "index"}).set_index("index")

    chart = st.line_chart(df)
    st.text("")
    st.text("")
    st.subheader("Gain comparaison")
    st.text("")
    st.text("")
    df1 = pd.DataFrame(
        {
            "ojt": nbObj,
        }
    )
    if bbcheck:
        df1["Branch and Bound"] = gain_bb
    if dpcheck:
        df1["Dynamic Programing"] = gain_dp
    if dogcheck:
        df1["Density Ordered Greedy"] = gain_dog
    if wdogTcheck:
        df1["Weighted Ordered Greedy (asc)"] = gain_wdogT
    if wdogFcheck:
        df1["Weighted Ordered Greedy (desc)"] = gain_wdogF
    if hrcheck:
        df1["Heuristic By Rounding"] = gain_hr
    if agcheck:
        df1["Genetic Algorithm"] = gain_ag
    if accheck:
        df1["Ant colony"] = gain_ac
    if rscheck:
        df1["Recuit simulé"] = gain_rs

    df1 = df1.rename(columns={"ojt": "index"}).set_index("index")
    chart1 = st.line_chart(df1)


#######################################################

#############################################


############ UI ############
def main():

    page = st.sidebar.selectbox(
        "Choose a algorithm to run :",
        [
            "Home",
            "Branch and Bound",
            "Dynamic Programing",
            "Density Ordered Greedy",
            "Weighted Ordered Greedy",
            "Heuristic By Rounding",
            "Recuit simulé",
            "Genetic Algorithm",
            "Ant colony",
            "Comparaison",
        ],
    )

    if page == "Home":
        st.title("Welcome to knapsack problem solver ")
        st.write(
            " The knapsack problem is a problem in combinatorial optimization: Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so  that the total weight is less than or equal to a given limit and the total value is as large as possible. It derives its name from the problem faced by someone who is constrained by a fixed-size knapsack and must fill it with the most valuable items. The problem often arises in resource allocation where the decision makers have to choose from a set of non-divisible projects or tasks under a fixed budget or time constraint, respectively"
        )
        st.title("Description")
        st.write("The most common problem being solved is the 0-1 knapsack problem, which restricts the number xi of copies of each kind of item to zero or one. Given a set of n items numbered from 1 up to n, each with a weight wi and a value vi, along with a maximum weight capacity W,")
        st.latex("maximize \: {\displaystyle \sum _{i=1}^{n}v_{i}x_{i}}")
        st.latex("subject \; \: to  \: {\displaystyle \sum _{i=1}^{n}w_{i}x_{i}\leq W}  \: and  \: {\displaystyle x_{i}\in \{0,1\}}.")
        st.write("Here xi represents the number of instances of item i to include in the knapsack. Informally, the problem is to maximize the sum of the values of the items in the knapsack so that the sum of the weights is less than or equal to the knapsack's capacity.")
        st.write("The bounded knapsack problem (BKP) removes the restriction that there is only one of each item, but restricts the number xi of copies of each kind of item to a maximum non-negative integer value c:")
        st.latex("maximize \: {\displaystyle \sum _{i=1}^{n}v_{i}x_{i}}")
        st.latex("subject \; \: to  \:{\displaystyle \sum _{i=1}^{n}w_{i}x_{i}\leq W}  \: and  \: {\displaystyle x_{i}\in \{0,1,2,...,c\}}.")
        st.write("The unbounded knapsack problem (UKP) places no upper bound on the number of copies of each kind of item and can be formulated as above except for that the only restriction on xi is that it is a non-negative integer.")
        st.latex("maximize \: {\displaystyle \sum _{i=1}^{n}v_{i}x_{i}}")
        st.latex("subject \; \: to  \: {\displaystyle \sum _{i=1}^{n}w_{i}x_{i}\leq W} \; and \; {\displaystyle x_{i}\geq 0,\ x_{i}\in \mathbb {Z} .}")
        st.title("Methodes")
       
        st.write("\t \t - Branch and Bound")
        st.write("\t \t - Dynamic Programing")
        st.write("\t \t - Density Ordered Greedy")
        st.write("\t \t - Weighted Ordered Greedy")
        st.write("\t \t - Heuristic By Rounding")
        st.write("\t \t - Recuit simulé")
        st.write("\t \t - Genetic Algorithm")
        st.write("\t \t - Ant colony")
            
   
       


    elif page == "Branch and Bound":
        st.title("Branch and Bound Algorithm")
        st.subheader("Description")
        st.write(
            "B&B is an algorithm that intelligently lists all possible solutions. In practice, only potentially good quality solutions will be listed, solutions that cannot lead to improvements to the current solution are not explored"
        )
        st.subheader("Algorithm")

        st.code(
            """
           class noeud:
                def __init__(self, objet, val, pere, poids, m):
                    #objet : ID de l'objet et qui correspond aussi au niveau du nœud dans l'arborescence
                    self.objet = objet
                    #val : nombre d'exemplaires pris pour cet objet 
                    self.val = val
                    #pere : pointeur vers le nœud père. Lorsqu'on trouve une solution, on remonte 
                    #en utilisant cet attribut jusqu'à arriver à la racine
                    self.pere = pere
                    #poids : c'est le volume restant dans le sac à dos après l'ajout des "val" exemplaires
                    #de l'objet "objet"
                    self.poids = poids
                    #m : c'est le gain cumulé des objets dans le sac à dos jusqu'à cet objet
                    self.m = m

            def first_solution(d,b,v,w,racine):
                #La première solution est trouvée grace à l'heuristique : Density Ordered Greedy
                #La variable "M" stockera le gain de la solution proposée par cette heuristique
                M = 0
                n = racine
                #On fait un parcours complet des objets
                for i in range(len(d)):
                    #Si la taille restante du sac à dos est égale à 0, on arrete le parcours
                    if w==0:
                        break
                    #Sinon, s'il reste de l'espace dans le sac à dos : 
                    #On calcule dans "nb" le nombre d'exemplaires possibles de l'objet d[i][1]
                    nb = int(w/v[d[i][1]])
                    #On met à jour le gain "M" après l'ajout de "nb" exemplaires de l'objet d[i][1]
                    M += nb * b[d[i][1]]
                    #On met à jour le poids restant dans le sac à dos "w"
                    w -= nb * v[d[i][1]]
                    #On construit notre branche au fur à mesure, elle correspond à la branche la plus 
                    #à droite dans l'arbre du branch and bound
                    n = noeud(i+1,nb,n,w,M)
                #On retourne le gain du sac à dos et le nœud feuille qui nous permet de trouver tous
                #les nœuds de la branche à la fin
                return M,n

            def diviser(n,b,v,d):
                #Trouver l'ID de l'objet qui suit l'objet du nœud "n"
                ind = d[n.objet][1]
                #Etant donné un nœud "n", on calcule dans "nb" combien d'exemplaires de l'objet suivant 
                #sont possibles à mettre dans le sac à dos
                nb = int(n.poids/v[ind])
                #On retourne une liste de "nb+1" nœuds, le premier ajoute au sac à dos "nb" exemplaires
                #le deuxième "nb-1" exemplaires et ainsi de suite jusqu'à le dernier qui n'ajoute rien
                return [noeud(n.objet+1,i,n,n.poids-i*v[ind],n.m+i*b[ind]) for i in range(nb,-1,-1)]

            def evaluer(n,d):
                if n.objet==len(d):
                    #Si le nœud correspond à une feuille donc on retourne la solution exacte
                    #qui se trouve dans l'attribut "m"
                    return n.m
                #Sinon, on utilise la meme fonction d'évaluation du cours
                return n.m+n.poids*d[n.objet][0]

            def branch_and_bound_ukp(b, v, w):
                #Initialisation du nœud racine
                racine = noeud(0,-1,None,w,0)
                #Construction de la liste des densités (utilités des objets)
                d = [(b[i]/v[i],i) for i in range(len(v))]
                #Ordonner la liste des densités par ordre décroissant
                d.sort(key=lambda x:x[0], reverse=True)
                ############################
                #Ce bloc permet de retourner une liste "min_w" tel que min_w[i] est le volume minimal 
                #des objets de la sous liste d[i :: nb_objets]. Cette liste va nous servir dans 
                #l'optimisation et l'anticipation de l'élagage dans certains cas
                mini = v[d[-1][1]]
                min_w = []
                for i in range(len(d)-1,-1,-1):
                    mini = min(mini, v[d[i][1]])
                    min_w.insert(0, mini)
                ############################
                #Trouver une première solution. "M" c'est la borne inférieure et "res" c'est
                #la solution actuelle
                M,res = first_solution(d,b,v,w,racine)
                #Initialisation de la liste "na", liste des nœuds actifs, avec le résulat de
                #la séparation du nœud racine
                na = diviser(racine,b,v,d)
                #Tant qu'il y aura de nœuds actifs, on termine notre exploration de l'arbre
                while len(na)!=0:
                    #On récupère le premier nœud actif dans "n" (on prend le premier nœud pour 
                    #assurer une exploration en profondeur de l'arbre)  
                    n = na.pop(0)
                    #S'il y a aucun autre objet qui peut tenir dans le volume restant dans le 
                    #sac à dos, alors on élage et on modifie la borne inférieure si notre branche
                    #donne un résultat meilleur (on remarque l'utilisation de la liste "min_w" pour 
                    #optimiser la vérification)
                    if n.poids<min_w[n.objet-1]:
                        if n.m>M:
                            M = n.m
                            res = n
                    #Dans le cas contraire, si l'évaluation de notre nœud donne une valeure inférieure
                    #à "M" on élague. Sinon, on sépare le nœud et pour chaque nœud fils "f", si son 
                    #évaluation peut améliorer la solution on le garde sinon on élague.
                    #Si le nœud fils "f" est une feuille et en plus il améliore la solution courante, 
                    #on le prend comme nouvelle solution.
                    #PS : on remarque l'utilisation de la fonction python int() après chaque évaluation,
                    #le but c'est de retourner une approximation à l'entier qui est juste inférieure à 
                    #l'évaluation et ça nous permet d'élaguer le plutot possible. 
                    elif int(evaluer(n,d))>M:     
                        fils = diviser(n,b,v,d)
                        fils_retenus = []
                        for f in fils:
                            evaluation = evaluer(f,d)
                            if int(evaluation)>M: #int(eval)>M : pour optimiser
                                if f.objet==len(d):
                                    M = evaluation
                                    res = f
                                else:
                                    fils_retenus.append(f)
                        na = fils_retenus + na
                #Remonter dans l'arborescence en utilisant l'attribut "pere" afin de retourner
                #la solution exacte trouvée
                sol = [0 for _ in range(len(b))]
                M = res.m
                while res.val!=-1:
                    sol[d[res.objet-1][1]] = res.val
                    res = res.pere
                return M,sol
                
        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button("Upload file"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # file_path = filedialog.askopenfilename(master=root)
            file_path = easygui.fileopenbox()
            
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr = branch_and_bound_ukp(b, v, w)

                arr = np.array(arr)
                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################

                ###pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + " in (time): " + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "bb")

    elif page == "Dynamic Programing":
        st.title("Dynamic Programing Algorithm")
        st.subheader("Description")
        st.write(
            "Dynamic programming is an algorithmic method to solve optimization problems. This resolution consists of breaking down the problem into smaller and smaller sub-problems by saving the intermediate results. This backup avoids calculating the same thing twice."
        )

        st.subheader("Algorithm")

        st.code(
            """
          def dp_ukp(w,n,b,v):
            # k contient le gain maximal associé aux poids allant de 0 à w (poids max)
            k=[0 for i in range(w+1)]
            
            # items contient la liste des objets choisis pour obtenir le gain maximal associé aux poids allant de 0 à w
            items=[[] for i in range(w+1)]
            
            for wi in range(w+1): 
                for j in range(n): 
                    if (v[j]<=wi): #si le poids de l'objet est inférieur au poids considéré
                        tmp=k[wi] # tmp sera utilise pour savoir si k[wi] a été modifié (pour modifier items en conséquences)
                        k[wi]=max(k[wi],k[wi-v[j]]+b[j])
                        if (k[wi]>tmp): # si k[wi] a changé (donc on a trouvé une val supérieur à la valeur précedente sauvgardée dans tmp), on met à jour les objets pris
                            items[wi]=[]
                            for l in range(len(items[wi-v[j]])):
                                items[wi].append(items[wi-v[j]][l])
                            items[wi].append(j+1) 
                            # donc la liste des objets pris est la liste des objets de items[wi-wt[j] en plus de l'objet j ajouté
            return k[w],items[w]
        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button("Upload file"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()
            # file_path = filedialog.askopenfilename(master=root)
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res,arr =dp_ukp(w, n, b, v)
                arr = np.array(arr)
                dispTime = time.time()
                #######################################################################
                objectIDs = [arr[0]]
                nbObjects = [1]
                for i in range(1,len(arr)):
                	if arr[i]==arr[i-1]:
                		nbObjects[-1] += 1
                	else:
                		objectIDs.append(arr[i])
                		nbObjects.append(1)
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                
                pdarr = pd.DataFrame(data=arr)
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(res)
                    + " in (time): "
                    + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "dp")

    elif page == "Density Ordered Greedy":
        st.title("Density Ordered Greedy Algorithm")
        st.subheader("Description")
        st.write(
            "In this algorithm, objects are sorted according to the descending order of their utilities (ratio between benefits and weight). We go through the sorted list and for each object, we take the maximum possible number of units of the latter.\n This is the most widely used method among construction methods, as it often gives good results, and in some cases even gives optimal solutions, whereas in the worst case, the solution obtained is half of the optimal solution."
        )

        st.subheader("Algorithm")

        st.code(
            """
           def density_ordered_greedy_ukp(b, v, w):
                #Construction de la liste des densités (utilités des objets)
                d = [(b[i]/v[i],i) for i in range(len(v))]
                #Ordonner la liste des densités par ordre décroissant
                d.sort(key=lambda x:x[0], reverse = True)
                #La variable "M" stockera le gain de la solution proposée par cette heuristique
                M = 0
                #La liste "res" contiendra la solution à la fin. res[i] est le nombre d'exemplaires
                #de l'objet "i"
                res = [0 for _ in range(len(d))]
                #On fait un parcours complet des objets
                for i in range(len(d)):
                    #Si la taille restante du sac à dos est égale à 0, on arrete le parcours
                    if w==0:
                        break
                    #Sinon, s'il reste de l'espace dans le sac à dos : 
                    #On calcule dans "nb" le nombre d'exemplaires possibles de l'objet d[i][1]
                    nb = int(w/v[d[i][1]])
                    #On met à jour le gain "M" après l'ajout de "nb" exemplaires de l'objet d[i][1]
                    M += nb * b[d[i][1]]
                    #On met à jour le poids restant dans le sac à dos "w"
                    w -= nb * v[d[i][1]]
                    #On sauvegarde le résultat "nb" dans la liste "res" à la position d[i][1] (ID de l'objet)
                    res[d[i][1]] = nb
                #On retourne le gain du sac à dos "M" et la liste des nombres d'exemplaires de chaque
                #objet "res"
                return M,res,w
        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button("Upload file"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()
            # file_path = filedialog.askopenfilename(master=root)
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr, poid = density_ordered_greedy_ukp(b, v, w)
                arr = np.array(arr)
                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + ", weight :"
                    + str(w-poid)+ " in (time): " + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "dog")

    elif page == "Weighted Ordered Greedy":
        st.title("Weighted Ordered Greedy Algorithm")
        st.subheader("Description")
        st.write(
            "This approach involves sorting objects in ascending or descending order of their weight (or volumes). The list of objects thus sorted is scanned and for each type of object, we take as many copies of the latter as possible and so on until the complete scan of the list of objects. This heuristic very seldom gives good results. Often, it is better to sort objects according to the descending order of their weight because usually objects, with small weights, have a small gain"
        )

        st.subheader("Algorithm")

        st.code(
            """
           def weighted_ordered_greedy_ukp(b, v, w, ordre_croissant=False):
                #Construction de la liste "d" contenant les couples (a,b) tel que a est le volume 
                #de l'objet b. Cette structure nous permet de trouver le gain de l'objet 
                #rapidement après le trie de cette liste.
                d = [(v[i],i) for i in range(len(v))]
                #Ordonner la liste "d" selon l'ordre croissant ou décroissant selon le paramètre
                #ordre_croissant
                d.sort(key=lambda x:x[0], reverse=not ordre_croissant)
                #La variable "M" stockera le gain de la solution proposée par cette heuristique
                M = 0
                #La liste "res" contiendra la solution à la fin. res[i] est le nombre d'exemplaires
                #de l'objet "i"
                res = [0 for _ in range(len(d))]
                #On fait la meme boucle que celle du Density Ordered Greedy
                for i in range(len(d)):
                    if w==0:
                        break
                    nb = int(w/v[d[i][1]])
                    M += nb * b[d[i][1]]
                    w -= nb * v[d[i][1]]
                    res[d[i][1]] = nb
                #On retourne le gain du sac à dos "M" et la liste des nombres d'exemplaires de chaque
                #objet "res"
                return M,res,w
        """,
            language="python",
        )
        colo1, colo2 = st.beta_columns((2, 1))
        colo1.subheader("Import data (Apply the algorithm on a signle file)")
        colo2.text("")
        ordreCcheck = colo2.checkbox("Ordre croissant")
        st.text("")
        col1, col2, col3 = st.beta_columns(3)

        if col2.button("Upload file"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()
            # file_path = filedialog.askopenfilename(master=root)
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr,poid = weighted_ordered_greedy_ukp(b, v, w, ordreCcheck)
                arr = np.array(arr)
                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) +", weight: "+str(w-poid)+ " in (time): " + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                if ordreCcheck:
                    stats(dir_path, "wdogT")

                else:
                    stats(dir_path, "wdogF")

    elif page == "Heuristic By Rounding":
        st.title("Heuristic By Rounding Algorithm")
        st.subheader("Description")
        st.write(
            "This heuristic, based on the greedy order Density approach, builds a solution piece by piece, always choosing the next piece that offers the most obvious and immediate advantage without with regard to future consequences. It produces good results that are close to optimal."
        )

        st.subheader("Algorithm")

        st.code(
            """
           def ratios(b,v,V_t):
                ratios = [(b[i]/v[i],i) for i in range(len(b))]
                ratios.sort(reverse=True)
                return ratios

            def intermediate_solution(b,v,V_t):
                x = [0 for i in range(len(b))]
                ratio = ratios(b,v,V_t)
                cap_act = 0
                capacite = V_t
                cap = cap_act
                i = 0
                benefice = 0 
                while( capacite >= cap_act + v[ratio[i][1]]):
                    if(i==len(b)):
                        i = 0
                    benefice += b[ratio[i][1]]
                    cap_act += v[ratio[i][1]]
                    x[ratio[i][1]] = x[ratio[i][1]]+1
                    i += 1  
                return x,cap_act,benefice,i

            def heuristic_arrondi(b,v,V_t):
                x,capacite,benefice,i_max = intermediate_solution(b,v,V_t)
                ratio = ratios(b,v,V_t)
                difference = V_t - capacite
                difference_min = 100000000 
                i = 0
                while(difference_min != difference):
                    if(i == i_max):
                        i = 0
                    if(x[ratio[i][1]] == 0):
                        i += 1
                    else:
                        if(capacite + b[ratio[i][1]] <= V_t):
                            benefice += b[ratio[i][1]]
                            capacite += v[ratio[i][1]]
                            x[ratio[i][1]] = x[ratio[i][1]]+1
                        i += 1  
                        difference = V_t - capacite
                        difference_min = min(difference , difference_min)
                return capacite,x,benefice

        """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")

        col1, col2, col3 = st.beta_columns(3)

        if col2.button("Upload file"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()
            # file_path = filedialog.askopenfilename(master=root)
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                poid, arr, res = heuristic_arrondi(b, v, w)
                arr = np.array(arr)
                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + ", weight :"
                    + str(poid) +" in (time): " + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "hr")

    elif page == "Recuit simulé":
        st.title("Recuit simulé Algorithm")
        st.subheader("Description")
        st.write(
            "Is an iterative procedure that seeks lower cost configurations while accepting controlled manner of configurations that degrade the cost function. At each step, the simulated annealing heuristics considers a state close s * to the current state s, and decide probabilistically between moving the system to state s * or staying in state s. These probabilities eventually lead the system to move to lower cost statements. In general, this step is repeated until the system reaches a state that is good enough for the application, or until a given calculation budget is exhausted"
        )

        st.subheader("Algorithm")

        st.code(
            """
        def cool(temprature, coolingFactor):
            return temprature * coolingFactor


        # Definir le voisinage d'une solution
        def getNeighbour(solution, taille, tab_poids_new, capacity):
            np.random.seed()
            sol = solution.copy()
            i = 0
            # generer un indice aleatoire
            x = np.random.randint(taille)
            # alterer la valeur de la solution correspondante a l'indice de 1 a 0
            if sol[x] == 1:
                sol[x] = 0
            # essayer d'alterer une des valeurs de la solution de 0 a 1
            else:
                capacityRest = capacity - get_poids_total(sol, tab_poids_new)
                listItemCanEnter = []
                for i in range(len(sol)):
                    if capacityRest > tab_poids_new[i] and sol[i] == 0:
                        listItemCanEnter.append(i)
                if len(listItemCanEnter) != 0:
                    ind = np.random.randint(len(listItemCanEnter))
                    sol[listItemCanEnter[ind]] = 1
                # essayer d'alterer une des valeurs de la solution de 1 a 0
                else:
                    listItemPris = []
                    for i in range(len(sol)):
                        if sol[i] == 1:
                            listItemPris.append(i)
                    if len(listItemPris) != 0:
                        ind = np.random.randint(len(listItemPris))
                        sol[listItemPris[ind]] = 0
                    # essayer d'alterer une des valeurs de la solution de 0 a 1
                    capacityRest = capacity - get_poids_total(sol, tab_poids_new)
                    listItemCanEnter = []
                    for i in range(len(sol)):
                        if capacityRest > tab_poids_new[i] and sol[i] == 0:
                            listItemCanEnter.append(i)
                    if len(listItemCanEnter) != 0:
                        ind = np.random.randint(len(listItemCanEnter))
                        sol[listItemCanEnter[ind]] = 1
            return sol


        # passage d'une solution a une autre
        def getNextState(solution, taille, tab_poids_new, tab_gain_new, capacity, temperature):

            # generer le voisin
            newSolution = getNeighbour(solution, taille, tab_poids_new, capacity)
            # evaluer le voisin
            evalNewSol = eval_solution(newSolution, tab_gain_new)
            # evaluer l'ancienne solution
            evalOldSol = eval_solution(solution, tab_gain_new)
            # calculer delta
            delta = evalNewSol - evalOldSol

            if delta > 0:
                return newSolution  # solution meilleur => accepée
            else:
                x = np.random.rand()  # generer un nombre aleatoire

                # critere d'acceptation de la solution
                if x < math.exp(delta / temperature):
                    return newSolution  # verifié => accepter nouvelle solution
                else:
                    return solution  # non verifié => garder l'ancienne solution


        # evaluation d'une solution obtenue
        def eval_solution(solution, tab_gain_new):
            gain_total = 0
            for i in range(len(solution)):
                gain_total = gain_total + solution[i] * tab_gain_new[i]

            return gain_total


        # trier par utilité
        def trier_objet_utility(items):
            items.sort(key=lambda x: x[1] / x[0], reverse=True)
            return items


        # le nombre maximal que peut contenir le sac de chaque objet
        def get_max_number_item(items, capacity=0):
            tab_number = [capacity // item[0] for item in items]
            return tab_number, sum(tab_number)


        # definir une table de gain correspondante a l'écriture binaire d'une solution
        def get_tab_gain_new(items_sorted, tab_max_nb):
            tab_gain = []
            for i in range(len(tab_max_nb)):
                tab = [items_sorted[i][1]] * tab_max_nb[i]
                tab_gain = tab_gain + tab

            return tab_gain


        # definir une table de poids  correspondante a l'ecriture binaire d'une solution
        def get_tab_poid_new(items_sorted, tab_max_nb):
            tab_poid = []
            for i in range(len(tab_max_nb)):
                tab = [items_sorted[i][0]] * tab_max_nb[i]
                tab_poid = tab_poid + tab
            return tab_poid


        # le poids obtenue par une solution ecrite sous sa forme binaire
        def get_poids_total(bsol, tab_poid_new):
            poid_total = 0
            for i in range(len(bsol)):
                poid_total = poid_total + bsol[i] * tab_poid_new[i]
            return poid_total


        # convertir une solution en n en une forme binaire
        def ntobinary(nsol, max_num_tab):
            bsol = []
            for i in range(len(max_num_tab)):
                for p in range(nsol[i]):
                    bsol.append(1)
                for p in range(nsol[i], max_num_tab[i]):
                    bsol.append(0)
            return bsol


        # generer une solution aleatoire
        def gen_random_sol(tab, n, capacity):
            weight = []
            profits = []
            capacityleft = capacity
            sol = []
            # initialiser la solution avec des 0
            for k in range(0, n):
                sol.append(0)
            for i in range(0, n):
                weight.append(tab[i][0])
                profits.append(tab[i][1])
            j = 0
            # TQ capacité max non atteinte
            while j < n and capacityleft > 0:
                # generer un indice aleatoire
                index = np.random.randint(0, n - 1)
                # calculer le nombre maximale d'exemplaires qu'on peut rajouter
                maxQuantity = int(capacityleft / weight[index]) + 1
                if maxQuantity == 0:
                    nbItems = 0
                else:  # si maxQuantity>0 generer un nombre aleatoire d'exemplaires inferieurs a maxQuantity
                    nbItems = np.random.randint(0, maxQuantity)
                sol[index] = nbItems
                capacityleft = capacityleft - weight[index] * sol[index]
                j = j + 1

            gain_out = 0  # calculer le gain obtenu
            for i in range(n):
                gain_out = gain_out + profits[i] * sol[i]

            return gain_out, capacityleft, sol


        # convertir une solution binaire en une solution en n
        def binaryToNsolution(solution, tab_max_nb):
            solN = []
            indMin = 0
            for i in range(len(tab_max_nb)):
                indMax = indMin + tab_max_nb[i]
                solN.append(sum(solution[indMin:indMax]))
                indMin = indMax
            return solN


        # la fonction principale du recuit simulé
        def simulatedAnnealing(
            itemsIn,
            capacity,
            solinit,
            samplingSize,
            temperatureInit,
            coolingFactor,
            endingTemperature,
        ):

            items = itemsIn.copy()
            for i in range(len(items)):
                items[i].append(solinit[i])
            # trier objets par utilité
            items_sorted = trier_objet_utility(items)
            # reordonner la solution
            solinitsorted = []
            for i in range(len(items_sorted)):
                solinitsorted.append(items_sorted[i][2])
            # recupere le tabeau contenant le nombre max d'exemplaires de chaque objet
            tab_max_nb, taille = get_max_number_item(items_sorted, capacity)
            tab_poids_new = get_tab_poid_new(items_sorted, tab_max_nb)
            tab_gain_new = get_tab_gain_new(items_sorted, tab_max_nb)
            # convertir la solution en une solution binaire
            solCurrent = ntobinary(solinitsorted, tab_max_nb)
            # evaluer la solution
            evalsol = eval_solution(solCurrent, tab_gain_new)
            # recuperer la temperature initaile
            temperature = temperatureInit
            # initialiser la meilleur solution
            bestSol = solCurrent.copy()
            bestEval = evalsol
            while temperature > endingTemperature:

                for i in range(samplingSize):
                    # passage a une nouvelle configuration
                    solCurrent = getNextState(
                        solCurrent, taille, tab_poids_new, tab_gain_new, capacity, temperature
                    )
                    # evaluer la nouvelle configuation
                    evalCurrent = eval_solution(solCurrent, tab_gain_new)
                    # si meilleur MAJ de la meilleur solution
                    if evalCurrent > bestEval:
                        bestSol = solCurrent.copy()
                        bestEval = evalCurrent
                # MAJ la temperature
                temperature = cool(temperature, coolingFactor)

            objects = []
            solution = []
            # convertir la solution binaire trouver en une solution en n
            Nsol = binaryToNsolution(bestSol, tab_max_nb)
            for i, item in enumerate(Nsol):
                if item != 0:
                    objects.append(items[i])
                    solution.append(item)
            poids = 0
            for i, obj in enumerate(objects):
                poids += obj[0] * solution[i]
            # retourne la solution son gain et son poids
            return objects, solution, Nsol, bestEval, poids

        """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")

        col1, col2, col3 = st.beta_columns(3)
        colo1, colo2, colo3 = st.beta_columns(3)
        random.seed(1)
        param1 = col1.number_input("Insert number of stages", format="%d", value=5)
        param2 = col2.number_input("Insert initial temp", format="%d", value=1000)
        param3 = col3.number_input("Insert cooling factor", value=0.9)
        param4 = col1.number_input("Insert final temp", format="%d", value=5)

        col3.text("")
        col3.text("")
        if col3.button("Upload file"):
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                capacity = w
                start_time = time.time()
                data= pd.DataFrame(list(zip(v,b)))
                items_init=data.values.tolist()
                sol = density_ordered_greedy_ukp(b, v, capacity)[1]
                gain_out, capacityleft, sol = gen_random_sol(items_init, n, capacity)
                objects, solution, Nsol, bestEval, poids = simulatedAnnealing(
                    items_init, capacity, sol, param1, param2, param3, param4
                )
                dispTime = time.time()
                solution = np.array(solution)
                pdarr = pd.DataFrame(solution, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(bestEval)
                    + ", weight :"
                    + str(poids)
                    + " in (time): "
                    + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "rs")

    elif page == "Genetic Algorithm":
        st.title("Genetic Algorithm")
        st.subheader("Description")
        st.write(
            "Genetic algorithms belong to the family of evolutionary algorithms. Their goal is to obtain an approximate solution to an optimization problem, when there is no exact method (or the solution is unknown) to solve it in a reasonable time. Genetic algorithms use the notion of natural selection and apply it to a population of potential solutions to the given problem. The solution is approached by successive «bonds», as in a procedure of separation and evaluation (branch & bound), except that these are formulas that are sought and no longer directly values."
        )

        st.subheader("Algorithm")

        st.code(
            """
            class chromosome:
                def __init__(self, gain, poids, solution):
                    self.gain = gain
                    self.poids = poids
                    self.solution = solution

            def generation_population(n, w, b, v, N):
                population = []
                gain,sol,poids = density_ordered_greedy_ukp(b, v, w)
                population.append(chromosome(gain, w-poids, sol))
                for _ in range(N-1):
                    sol = []
                    gain = 0
                    poids = w
                    for i in range(n):
                        nb = random.randint(0, int(poids/v[i]))
                        gain += nb*b[i]  
                        poids -= nb*v[i]
                        sol.append(nb)
                    population.append(chromosome(gain, w-poids, sol))
                return population

            def recherche_dico(liste, v):
                i = 0
                j = len(liste)-1
                while i <= j:
                    m = (i + j)//2
                    if liste[m]==v:
                        return m 
                    elif liste[m]<v:
                        i = m + 1
                    else:
                        j = m - 1
                return i

            def selection_elitiste(population, NI):
                pool = []
                population.sort(key=lambda x:x.gain, reverse=True)
                for i in range(min(len(population), NI)):
                    pool.append(population[i])
                return pool

            def selection_roue_loterie(population, NI):
                pop = list(population)
                pool = []
                for _ in range(min(len(pop), NI)):
                    gain_t = 0
                    probas = []
                    for chromosome in pop:
                        gain_t += chromosome.gain
                        probas.append(gain_t)
                    probas = np.array(probas)/gain_t
                    rand = random.uniform(0,1)
                    ind = recherche_dico(probas,rand)
                    pool.append(pop[ind])
                    pop.pop(ind)
                return pool

            def corriger(liste, n, w, b, v):
                poids = 0
                gain = 0
                for i in range(n):
                    poids += liste[i] * v[i]
                    gain += liste[i] * b[i]
                while poids>w:
                    r = random.randint(0,n-1)
                    if liste[r]>0:
                    liste[r] -= 1
                    poids -= v[r]
                    gain -= b[r]
                return chromosome(gain, poids, liste)

            def croisement_1_point(parent1, parent2, n, w, b, v):
                k = random.randint(1,n-1)
                enfant1 = corriger(parent1.solution[0:k] + parent2.solution[k::], n, w, b, v)
                enfant2 = corriger(parent2.solution[0:k] + parent1.solution[k::], n, w, b, v)
                return enfant1, enfant2

            def croisement(pool, Pc, n, w, b, v):
                selected = []
                for chromosome in pool:
                    r = random.uniform(0,1)
                    if r<=Pc:
                        selected.append(chromosome)
                enfants = []
                for i in range(0,len(selected)-1,2):
                    enfant1, enfant2 = croisement_1_point(selected[i], selected[i+1], n, w, b, v)
                    enfants.append(enfant1)
                    enfants.append(enfant2)
                return enfants

            def mutation(pool, Pm, n, w, b, v):
                enfants = []
                for chromosome in pool:
                    r = random.uniform(0,1)
                    if r<=Pm:
                        liste = list(chromosome.solution)
                        r = random.randint(0,n-1)
                        val_s = str(bin(liste[r]))
                        rand = random.randint(2,len(val_s)-1)
                        if val_s[rand]=="0":
                            val_s = val_s[0:rand] + '1' + val_s[rand+1:]
                        else:
                            val_s = val_s[0:rand] + '0' + val_s[rand+1:]
                        liste[r] = int(val_s,2)
                        enfants.append(corriger(liste, n, w, b, v))
                return enfants

            def maj_population(population, enfants1, enfants2, N):
                population = population + enfants1 + enfants2
                population.sort(key=lambda x:x.gain, reverse=True)
                return population[0:N]

            def arreter(t, max_it, mem, max_n, stagnation):
                if t==max_it:
                    return True
                if stagnation:
                    if len(mem)==max_n:
                        val = mem[0]
                        for elt in mem:
                            if elt!=val:
                                return False
                        return True
                return False  

            def AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation):
                population = generation_population(n, w, b, v, N)
                t = 1
                mem = []
                while not arreter(t, max_it, mem, max_n, stagnation):
                    pool = selection_elitiste(population, NI)
                    enfants1 = croisement(pool, Pc, n, w, b, v)
                    enfants2 = mutation(pool, Pm, n, w, b, v)
                    population = maj_population(population, enfants1, enfants2, N)
                    if stagnation:
                        if len(mem)==max_n:
                            mem.pop(0)
                        mem.append(population[0].gain)
                    t += 1
                return population[0].gain, population[0].poids, population[0].solution

        """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        colo1, colo2, colo3 = st.beta_columns(3)
        random.seed(1)
        max_it = col1.number_input("Insert a iteration max number", format="%d", value=500)
        max_n = col2.number_input("Insert a max stagnation iteration number", format="%d", value=10)
        N = col3.number_input("Insert a population size", format="%d", value=2500)
        NI = col1.number_input("Insert a subpopulation size", format="%d", value=100)
        Pc = col2.number_input("Insert a probability of crossing",value=0.6)
        Pm = col3.number_input("Insert a probability of mutation",value=0.4)
        stagnation = col1.checkbox("stagnation")

        if col3.button("Upload file"):

            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, poid, arr = AG(
                    n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation
                )

                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(res)
                    + ", weight : "
                    + str(poid)
                    + " in (time): "
                    + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "ag")

    elif page == "Ant colony":
        st.title("Ant colony Algorithm")
        st.subheader("Description")
        st.write(
            "This approach involves sorting objects in ascending or descending order of their weight (or volumes). The list of objects thus sorted is scanned and for each type of object, we take as many copies of the latter as possible and so on until the complete scan of the list of objects. This heuristic very seldom gives good results. Often, it is better to sort objects according to the descending order of their weight because usually objects, with small weights, have a small gain"
        )

        st.subheader("Algorithm")

        st.code(
            """
                class AntColony:

                    def __init__(self,benifices,poids, utilites,n_objets, W,densitySol,n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
                        
                        #Args:
                            #benifices (1D numpy.array): Les benifices de chaque objet.
                            #poids (1D numpy.array): Les poids de chaque objet.
                            #poids (1D numpy.array): L'utilité d'un objet de chaque objet.
                            #n_objets (int): Nombre d'objets
                            #W (int): La capacité du sac
                            #densitySol (liste): Solution generé par heuristique spécifique
                            #n_ants (int): Nombre de fourmis par iterations
                            #n_best (int): Nombre de meilleures fourmis qui déposent le pheromone
                            #n_iteration (int): nombre d'iteration
                            #decay (float): 1-Taux d'evaporation de pheromone
                            #alpha (int or float): Exposant dans le pheromone, Alpha grand donne plus de poid au pheromone
                            #beta (int or float): Exposant sur l'utilité, Beta grand donne plus de poid a l'utilité
                        #Example:
                            #ant_colony = AntColony(benifices,poids, utilites,n_objets, W,densitySol,n_ants, n_best, n_iterations, decay, alpha=1, beta=2)         
                        
                        self.utilites  = utilites
                        self.W=W
                        self.n_objets=n_objets
                        self.poids=poids
                        self.benifices=benifices
                        self.pheromone = np.ones(n_objets)
                        #ajouter du pheromone au objets generé par heuristique
                        for i,s in enumerate(densitySol):
                        if s>0:
                            self.pheromone[i]+=s*0.1
                        self.all_inds = range(len(utilites))
                        self.n_ants = n_ants
                        self.n_best = n_best
                        self.n_iterations = n_iterations
                        self.decay = decay
                        self.alpha = alpha
                        self.beta = beta

                    

                    def run(self,n_candidats,densitySol):
                        
                        #Args:
                            #n_candidats (int): Nombre de candidats pour construire les solutions
                            #densitySol (Gain:int,sol:liste,Poid:int): Solution generé par heuristique spécifique
                        #Example:
                            #best_sol = ant_colony.run(n_candidats,densitySol)         
                        
                        best_solution = (densitySol[1], densitySol[0],densitySol[2])
                        best_solution_all_time = (densitySol[1], densitySol[0],densitySol[2])
                        for i in range(self.n_iterations):
                            #generer toutes les solutions par les fourmis 
                            all_solutions = self.gen_all_solutions(n_candidats)
                            #mise a jours des pistes pheromones
                            self.spread_pheronome(all_solutions, self.n_best, best_solution=best_solution)
                            #Choisir meilleure solution dans l'iteration actuelle
                            best_solution = max(all_solutions, key=lambda x: x[1])
                            #print (best_solution)
                            #Mettre a jour la meilleure solution globale 
                            if best_solution[1] > best_solution_all_time[1]:
                                best_solution_all_time = best_solution  
                                
                            #evaporation de pheromone            
                            self.pheromone= self.pheromone * self.decay
                            self.pheromone[self.pheromone<1]=1   
                            
                        print(self.gen_sol_gain(best_solution_all_time[0]) )  
                        print(self.gen_path_poid(best_solution_all_time[0]) )          
                        return best_solution_all_time
                    
                
                    def spread_pheronome(self, solutions, n_best, best_solution):
                    
                        #Dépose le pheromone sur les n_best meilleures solutions
                            
                    sorted_solution = sorted(solutions, key=lambda x: x[1],reverse=True)
                    for sol, gain,poid in sorted_solution[:n_best]:
                            for i in sol:
                            self.pheromone+= 0.00001*gain
                            

                    def gen_sol_gain(self, sol):
                    
                        #Calcul le gain d'une solution.
                        #Pas necessaire mais peut servir à verifier les résultats (test unitaire)
                            
                    
                    total_fitness = 0
                    for i,ele in enumerate(sol):
                            total_fitness += self.benifices[i]*ele
                    return total_fitness

                    def gen_path_poid(self, sol):
                    
                        #Calcul le poid d'une solutions.
                        #Pas necessaire mais peut servir à verifier les résultats (test unitaire)
                            
                    
                    total_fitness = 0
                    for i,ele in enumerate(sol):
                            total_fitness += self.poids[i]*ele
                    return total_fitness



                    def gen_all_solutions(self,n_candidats):
                        
                        #Generer toutes les solutions par les fourmis
                            
                        
                        all_solutions = []
                        for i in range(self.n_ants):
                        #Positionner la fourmis sur un objets de départ aleatoirement
                        n=rn.randint(0,self.n_objets-1)
                        #generation de la solution par la fourmis en utilisant n_candidats
                        solution = self.gen_sol(n,n_candidats)
                        
                        #ajouter la solution a la liste de toute les solutions
                        all_solutions.append((solution[0], solution[1], solution[2]))
                        return all_solutions


                    def listeCandidate(self,phero,visited,n_candidats):
                    
                        #retourne La liste des candidats pour une solution
                            
                    
                    pheromone=phero.copy()
                    
                    pheromone[list(visited)]= 0
                    #rn.choices returns a list with the randomly selected element from the list.
                    #weights to affect a probability for each element
                    
                    c=rn.choices(self.all_inds,weights=[p for p in pheromone],k=n_candidats)
                    c=list(set(c))
                    i=len(c)
                    #while i<n_candidats:
                        #n=rn.randint(0,self.n_objets-1)
                        #if n not in visited:
                            #c.append(n)
                            #i+=1
                    nb_candidats=len(c)
                    
                    return c ,pheromone


                    #generer solution c'est bon
                    def gen_sol(self, start,n_candidats):
                        
                        #Construit la solution avec n_candidats et start comme premier objet
                            
                        
                        sol = np.zeros(self.n_objets)
                        poidrestant=self.W
                        visited = set()#liste des objets visité
                        #ajouter le premier objet
                        r=rn.randint(1,poidrestant//poids[start])
                        sol[start]=r
                        poidrestant-=poids[start]*r
                        gain=r*benifices[start]
                        visited.add(start)#ajouter le debut a la liste civité
                        
                        #la liste candidates avec les pheromones mis a jours localement (0 sur les visited)
                        candidats,pheromones=self.listeCandidate(self.pheromone,visited,n_candidats)
                        
                        for i in candidats :
                            #Choisir le prochain objets parmi les candidats ainsi que le nombre
                            move,nb = self.pick_move(pheromones, candidats,n_candidats,self.utilites, visited)
                            toPop=candidats.index(move)
                            candidats.pop(toPop)
                            n_candidats-=1
                            np.delete(pheromones,toPop)#rendre le pheromone à 0 pour indiquer qu'il a été visité
                        
                            
                            
                            #Mise a jour poidRestant et gain de la solution
                            poidrestant-=poids[move]*nb
                            while poidrestant < 0:
                            nb-=1
                            poidrestant+=poids[move]
                            
                            sol[move]=nb
                            gain+=nb*benifices[move]
                            
                            #ajouter l'objet a visited
                            visited.add(move)  
                        #print("s",i,sol,gain)
                        return sol,gain,self.W-poidrestant
                    
                    def pick_move(self, pheromone,liste_cand, n_candidats, utilite, visited):
                        
                        pheromone=pheromone.copy()[liste_cand]
                    #generer le regle de déplacement sur les candidat
                        numerateurs=(pheromone**self.alpha)* (( 1.0 / (utilite[liste_cand]))**self.beta)
                        
                    #formule vu en cours
                        
                        P = (numerateurs / numerateurs.sum())
                        
                        #choisir l'objet suivant en utilisant les probabilité P
                        move = np_choice(liste_cand, 1, p=P)[0] 
                        #nombre d'objet a prendre
                        nb=self.W//self.poids[move]
                        #nb=rn.randint(0,self.W//self.poids[move])
                    
                        return (move,nb)
                    
                def density_ordered_greedy_ukp(b, v, w):
                    d = [(b[i]/v[i],i) for i in range(len(v))]
                    d.sort(key=lambda x:x[0], reverse = True)
                    M = 0
                    res = [0 for _ in range(len(d))]
                    for i in range(len(d)):
                        if w==0:
                            break
                        nb = int(w/v[d[i][1]])
                        M += nb * b[d[i][1]]
                        w -= nb * v[d[i][1]]
                        res[d[i][1]] = nb
                    return M,res,w  

            """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)

        n_ants = col1.number_input("Insert a number of ants", format="%d", value=100)
        n_best = col2.number_input("Insert a n_best", format="%d", value=10)
        n_iterations = col3.number_input("Insert a number of iterations", format="%d", value=10)
        decay = col1.number_input("Insert a decay",value=0.8)
        alpha = col2.number_input("Insert a alpha", format="%d", value=1)
        beta = col3.number_input("Insert a beta", format="%d", value=2)
        if col2.button("Upload file"):
            file_path = easygui.fileopenbox()
            if file_path != "":
                st.text("File imported !")
                n, w, v, b = read_data_3_single(file_path)
                densitySol = density_ordered_greedy_ukp(b, v, w)
                benifices = np.array(b)
                poids = np.array(v)
                utilites = poids / benifices
                st.subheader("Solution")

                colony = AntColony(
                    benifices,
                    poids,
                    utilites,
                    n,
                    w,
                    densitySol[1],
                    n_ants,
                    n_best,
                    n_iterations,
                    decay,
                    alpha=alpha,
                    beta=beta,
                )
                random.seed(1)
                start_time = time.time()
                arr, res, poid = colony.run(30, densitySol)
                dispTime = time.time()
                #######################################################################
                objectIDs = []
                nbObjects = []
                for i in range(1,len(arr)+1):
                	if arr[i-1]!=0:
                		objectIDs.append(i)
                		nbObjects.append(arr[i-1])
                arr = {'Object ID': np.array(objectIDs), "Number of elements": np.array(nbObjects)}
                pdarr = pd.DataFrame(data=arr)
                #########################################
                #pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(res)
                    + ", weight :"
                    + str(poid)
                    + " in (time): "
                    + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time results")
        colo1.text("")
        if colo2.button("Select instances directory"):
            # root = tk.Tk()
            # root.focus_get()
            # root.withdraw()
            # root.focus_force()

            # dir_path = filedialog.askdirectory(master=root)
            dir_path = easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "ac")

    else:
        st.title("Comparaison")
        st.text("")
        colo1, colo2 = st.beta_columns((3, 5))

        colo1.text("Select methods to compare time and gain:")
        bbcheck = colo2.checkbox("Branch and Bound")
        dpcheck = colo2.checkbox("Dynamic Programing")
        dogcheck = colo2.checkbox("Density Ordered Greedy")
        wdogTcheck = colo2.checkbox("Weighted Ordered Greedy (asc)")
        wdogFcheck = colo2.checkbox("Weighted Ordered Greedy (desc)")
        hrcheck = colo2.checkbox("Heuristic By Rounding")
        agcheck = colo2.checkbox("Genetic Algorithm")
        accheck = colo2.checkbox("Ant colony")
        rscheck = colo2.checkbox("Recuit simulé")
        if agcheck:

            random.seed(1)
            st.subheader("Genetic Algorithm Parameters")
            co1, co2, co3 = st.beta_columns(3)

            max_it = co1.number_input("Insert iteration max number", format="%d", value=500)
            max_n = co2.number_input("Insert max stagnation iteration number", format="%d", value=10)
            N = co3.number_input("Insert population size", format="%d", value=2500)
            NI = co1.number_input("Insert subpopulation size", format="%d", value=100)
            Pc = co2.number_input("Insert probability of crossing",value=0.6)
            Pm = co3.number_input("Insert probability of mutation",value=0.4)
            stagnation = co1.checkbox("Stagnation")

        if accheck:
            st.subheader("Ant Colony Parameters")
            c1, c2, c3 = st.beta_columns(3)

            n_ants = c1.number_input("Insert the number of ants", format="%d", value=100)
            n_best = c2.number_input("Insert n_best", format="%d", value=10)
            n_iterations = c3.number_input("Insert the number of iterations", format="%d", value=10)
            decay = c1.number_input("Insert decay",value=0.8)
            alpha = c2.number_input("Insert alpha", format="%d", value=1)
            beta = c3.number_input("Insert beta", format="%d", value=2)
        if rscheck:
            st.subheader("Recuit simulé Parameters")
            cl1, cl2, cl3 = st.beta_columns(3)
            param1 = cl1.number_input("Insert number of stages", format="%d", value=5)
            param2 = cl2.number_input("Insert initial temp", format="%d", value=1000)
            param3 = cl3.number_input("Insert cooling factor", value=0.9)
            param4 = cl1.number_input("Insert final temp", format="%d", value=5)


        col1, col2, col3 = st.beta_columns(3)
        if col3.button("Select instances directory"):

            dir_path = easygui.diropenbox()
            if dir_path != "":

                st.subheader("Time Comparaison")
                st.text("")
                st.text("")
                statsComp(
                    dir_path,
                    bbcheck,
                    dpcheck,
                    dogcheck,
                    wdogTcheck,
                    wdogFcheck,
                    hrcheck,
                    agcheck,
                    accheck,
                    rscheck,
                )


################################################################

main()
