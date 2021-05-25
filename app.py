import streamlit as st
import numpy as np
import pandas as pd

# import tkinter as tk
# from tkinter import filedialog
import easygui
import random
import random as rn
from numpy.random import choice as np_choice
import time
from os import listdir
from os.path import isfile, join

##import matplotlib.pyplot as plt
# from tabulate import tabulate

import time

# b = [5, 4, 6, 2]  # Bénéfice
# v = [49, 33, 60, 32]  # Volume
# w = 130
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

        print(self.gen_sol_gain(best_solution_all_time[0]))
        print(self.gen_path_poid(best_solution_all_time[0]))
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
        i = len(c)
        while i < n_candidats:
            n = rn.randint(0, self.n_objets - 1)
            if n not in visited:
                c.append(n)
                i += 1

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
            candidats.pop(candidats.index(move))
            pheromones[
                move
            ] = 0  # rendre le pheromone à 0 pour indiquer qu'il a été visité

            # Mise a jour poidRestant et gain de la solution
            poidrestant -= self.poids[move] * nb
            while poidrestant < 0:
                nb -= 1
                poidrestant += self.poids[move]

            sol[move] = nb
            gain += nb * self.benifices[move]

            # ajouter l'objet a visited
            visited.add(move)

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
    return M, res


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
    files = dir_files(dir_path)
    for file in files:

        n, w, v, b = read_data_3_single(file)
        nbObj.append(n)
        #############################################

        if meth == "bb":
            start_time = time.time()
            branch_and_bound_ukp(b, v, w)
        elif meth == "dp":
            start_time = time.time()
            dp_ukp(w, n, b, v)
        elif meth == "dog":
            start_time = time.time()
            density_ordered_greedy_ukp(b, v, w)
        elif meth == "wdogT":
            start_time = time.time()
            weighted_ordered_greedy_ukp(b, v, w, True)
        elif meth == "wdogF":
            start_time = time.time()
            weighted_ordered_greedy_ukp(b, v, w, False)
        elif meth == "ag":
            # random.seed(1)
            # max_it = 500
            # max_n = 10
            # N = 2500
            # NI = 100
            # Pc = 0.6
            # Pm = 0.4
            # stagnation = True
            start_time = time.time()
            AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation)
        elif meth == "hr":
            start_time = time.time()
            heuristic_arrondi(b, v, w)
        elif meth == "ac":
            start_time = time.time()
            densitySol = density_ordered_greedy_ukp(b, v, w)
            benifices = np.array(b)
            poids = np.array(v)
            utilites = poids / benifices
            n_ants = 100
            n_best = 10
            n_iterations = 10
            decay = 0.8

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
            colony.run(30, densitySol)[0]
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
):
    nbObj = []
    time_bb, time_dp, time_dog, time_wdogT, time_wdogF, time_hr, time_ag, time_ac = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    files = dir_files(dir_path)
    for file in files:

        n, w, v, b = read_data_3_single(file)
        nbObj.append(n)
        #############################################
        if bbcheck:
            start_time = time.time()
            branch_and_bound_ukp(b, v, w)
            time_bb.append(time.time() - start_time)
        if dpcheck:
            start_time = time.time()
            dp_ukp(w, n, b, v)
            time_dp.append(time.time() - start_time)
        if dogcheck:
            start_time = time.time()
            density_ordered_greedy_ukp(b, v, w)
            time_dog.append(time.time() - start_time)
        if wdogTcheck:
            start_time = time.time()
            weighted_ordered_greedy_ukp(b, v, w, True)
            time_wdogT.append(time.time() - start_time)
        if wdogFcheck:
            start_time = time.time()
            weighted_ordered_greedy_ukp(b, v, w, False)
            time_wdogF.append(time.time() - start_time)
        if hrcheck:
            start_time = time.time()
            heuristic_arrondi(b, v, w)
            time_hr.append(time.time() - start_time)
        if agcheck:
            random.seed(1)
            max_it = 500
            max_n = 10
            N = 2500
            NI = 100
            Pc = 0.6
            Pm = 0.4
            stagnation = True
            start_time = time.time()
            AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation)
            time_ag.append(time.time() - start_time)
        if accheck:
            start_time = time.time()
            densitySol = density_ordered_greedy_ukp(b, v, w)
            benifices = np.array(b)
            poids = np.array(v)
            utilites = poids / benifices

            n_ants = 100
            n_best = 10
            n_iterations = 10
            decay = 0.8
            alpha = 1
            beta = 2
            print(w)

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
            colony.run(30, densitySol)
            time_ac.append(time.time() - start_time)

        #############################################
    # data_gain = [gain_bb]
    # data_gain_rotated = list(rotated(data_gain))
    # data_time = [ time_bb]
    # data_time_rotated = list(rotated(data_time))

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
        df["Weighted Ordered Greedy (true)"] = time_wdogT
    if wdogFcheck:
        df["Weighted Ordered Greedy (false)"] = time_wdogF
    if hrcheck:
        df["Heuristic By Rounding"] = time_hr
    if agcheck:
        df["Genetic Algorithm"] = time_ag
    if accheck:
        df["Ant colony"] = time_ac

    df = df.rename(columns={"ojt": "index"}).set_index("index")

    chart = st.line_chart(df)


#######################################################
def statsCompGain(
    dir_path,
    bbcheck,
    dpcheck,
    dogcheck,
    wdogTcheck,
    wdogFcheck,
    hrcheck,
    agcheck,
    accheck,
):

    gain_bb, gain_dp, gain_dog, gain_wdogT, gain_wdogF, gain_hr, gain_ag, gain_ac = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    nbObj = []
    files = dir_files(dir_path)
    for file in files:

        n, w, v, b = read_data_3_single(file)
        nbObj.append(n)
        #############################################
        if bbcheck:

            gain_bb.append(branch_and_bound_ukp(b, v, w))

        if dpcheck:

            gain_dp.append(dp_ukp(w, n, b, v)[0])

        if dogcheck:

            gain_dog.append(density_ordered_greedy_ukp(b, v, w)[0])

        if wdogTcheck:
            gain_wdogT.append(weighted_ordered_greedy_ukp(b, v, w, True)[0])

        if wdogFcheck:
            gain_wdogF.append(weighted_ordered_greedy_ukp(b, v, w, False)[0])

        if hrcheck:

            gain_hr.append(heuristic_arrondi(b, v, w)[0])

        if agcheck:
            random.seed(1)
            max_it = 500
            max_n = 10
            N = 2500
            NI = 100
            Pc = 0.6
            Pm = 0.4
            stagnation = True

            gain_ag.append(AG(n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation)[0])
        if accheck:

            densitySol = density_ordered_greedy_ukp(b, v, w)
            benifices = np.array(b)
            poids = np.array(v)
            utilites = poids / benifices

            n_ants = 100
            n_best = 10
            n_iterations = 10
            decay = 0.8
            alpha = 1
            beta = 2

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

            gain_ac.append(colony.run(30, densitySol)[1])

        #############################################
    # data_gain = [gain_bb]
    # data_gain_rotated = list(rotated(data_gain))
    # data_time = [ time_bb]
    # data_time_rotated = list(rotated(data_time))

    df = pd.DataFrame(
        {
            "ojt": nbObj,
        }
    )
    if bbcheck:
        df["Branch and Bound"] = gain_bb
    if dpcheck:
        df["Dynamic Programing"] = gain_dp
    if dogcheck:
        df["Density Ordered Greedy"] = gain_dog
    if wdogTcheck:
        df["Weighted Ordered Greedy (true)"] = gain_wdogT
    if wdogFcheck:
        df["Weighted Ordered Greedy (false)"] = gain_wdogF
    if hrcheck:
        df["Heuristic By Rounding"] = gain_hr
    if agcheck:
        df["Genetic Algorithm"] = gain_ag
    if accheck:
        df["Ant colony"] = gain_ac

    df = df.rename(columns={"ojt": "index"}).set_index("index")
    chart = st.line_chart(df)


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
        st.components.v1.html(
            """
        <div><p style="color:white">

        ...
  </p> </div>
        """,
            width=600,
        )
        st.title("Methodes")
        st.components.v1.html(
            """
        <div><p style="color:white">

        ...
  </p> </div>
        """,
            width=600,
        )

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
                    self.objet = objet
                    self.val = val
                    self.pere = pere
                    self.poids = poids
                    self.m = m

            def first_solution(d,b,v,w,racine):
                M = 0
                n = racine
                for i in range(len(d)):
                    if w==0:
                        break
                    nb = int(w/v[d[i][1]])
                    M += nb * b[d[i][1]]
                    w -= nb * v[d[i][1]]
                    n = noeud(i+1,nb,n,w,M)
                return M,n

            def diviser(n,b,v,d):
                ind = d[n.objet][1]
                nb = int(n.poids/v[ind])
                return [noeud(n.objet+1,i,n,n.poids-i*v[ind],n.m+i*b[ind]) for i in range(nb,-1,-1)]

            def evaluer(n,d):
                if n.objet==len(d):
                    return n.m
                return n.m+n.poids*d[n.objet][0]

            def branch_and_bound_ukp(b, v, w):
                racine = noeud(0,-1,None,w,0)
                d = [(b[i]/v[i],i) for i in range(len(v))] #Densité
                d.sort(key=lambda x:x[0], reverse=True)
                M,res = first_solution(d,b,v,w,racine)
                na = diviser(racine,b,v,d)
                while len(na)!=0:
                    n = na.pop(0)
                    if n.poids==0:
                        if n.m>M:
                            M = n.m
                            res = n
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
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!" + file_path)
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr = branch_and_bound_ukp(b, v, w)

                arr = np.array(arr)
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + " in (time): " + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
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
            return k[w]
        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button("Upload file"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                arr = np.array(dp_ukp(w, n, b, v)[1])
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(dp_ukp(w, n, b, v)[0])
                    + " in (time): "
                    + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
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

        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button("Upload file"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr,poid = density_ordered_greedy_ukp(b, v, w)
                arr = np.array(arr)
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + " in (time): " + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
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
            d = [(v[i],i) for i in range(len(v))]
            d.sort(key=lambda x:x[0], reverse=not ordre_croissant)
            M = 0
            res = [0 for _ in range(len(d))]
            for i in range(len(d)):
                if w==0:
                    break
                nb = int(w/v[d[i][1]])
                M += nb * b[d[i][1]]
                w -= nb * v[d[i][1]]
                res[d[i][1]] = nb
            return M,res

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
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, arr = weighted_ordered_greedy_ukp(b, v, w, ordreCcheck)
                arr = np.array(arr)
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + " in (time): " + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
            if dir_path != "":
                if ordreCcheck:
                    stats(dir_path, "wdogF")
                else:
                    stats(dir_path, "wdogF")

    elif page == "Heuristic By Rounding":
        st.title("Heuristic By Rounding Algorithm")
        st.subheader("Description")
        st.write(
            "This approach involves sorting objects in ascending or descending order of their weight (or volumes). The list of objects thus sorted is scanned and for each type of object, we take as many copies of the latter as possible and so on until the complete scan of the list of objects. This heuristic very seldom gives good results. Often, it is better to sort objects according to the descending order of their weight because usually objects, with small weights, have a small gain"
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
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                poid, arr, res = heuristic_arrondi(b, v, w)
                arr = np.array(arr)
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :" + str(res) + " in (time): " + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "hr")

    elif page == "Recuit simulé ":
        st.title("Recuit simulé Algorithm")
        st.subheader("Description")
        st.write(
            "This approach involves sorting objects in ascending or descending order of their weight (or volumes). The list of objects thus sorted is scanned and for each type of object, we take as many copies of the latter as possible and so on until the complete scan of the list of objects. This heuristic very seldom gives good results. Often, it is better to sort objects according to the descending order of their weight because usually objects, with small weights, have a small gain"
        )

        st.subheader("Algorithm")

        st.code(
            """
          

        """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")

        col1, col2, col3 = st.beta_columns(3)

        if col2.button("Upload file"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                arr = np.array(heuristic_arrondi(b, v, w)[1])
                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(heuristic_arrondi(b, v, w)[2])
                    + " in (time): "
                    + str(dispTime - start_time)
                )

        st.subheader("Statistics")
        instances = ([str(i) for i in range(5, 210, 5)], 3)
        stats(instances, "hr")

    elif page == "Genetic Algorithm":
        st.title("Genetic Algorithm")
        st.subheader("Description")
        st.write(
            "This approach involves sorting objects in ascending or descending order of their weight (or volumes). The list of objects thus sorted is scanned and for each type of object, we take as many copies of the latter as possible and so on until the complete scan of the list of objects. This heuristic very seldom gives good results. Often, it is better to sort objects according to the descending order of their weight because usually objects, with small weights, have a small gain"
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
        max_it = col1.number_input("Insert a max_it", format="%d", value=0)
        max_n = col2.number_input("Insert a max_n", format="%d", value=0)
        N = col3.number_input("Insert a n", format="%d", value=0)
        NI = col1.number_input("Insert a ni", format="%d", value=0)
        Pc = col2.number_input("Insert a pc")
        Pm = col3.number_input("Insert a pm")
        stagnation = col1.checkbox("stagnation")

        if col3.button("Upload file"):

            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
                n, w, v, b = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()
                res, poid, arr = AG(
                    n, w, b, v, N, NI, Pc, Pm, max_it, max_n, stagnation
                )

                dispTime = time.time()
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(res)
                    + ", poid : "
                    + str(poid)
                    + " in (time): "
                    + str(dispTime - start_time)
                )

        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
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
                        
                        #Args:
                            # benifices (1D numpy.array): Les benifices de chaque objet.
                            # poids (1D numpy.array): Les poids de chaque objet.
                            # poids (1D numpy.array): L'utilité d'un objet de chaque objet.
                            # n_objets (int): Nombre d'objets
                            # W (int): La capacité du sac
                            # densitySol (liste): Solution generé par heuristique spécifique
                            # n_ants (int): Nombre de fourmis par iterations
                            # n_best (int): Nombre de meilleures fourmis qui déposent le pheromone
                            # n_iteration (int): nombre d'iteration
                            # decay (float): 1-Taux d'evaporation de pheromone
                            # alpha (int or float): Exposant dans le pheromone, Alpha grand donne plus de poid au pheromone
                            # beta (int or float): Exposant sur l'utilité, Beta grand donne plus de poid a l'utilité
                        # Example:
                            # ant_colony = AntColony(benifices,poids, utilites,n_objets, W,densitySol,n_ants, n_best, n_iterations, decay, alpha=1, beta=2)
                      
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
                        
                        # Args:
                            # n_candidats (int): Nombre de candidats pour construire les solutions
                            # densitySol (Gain:int,sol:liste,Poid:int): Solution generé par heuristique spécifique
                        # Example:
                            # best_sol = ant_colony.run(n_candidats,densitySol)
                        
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

                        print(self.gen_sol_gain(best_solution_all_time[0]))
                        print(self.gen_path_poid(best_solution_all_time[0]))
                        return best_solution_all_time

                    def spread_pheronome(self, solutions, n_best, best_solution):
                        
                        # Dépose le pheromone sur les n_best meilleures solutions

                        
                        sorted_solution = sorted(solutions, key=lambda x: x[1], reverse=True)
                        for sol, gain, poid in sorted_solution[:n_best]:
                            for i in sol:
                                self.pheromone += 0.00001 * gain

                    def gen_sol_gain(self, sol):
                       
                        # Calcul le gain d'une solution.
                        # Pas necessaire mais peut servir à verifier les résultats (test unitaire)

                        
                        total_fitness = 0
                        for i, ele in enumerate(sol):
                            total_fitness += self.benifices[i] * ele
                        return total_fitness

                    def gen_path_poid(self, sol):
                       
                        # Calcul le poid d'une solutions.
                        # Pas necessaire mais peut servir à verifier les résultats (test unitaire)

                       
                        total_fitness = 0
                        for i, ele in enumerate(sol):
                            total_fitness += self.poids[i] * ele
                        return total_fitness

                    def gen_all_solutions(self, n_candidats):
                        
                        # Generer toutes les solutions par les fourmis

                        
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
                        
                        # retourne La liste des candidats pour une solution

                        
                        pheromone = phero.copy()

                        pheromone[list(visited)] = 0
                        # rn.choices returns a list with the randomly selected element from the list.
                        # weights to affect a probability for each element
                        c = rn.choices(self.all_inds, weights=[p for p in pheromone], k=n_candidats)
                        i = len(c)
                        while i < n_candidats:
                            n = rn.randint(0, self.n_objets - 1)
                            if n not in visited:
                                c.append(n)
                                i += 1

                        return c, pheromone

                    # generer solution c'est bon
                    def gen_sol(self, start, n_candidats):
                       
                        # Construit la solution avec n_candidats et start comme premier objet

                        
                        sol = np.zeros(self.n_objets)
                        poidrestant = self.W
                        visited = set()  # liste des objets visité
                        # ajouter le premier objet
                        r = rn.randint(1, poidrestant // poids[start])
                        sol[start] = r
                        poidrestant -= poids[start] * r
                        gain = r * benifices[start]
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
                            candidats.pop(candidats.index(move))
                            pheromones[
                                move
                            ] = 0  # rendre le pheromone à 0 pour indiquer qu'il a été visité

                            # Mise a jour poidRestant et gain de la solution
                            poidrestant -= poids[move] * nb
                            while poidrestant < 0:
                                nb -= 1
                                poidrestant += poids[move]

                            sol[move] = nb
                            gain += nb * benifices[move]

                            # ajouter l'objet a visited
                            visited.add(move)

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

            """,
            language="python",
        )

        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)

        n_ants = col1.number_input("Insert a n_ants", format="%d", value=0)
        n_best = col2.number_input("Insert a n_best", format="%d", value=0)
        n_iterations = col3.number_input("Insert a n_iterations", format="%d", value=0)
        decay = col1.number_input("Insert a decay")
        alpha = col2.number_input("Insert a alpha", format="%d", value=0)
        beta = col3.number_input("Insert a beta", format="%d", value=0)
        if col2.button("Upload file"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()
            #file_path = filedialog.askopenfilename(master=root)
            file_path=easygui.fileopenbox()
            if file_path != "":
                st.text("imported !!")
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
                pdarr = pd.DataFrame(arr, columns=["Number of elements"])
                st.dataframe(pdarr.T)

                st.text(
                    "Result :"
                    + str(res)
                    + ", poid :"
                    + str(poid)
                    + " in (time): "
                    + str(dispTime - start_time)
                )
        colo1, colo2 = st.beta_columns((2, 1))
        colo2.text("")
        colo1.subheader("Time Comparaison")
        colo1.text("")
        if colo2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
            if dir_path != "":
                stats(dir_path, "ac")

    else:

        st.title("Comparaison")

        col1, col2 = st.beta_columns((2, 1))
        col2.text("")

        colo1, colo2 = st.beta_columns((2, 5))
        colo1.text("Select methods to compare time:")
        bbcheck = colo2.checkbox("Branch and Bound")
        dpcheck = colo2.checkbox("Dynamic Programing")
        dogcheck = colo2.checkbox("Density Ordered Greedy")
        wdogTcheck = colo2.checkbox("Weighted Ordered Greedy (true)")
        wdogFcheck = colo2.checkbox("Weighted Ordered Greedy (false)")
        hrcheck = colo2.checkbox("Heuristic By Rounding")
        agcheck = colo2.checkbox("Genetic Algorithm")
        accheck = colo2.checkbox("Ant colony")
        if col2.button("Select instances directory"):
            #root = tk.Tk()
            #root.focus_get()
            #root.withdraw()
            #root.focus_force()

            #dir_path = filedialog.askdirectory(master=root)
            dir_path=easygui.diropenbox()
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
                )
                st.text("")
                st.text("")
                st.subheader("Gain comparaison")
                st.text("")
                st.text("")
                statsCompGain(
                    dir_path,
                    bbcheck,
                    dpcheck,
                    dogcheck,
                    wdogTcheck,
                    wdogFcheck,
                    hrcheck,
                    agcheck,
                    accheck,
                )

        #


################################################################

main()
