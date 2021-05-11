import streamlit as st
import numpy as np
import pandas as pd
import tkinter as tk  
from tkinter import filedialog 
import time
##import matplotlib.pyplot as plt
#from tabulate import tabulate

import time
#b = [5, 4, 6, 2]  # Bénéfice
#v = [49, 33, 60, 32]  # Volume
#w = 130

####### Branch and bound ##########
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
    M, res = first_solution(d, b, v, w, racine)
    na = diviser(racine, b, v, d)
    while len(na) != 0:
        n = na.pop(0)
        if n.poids == 0:
            if n.m > M:
                M = n.m
                res = n
        elif int(evaluer(n, d)) > M:
            fils = diviser(n, b, v, d)
            fils_retenus = []
            for f in fils:
                evaluation = evaluer(f, d)
                if int(evaluation) > M:  # int(eval)>M : pour optimiser
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



##########  ########## 
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

#######################################################


def read_data_3(instance): #Ex : instance = 10, 15, 20, 25, ..., 205. 
    file = open("./generated/"+instance+".ukp")
    n = 0
    b = []
    v = []
    for i, line in enumerate(file):
        if i == 0:
            n = int(line.strip().split()[1])
        elif i == 1:
            w = int(line.strip().split()[1])
        elif i >= 3 and i < 3+n:
            v.append(int(line.strip().split()[0]))
            b.append(int(line.strip().split()[1]))
    file.close()
    return n,w,b,v

def read_data_3_single(instance): #Ex : instance = 10, 15, 20, 25, ..., 205. 
    file = open(instance)
    n = 0
    b = []
    v = []
    for i, line in enumerate(file):
        if i == 0:
            n = int(line.strip().split()[1])
        elif i == 1:
            w = int(line.strip().split()[1])
        elif i >= 3 and i < 3+n:
            v.append(int(line.strip().split()[0]))
            b.append(int(line.strip().split()[1]))
    file.close()
    return n,w,b,v


def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]



def stats(instances,meth):
    gain_bb = []
    time_bb = []
    for inst in instances[0]:
        
        n,w,b,v = read_data_3(inst)
        #############################################
        start_time = time.time()
        if meth=="bb":
            gain_bb.append(branch_and_bound_ukp(b,v,w)[0])
        elif meth=="dp":
            dp_ukp( w,n,b,v)
            #gain_bb.append()
        time_bb.append(time.time()-start_time)
        #############################################
    data_gain = [gain_bb]
    data_gain_rotated = list(rotated(data_gain))
    data_time = [ time_bb]
    data_time_rotated = list(rotated(data_time))
    
    headers_gain = ["Gain B&B"]
    headers_time = ["Time B&B"]
   
    c = np.array([time_bb])
    print(c.T)
    #print(tabulate(data_gain_rotated, headers=headers_gain) + "\n")
    #print(tabulate(data_time_rotated, headers=headers_time))
    #plot(data_gain[::-1], headers_gain)
    chart = st.line_chart( c.T)
    #plot(data_time[::-1], headers_time)


def statsComp(instances):
    gain_bb = []
    time_bb,time_dp = [],[]
    for inst in instances[0]:
        
        n,w,b,v = read_data_3(inst)
        #############################################
        start_time = time.time()
        gain_bb.append(branch_and_bound_ukp(b,v,w)[0])
        time_bb.append(time.time()-start_time)
        start_time = time.time()
        dp_ukp( w,n,b,v)
        time_dp.append(time.time()-start_time)
        
        
        #############################################
    #data_gain = [gain_bb]
    #data_gain_rotated = list(rotated(data_gain))
    #data_time = [ time_bb]
    #data_time_rotated = list(rotated(data_time))
    

    df = pd.DataFrame({

  'ojt': [i for i in range(5,210,5)],
  'Branch and Bound': time_bb,
  'Dynamic Programing ': time_dp

})
    df = df.rename(columns={'ojt':'index'}).set_index('index')
    #c = np.array([time_bb,time_dp])
    #print(c.T)
    #print(tabulate(data_gain_rotated, headers=headers_gain) + "\n")
    #print(tabulate(data_time_rotated, headers=headers_time))
    #plot(data_gain[::-1], headers_gain)
    chart = st.line_chart( df)
    #plot(data_time[::-1], headers_time)
    


############ UI ############
def main():

    page = st.sidebar.selectbox(
        "Choose a algorithm to run :", ["Home", "Branch and bound","Dynamic Programing", "Comparaison"]
    )

    if page == "Home":
        st.title("Welcome to knapsack problem solver ")
        st.components.v1.html(
            """
        <div><p style="color:white">

        The knapsack problem is a problem in combinatorial optimization: Given a set of items, each 
        with a weight and a value, determine the number of each item to include in a collection so 
        that the total weight is less than or equal to a given limit and the total value is as large 
        as possible. It derives its name from the problem faced by someone who is constrained by a 
        fixed-size knapsack and must fill it with the most valuable items. The problem often arises 
        in resource allocation where the decision makers have to choose from a 
        set of non-divisible projects or tasks under a fixed budget or time constraint, respectively
  </p> </div>
        """,
            width=600,
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

    elif page == "Branch and bound":
        st.title("Branch and bound algorithem")
        st.subheader("Pseudo Algorithm")
         
        st.code(
            """
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
        """,
            language="python",
        )
        st.subheader("Import data (Apply the algorithm on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button('Upload file'):  
            root = tk.Tk() 
            root.focus_get() 
            root.withdraw() 
            root.focus_force() 
            file_path = filedialog.askopenfilename(master=root) 
            if file_path!="":
                st.text("imported !!")
                n,w,b,v = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()              
                arr = np.array(branch_and_bound_ukp(b, v, w)[1])
                dispTime=time.time()
                pdarr= pd.DataFrame(arr,columns=['Number of elements'])
                st.dataframe(pdarr.T)
                print(dispTime-start_time)
                st.text("Result :"+str(branch_and_bound_ukp(b, v, w)[0])+" in (time): "+str(dispTime-start_time))

        
       
        st.subheader("Statistques")
        instances = ([str(i) for i in range(5,210,5)], 3)
        stats(instances,"bb")

    elif page == "Dynamic Programing":
        st.title("Dynamic Programing algorithem")
        st.subheader("Pseudo Algorithm")
         
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
        if col2.button('Upload file'):  
            root = tk.Tk() 
            root.focus_get() 
            root.withdraw() 
            root.focus_force() 
            file_path = filedialog.askopenfilename(master=root) 
            if file_path!="":
                st.text("imported !!")
                n,w,b,v = read_data_3_single(file_path)
                st.subheader("Solution")
                start_time = time.time()              
                arr = np.array(dp_ukp( w,n,b,v)[1])
                dispTime=time.time()
                pdarr= pd.DataFrame(arr,columns=['Number of elements'])
                st.dataframe(pdarr.T)
                print(dispTime-start_time)
                st.text("Result :"+str(dp_ukp( w,n,b,v)[0])+" in (time): "+str(dispTime-start_time))

        
       
        st.subheader("Statistques")
        instances = ([str(i) for i in range(5,210,5)], 3)
        stats(instances,"dp")
    else:

        st.title("Comparaison")
       
        st.subheader("Import data (Compare the algorithms on a signle file)")
        col1, col2, col3 = st.beta_columns(3)
        if col2.button('Upload file'):  
            root = tk.Tk() 
            root.focus_get() 
            root.withdraw() 
            root.focus_force() 
            file_path = filedialog.askopenfilename(master=root) 
            if file_path!="":
                st.text("imported !!")
                colo1, colo2 = st.beta_columns(2)
               
                n,w,b,v = read_data_3_single(file_path)
                colo1.subheader("Solution Branch and Bound")

                start_time = time.time()              
                branch_and_bound_ukp(b,v,w)
                dispTime=time.time()
                
            
                print(dispTime-start_time)
                colo1.text("Result :"+str(branch_and_bound_ukp(b,v,w)[0])+" in (time): "+str(dispTime-start_time))

                colo2.subheader("Solution Dynamic Programing ")
                start_time = time.time()              
                dp_ukp( w,n,b,v)
                dispTime=time.time()

                colo2.text("Result :"+str(dp_ukp( w,n,b,v)[0])+" in (time): "+str(dispTime-start_time))
              
                print(dispTime-start_time)
              

                

                

        
       
        st.subheader("Statistques & Comparaison")
        instances = ([str(i) for i in range(5,210,5)], 3)
        statsComp(instances)
################################################################

main()
