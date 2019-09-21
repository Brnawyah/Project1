# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:09:18 2019

@author: umroot
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:40:00 2019

@author: umroot
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 00:07:27 2019

@author: Raheel
"""
import pandas as pd

import numpy as np
from random import randint
from collections import Counter 
from kmodes.kmodes import KModes
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from functools import reduce
import graphAMatrix as graph 
#from matplotlib import pyplot as plt
#from pyclustering.cluster.birch import birch, measurement_type
from scipy.cluster.hierarchy import  linkage 
from sklearn.cluster import Birch
import networkx as nx
from scipy.cluster.hierarchy import cut_tree
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report 



class Modul_info:
    def __init__(self, module_num,pos_objects,bnd_onbjects,not_clusters,y_postive,t_bnd,discover_cluster):
        self.module_num= module_num
        self.pos_objects=pos_objects
        self.bnd_onbjects=bnd_onbjects
        self.not_clusters=not_clusters
        self.y_postive=y_postive
        self.t_bnd=t_bnd
        self.discover_cluster=discover_cluster
        
class clusters:
    
    def __init__(self, numcluster, mode, cluster_size, objects,assigned_objects):
        self.numcluster=numcluster
        self.mode=mode
        self.cluster_size= cluster_size
        self.objects= objects
        self.assigned_objects=assigned_objects
        
    def add_objects(self, obj_index):
        self.assigned_objects.append(obj_index)
    def update(self):
        self.objects= self.objects + self.assigned_objects
        self.assigned_objects=[]
    

def BuildadjacencyMatrix(allNodes):
        if len(allNodes) >= 1:
           adjacency_matrix = np.zeros(shape=(len(allNodes),len(allNodes)))
           adjacency_matrix1 = np.zeros(shape=(len(allNodes),len(allNodes)))
           for i in range(len(allNodes)):
                vertex=allNodes[i]
                for j in range(len(allNodes)):
                    node=allNodes[j]
                    res_s1=vertex.objects
                    res_s2=node.objects
                    
                    simm=intersection1(res_s1, res_s2)
                    adjacency_matrix[i,j] = int(len(simm))
                    res_s1 = vertex.node_mode
                    res_s2 = node.node_mode
                
                    sim=intersection1(res_s1, res_s2)
                    adjacency_matrix1[i,j] = int(len(sim))    

           return adjacency_matrix ,adjacency_matrix1
        else:
            return dict() 
def Bulid_Objects_sim_matrix(allNodes):
    count_accurences=0
    object_matrix = np.zeros(shape=(47,47))
    for i in range(1,48):
        for j in range(1,48):
            if i !=j:
                for node in allNodes:
                    if  i and j in  node.objects :
                        count_accurences +=1
                object_matrix[i-1,j-1]=count_accurences
                count_accurences=0
                
                
    return(object_matrix)
                
                

        
def adjacencyMatrix(allNodes):
        if len(allNodes) >= 1:
           G = nx.Graph()
           G1=nx.Graph()
           adjacency_matrix = np.zeros(shape=(len(allNodes),len(allNodes)))
           for i in range(len(allNodes)):
                vertex=allNodes[i]
                for j in range(len(allNodes)):
                    node=allNodes[j]
                    if vertex.module_number != node.module_number:
                        res_s1 = vertex.node_mode
                        res_s2 = node.node_mode
                        #res_s1=vertex.objects
                        #res_s2=node.objects
                        #att_s1=vertex.features
                        #att_s2=node.features
                        s1= set(res_s1)
                        s2=set(res_s2)
                        #ss1= set(att_s1)
                        #ss2=set(att_s2)
                        simm=intersection1(res_s1, res_s2)
                        #simm_attribute=intersection1(att_s1, att_s2)
                        
                        sim=float(len(simm)) / float(len(s1.union(s2)))
                        
                       # sim_sim_attribute= float(len(simm_attribute)) / float ((len(ss1.union(ss2)))+0.1)
                        
                        #sim_sim_attribute= float(len(simm_attribute)) / 16.0
                        #print(sim_sim_attribute)
                        #adjacency_matrix[i,j] = float( sim )
                        adjacency_matrix[i,j] = int(len(simm))
                        weight= int(1.0-(sim ))
                        G.add_edge(vertex.node_index,node.node_index,weight=weight)
                        G.add_edge(i,j)
                        G.node[i]['node_value'] = weight

                    else:
                       adjacency_matrix[i,j] = 0.0 
                       #G1.node[i]['node_value'] = int(1.0)

           return adjacency_matrix , G,G1
        else:
           
            return dict() 

def skitleanBirch():
        data=pd.read_csv("soy_rock.csv",header=None)
        X = data.values.tolist()
        randomm=randint(5,20)
        
        brc = Birch(branching_factor=randomm, n_clusters=4, threshold=0.1,
        compute_labels=True)
        brc.fit(X) 
        pred=brc.predict(X)
        return pred

def findPartition(posId_list, posPredict_list):
    result = {i:[] for i in posPredict_list}
    for i,j in zip(posPredict_list,posId_list):
        result[i].append(j)
    return result
    

    
          
def intersectionMultiList(multiList):
    intersect_result=list(reduce(set.intersection, [set(item) for item in multiList ]))
    return intersect_result
    
    
    
    
def most_frequent(List): # find the mode of group of data (cluster)
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 

def intersection1(ind_cond, ind_deci): 
    temp = set(ind_deci) 
    lst3 = [value for value in ind_cond if value in temp] 
    return lst3 


def do_clustering(newDF,number_cluster):
        clusters=[]
        randomm=randint(2,10)
        rand_clusters=randint(number_cluster,2 *number_cluster)
        km = KModes(n_clusters=number_cluster, init='random', n_init=randomm, verbose=0)
        km.fit_predict(newDF)
        clusters=list(km.labels_) 
        print(len(clusters))        
        return clusters

def PVRS (equavlance_cond, equavlance_deci,Beta):
    lower_of_decisionX=[]
    upper_of_decisionX= []
    BND_region=[]
    NG_region=[]
    #Module_BND=[]
    Beta=Beta
    postiveReg=[]
    for decition_value in equavlance_deci:
        #print("---------decistion X------------------" ,decition_value)
        for element_equavlance in equavlance_cond:
            intersect_result=intersection1(element_equavlance, decition_value)

            c= 1.0 - (float(len(intersect_result))/ float(len(element_equavlance)))
            if c<= Beta:
                lower_of_decisionX.append(element_equavlance)
            b= 1.0 - Beta
            if c < b :
                upper_of_decisionX.append(element_equavlance)
               
        lower=sum(lower_of_decisionX,[])
        upper=sum(upper_of_decisionX,[])
        BND= list(set(upper) - set(lower))
        NGR= list(set(listt)- set(BND) - set(lower))
        BND_region.append(BND)
        postiveReg.append(lower)
        NG_region.append(NGR)
        lower_of_decisionX=[]
        upper_of_decisionX=[]

    POS= sum(postiveReg,[])
    BBND=sum(BND_region,[])
    NGRR=sum(NG_region,[])
    
    return POS ,BBND,NGRR

def  select_rows (rows_index):
    original_data = pd.read_csv(data_set_name)
    selected_rows= original_data.iloc[rows_index,:]
    modes_= (selected_rows.apply(most_frequent).tolist())
    return modes_
    
    

def printGraph(g):
    """ Function to print a graph as adjacency list and adjacency matrix. """
    return g.adjacencyMatrix()
    #return str(g.adjacencyList()) + '\n' + '\n' + str(g.adjacencyMatrix())


def check_DiversityARI(pervious_labels, current_pred_label):
    for lable in pervious_labels:
        cehclARI=adjusted_rand_score(lable,current_pred_label) 
        if cehclARI == 1.0 :
            return True 
        else:
            return False
        
        
def decode_data():
    for col in dfVPRST:
        lsit_unique=dfVPRST[col].unique()
        for uq in lsit_unique:
            dfVPRST[col].replace(uq,(str(col)+str(uq)),inplace=True)
    dfVPRST.to_csv("data_set_name.csv",index=False)
    
# KNN cant be used for more than 2 classes         
def findKKN(X,y,pred):
    np.asarray(X)
    np.asarray(y)
    np.asarray(pred)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X,y)
    print("KNN")
    print(neigh.predict(pred))
    
def filter_boundary(list_boundery, list_postive):
    new_list=[x for x in list_postive]
    for i in list_boundery:
        if i in new_list:
            new_list.remove(i)
    return new_list
    
def compute_nearest_cluster(mode, Obj) :
      sim= len(Obj.intersection(mode)) / len(Obj.union(mode))
      return float(sim)
    
"""
This method is used to find the top best members and rank them 
form highest to lower 
return: total diversity of the entire ensemble clustering, and sorted dict
"""
def find_SNMI(ensemble_member):
  SNMI=0.0
  total_diversity_degree=0.0
  dict_SNMI={}
  for i in range (0, len(ensemble_member)):
    for j in range(0,len(ensemble_member)):
      if i !=j :
        SNMI += normalized_mutual_info_score(ensemble_member[i],ensemble_member[j])
    total_diversity_degree += SNMI
    dict_SNMI[i]= SNMI / (float(len(ensemble_member)-1))
    SNMI=0.0
  return total_diversity_degree ,(sorted(dict_SNMI,key=(lambda key:dict_SNMI[key]),reverse=True))
  # this return sorted dic
  #return total_diversity_degree ,(sorted(dict_SNMI.items(),key=lambda x:x[1],reverse=True))

"""
"soy_RST.csv"
"classSoy.csv"
----------------
voteRST.csv
classVote.csv
--------------------
lung_RST.csv"
"classlung.csv"
-----------------
zoo_RST.csv
classZoo.csv
----------
2
lyoumph_RST.csv
lyoumphClass.csv
-----------
3
dermatologyClass.csv
dematology_RST.csv
---------
3
"balance_RST.csv"
"balance_class.csv"
"""
G = nx.Graph()
data_set_name="soy_RST.csv"
file_true_classes="classSoy.csv"
dfVPRST = pd.read_csv(data_set_name)
trueVote= pd.read_csv(file_true_classes,header=None)
true= sum (trueVote.values.tolist(),[])

attribute=list(dfVPRST.columns.values)
dataSize= float(len(dfVPRST.index))
size_data=len(dfVPRST.index)
highestK= 0.0
posObjects=[]
posFeatures=[]
y_pred=[]
t=[]
t_bnd=[]
best_Mdule_features=[]
all_postive_Module=[]
all_boundry_Module=[]
uncluster=[]
number_cluster=4
candidate_class=""       
bndObjects=[]
final_voting=[]
possible_postive=[]
module_number=0
allNodes=[]
node_index=0
modules_accuracy=[]
listt= list(range(int (dataSize)))
check_diversity =[]
stust=[]
ARR=[]
counter=-1
Statuse= True
pp=[]
bb=[]
pos_counter=0
module_clusters={}
dict_discover={}
discover_cluster=0
decode_data() # this step to deal with categorical data in numbers form
#print(dfVPRST)
"""
Pre-processing 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step 1 check diverity >>>>>>>>>>>>>>>>>>>>>
"""

#>>>> 1.1 generate number of diverse clustering solution        
while ( len (check_diversity) <10 or counter <10):
         counter +=1
         if counter==2:
             #p#reVote= pd.read_csv("zoo.csv",header=None)
             #labels_pred= sum (trueVote.values.tolist(),[])
             check_diversity.append(true)
             ARI=normalized_mutual_info_score(true,true)
             ARR.append(ARI)

         else:
             print(len(true))
             labels_pred= do_clustering(dfVPRST,number_cluster)
             ARI=normalized_mutual_info_score(true,labels_pred)
             if len (check_diversity) == 0:
                 ARR.append(ARI)
                 check_diversity.append(labels_pred)
           
             else:
                 print(ARR, ARI)
                 if (ARI  not in ARR ):
                     ARR.append(ARI)
                     check_diversity.append(labels_pred) 
#
## """   
## Find the degree of diversity for the entire ensemble clustering
## Find the SNMI of each clustering solution 
## """
#total_diver,  dict_SNMI_degree= find_SNMI(check_diversity)
#print(dict_SNMI_degree)                                         
## """
#
## >>>>>>>>>>>>>>>>>>>>>>>>> Step 2 find POS, BND, NGR for each module>>>>>>>>>>>>>>
## """ 
labels_df = pd.DataFrame(check_diversity)
labels_df.to_csv("Partition1.csv",index=False)
average_NMI=0.0
pp_sixe=0.0
y_pred_pos=[]
##Features=['14', '13']
Results_partition= pd.read_csv("Partition1.csv")

ensemble_P= Results_partition.values.tolist()

P=pd.DataFrame(ensemble_P)
list_P=P.values.tolist() 

Ensemble_info=[]
counter_DD=0
avarage_DD=0.0
avarage_module_NMI=0.0
for labels_pred in list_P:
       labaledf = pd.DataFrame(labels_pred)
       print(labels_pred)
       #labaledf = pd.DataFrame(labels_pred) # covert the clustering label to df to add 
       dfVPRST['class']=labels_pred # add class coulmn to the df
       dfVPRST['id'] = dfVPRST.index# add index to df 
       NMI=normalized_mutual_info_score(true,labels_pred)
       #print("Mdule accuracy before selection", NMI)
       avarage_module_NMI +=NMI
       print("Module score before selection", NMI)
       equavlance_deci = ( dfVPRST.groupby("class")['id'].apply(list)).tolist() 
       print(module_number)
       
       Beta=0.3
       dfVPRST.pop("class")
       
       for item in attribute: # find which attributes more infultian in consturcting the clu
               equavlance_cond = (dfVPRST.groupby(item)['id'].apply(list)).tolist()
               #print(equavlance_cond)
               Module_pos,Module_BND, NG=PVRS(equavlance_cond,equavlance_deci,Beta)
               K= float(len(Module_pos))/ dataSize
               if K >=0.5 :
                    avarage_DD += K
                    counter_DD +=1
                    print("DD", K)
                    pp_sixe +=1.0
                    posObjects.append(Module_pos) # postive object for the module 
                    bndObjects.append(Module_BND) # boundary for the model for each attributes
               #find the intersetion of all postive 
       
       if len(posObjects)!= 0: 
           Pos_Module_attribute= intersectionMultiList(posObjects)
       #Pos_Module_attribute=set().union(*posObjects) # we took the union of all postive  coz intersection might not find all 
       #print("Pos_Module_attribute",Pos_Module_attribute )
       if len(bndObjects)!= 0:
           BND_Module_attribute=intersectionMultiList(bndObjects)
           #BND_Module_attribute=set().union(*bndObjects)
           
           #print("BND_Module_attribute",BND_Module_attribute)
           #intersetPOSandBND=intersection1(Pos_Module_attribute,BND_Module_attribute)
           #print("intersection POS, BND",intersetPOSandBND )
           BND_regions= list((set(BND_Module_attribute) - set(Pos_Module_attribute)))
           all_boundry_Module.append(list(BND_regions))
           print("BND regions",BND_regions)
       
       # if there is an object that is postive in most of the time and boundery some time 
       # keep it as postive, other wise delete it form postive
       
       for i in range(0, size_data):
              if i in list(Pos_Module_attribute):
                 y_pred_pos.append(labels_pred[i])
                 t.append(true[i])
              else:
                 if i  in list(BND_regions):
                    uncluster.append(i)
                    t_bnd.append(true[i])
       print(len(labels_pred),len(t), len(y_pred_pos))
      
       
       NMI_pos=normalized_mutual_info_score(t,y_pred_pos)
       MNI_boundery=normalized_mutual_info_score(t_bnd,uncluster)
       if len(set(y_pred_pos))> discover_cluster:
           discover_cluster = len(set(y_pred_pos))
           if discover_cluster in dict_discover:
               dict_discover[discover_cluster] += 1
           else:
               dict_discover[discover_cluster]=0
           # do clustering to find the missing ones
           
           
       print("dicover clusters",set(y_pred_pos),":",len(set(y_pred_pos)))
       print(sorted(y_pred_pos))
       if len(list(Pos_Module_attribute))!=0 :
           print("Mdule postive Accuracy", NMI_pos)
           print("Mdule BND Accuracy", MNI_boundery)
           pos_counter +=1
           average_NMI += NMI_pos
           E_info=Modul_info(module_number,Pos_Module_attribute,BND_Module_attribute,BND_regions,y_pred_pos,t_bnd,discover_cluster)
       Ensemble_info.append(E_info)
           #module_clusters=findPartition(Pos_Module_attribute,y_pred_pos) 
       
#       for key, value_objects in module_clusters.items():
#            if len(value_objects) >1:
#                node_mode=select_rows(value_objects)
#                avarage_DD = avarage_DD / float(len(attribute))
#                node =graph.Vertex(value_objects, module_number,node_index,node_mode,avarage_DD )
#                G.add_node(node)
#                allNodes.append(node)
#                node_index +=1 
       
    
       final_voting.append(y_pred_pos)
       possible_postive.append(uncluster)
       posFeatures=[]
       y_pred=[]
       t=[]
       t_bnd=[]
       uncluster=[]
       module_number +=1
       POS_module=[]
       y_pred_pos=[]
       posObjects=[]
       bndObjects=[]
       avarage_DD=0.0
       counter_DD=0.0
       print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
       print()
# #
# #
       
       

"""
This Step is required if we want to feed back the results into calssifer
"""
#clustered_data=intersectionMultiList (posObjects)
#clustered_data = set().union(*all_postive_Module)
#print("cluster Data  intersection of all postive moduls",clustered_data)
#NEG_intersectionBND= intersectionMultiList (bndObjects)
#print("Negative region ", NEG_intersectionBND)
#BND_regions= (set(listt) - set(clustered_data) - set(NEG_intersectionBND) )
#print("BND_regions",BND_regions)

y_pred=[]
t=[]

#filttering=list(set(sum(all_boundry_Module,[])))
##print("***************************************",filttering)
#print(len(filttering))
#allNodes_onjects_list=[]
#node_counter=0
#for node in allNodes:
#    #node.remove_object(filttering)
#    allNodes_onjects_list.append(node.objects)
#    node_counter +=1
#
#matrix_adjecny, graphnetwork ,GG=adjacencyMatrix(allNodes)
#labaledf = pd.DataFrame(matrix_adjecny)
#labaledf.to_csv("out1.csv")
#
#
#linked=linkage( np.array(matrix_adjecny), 'complete')
#postive_AHC_results= sum((cut_tree(linked, n_clusters = 3).T).tolist(),[]) 
#
#d=findPartition(allNodes_onjects_list,postive_AHC_results) 
#newD = {k:list(set(sum(v,[]))) for k, v in d.items()}
#print(newD)
#ensemble_clustering=[]
#true_ensemble=[]
#for i in range(0,int(dataSize)):
#    for key, value in newD.items():
#        if i in value:
#            ensemble_clustering.append(key)
#            true_ensemble.append(true[i])
#            
#            
#print("ensemble_clustering",ensemble_clustering)
#print(len(ensemble_clustering),len(true_ensemble))
#print("Combined Result",normalized_mutual_info_score(true_ensemble,ensemble_clustering))
#print("number of discovered clusters",discover_cluster)
#print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#print("average pos NMI=",average_NMI / float(pos_counter))
#print("avarage_module_NMI=",avarage_module_NMI / float (len(list_P)))
#print("participate partions",pos_counter)
#print("size of ensmble size",len (check_diversity))
##counter=0
##for node in allNodes:
##    counter +=1
##    print (counter, ":", node.objects)
allNodes=[]
module_clusters={}
node_index =0
for m in Ensemble_info:
    print(m.module_num)
    print(m.pos_objects)
    print(m.y_postive)
    print(m.not_clusters)
    print(m.t_bnd)
    module_clusters=findPartition(m.pos_objects,m.y_postive)
    print(module_clusters)
    for key, value_objects in module_clusters.items():
            if len(value_objects) >1:
                node_mode=select_rows(value_objects)
                avarage_DD = avarage_DD / float(len(attribute))
                node =graph.Vertex(value_objects, m.module_num,node_index,node_mode,avarage_DD )
                allNodes.append(node)
                node_index +=1 
            if  number_cluster - m.discover_cluster == 1:
                    node =graph.Vertex(m.not_clusters, m.module_num,node_index,node_mode,avarage_DD )
                    allNodes.append(node)
                    node_index +=1 
    print("-----------------------")
matrix_adjecny, matrix_adjecny1 =BuildadjacencyMatrix(allNodes)
labaledf = pd.DataFrame(matrix_adjecny)
labaledf.to_csv("C:\\Users\\umroot\\Downloads\\ClusterEnsembleV20\\ClusterEnsemble-V2.0\\out.csv",header=False, index=False) 
labaledf = pd.DataFrame(matrix_adjecny1)
labaledf.to_csv("C:\\Users\\umroot\\Downloads\\ClusterEnsembleV20\\ClusterEnsemble-V2.0\\out1.csv",header=False, index=False)    

for node in allNodes:
    counter +=1
    print (counter, ":", node.objects)       
        
object_matrix=Bulid_Objects_sim_matrix(allNodes)
simi_matrix = pd.DataFrame(object_matrix)
simi_matrix.to_csv("C:\\Users\\umroot\\Downloads\\ClusterEnsembleV20\\ClusterEnsemble-V2.0\\sim_mat.csv",header=False, index=False)   
max_key = max(dict_discover, key=lambda k: dict_discover[k])
print("dict_discover",max_key)