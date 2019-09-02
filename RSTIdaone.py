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
from matplotlib import pyplot as plt
from pyclustering.cluster.birch import birch, measurement_type
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster
from sklearn.cluster import Birch
import networkx as nx
from scipy.cluster.hierarchy import cut_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 



class Modul_info:
    def __init__(self, num_clus,pos_objects,bnd_onbjects,not_clusters):
        self.num_clus= num_clus
        self.pos_objects=[]
        self.bnd_onbjects=bnd_onbjects
        self.not_clusters=not_clusters
        
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
    def update_mode(self,new_mode):
        print("mode has updated")
        self.mode=new_mode
   

def adjacencyMatrix(allNodes):
        if len(allNodes) >= 1:
           G = nx.Graph()
           adjacency_matrix = np.zeros(shape=(len(allNodes),len(allNodes)))
           for i in range(len(allNodes)):
                vertex=allNodes[i]
                for j in range(len(allNodes)):
                    node=allNodes[j]
                    if vertex.module_number != node.module_number:
                        s1= set(vertex.objects)
                        s2=set(node.objects)
                        if len(list(s1)) or len(list(s2)) !=0:
                            sim=len(s1.intersection(s2)) / len(s1.union(s2))
                            adjacency_matrix[i,j] = sim 
                            G.add_edge(vertex.node_index,node.node_index,length = 1,weight=sim)
                        else:
                            adjacency_matrix[i,j] = 0
                            #G.add_edge(vertex.node_index,node.node_index,length = 100,weight=0.0)
                            
                    else:
                       adjacency_matrix[i,j] = 0 
           return adjacency_matrix , G
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
        randomm=randint(20,100)
       
        km = KModes(n_clusters=number_cluster, init='Huang', n_init= randomm, verbose=0)
        km.fit_predict(newDF)
        clusters=list(km.labels_)         
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


def printGraph(g):
    """ Function to print a graph as adjacency list and adjacency matrix. """
    return g.adjacencyMatrix()
    #return str(g.adjacencyList()) + '\n' + '\n' + str(g.adjacencyMatrix())

def doBirchClustering (i):
    #data=pd.read_csv("soy_rock.csv",header=None,dtype='int')
    #data=pd.read_csv("votepreporcess.csv",header=None,dtype='int')
    data=pd.read_csv("lung_number.csv",header=None,dtype='int')
    data_size=len(data.index)
    sample = data.values.tolist()
    
    #i=12
    #print( "random initlization i in Brich ", i )
    birch_instance = birch(sample, 7,branching_factor = i, max_node_entries = i, initial_diameter = 0.1)
    birch_instance.process()
    clusters = birch_instance.get_clusters()
    value=[]
    for data in range (0,data_size):
        for cluster_number in range(0,len(clusters)):
            if data in clusters[cluster_number]:
                value.append(cluster_number)
                break
    return value
def check_DiversityARI(pervious_labels, current_pred_label):
    for lable in pervious_labels:
        cehclARI=adjusted_rand_score(lable,current_pred_label) 
        if cehclARI == 1.0 :
            return True 
        else:
            return False
# KNN cant be used for more than 2 classes         
def findKKN(X,y,pred):
    np.asarray(X)
    np.asarray(y)
    np.asarray(pred)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X,y)
    print("KNN")
    print(neigh.predict(pred))
    
    
def compute_nearest_cluster(mode, Obj) :
      sim= len(Obj.intersection(mode)) / len(Obj.union(mode))
      return float(sim)
    



"""
"soy_RST.csv"
"soyclass.csv"
----------------
voteRST.csv
classVote.csv
--------------------
lung_RST.csv"
"classlung.csv"
"""
G = nx.Graph()
data_set_name="zoo_RST.csv"
file_true_classes="classZoo.csv"
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
best_Mdule_features=[]
all_postive_Module=[]
all_boundry_Module=[]
uncluster=[]
number_cluster=7
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
counter=0
Statuse= True
"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step 1 check diverity >>>>>>>>>>>>>>>>>>>>>
"""
for i in range (0, 10):
        print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", i,"^^^^^^^^^^^^^^^^^")
        module_number +=1
        while (Statuse == True or len (check_diversity) <5):
            counter +=1
            if i ==30:
                randomm=randint(5,20)
                labels_pred= doBirchClustering(12)
#                dfObj = pd.DataFrame(labels_pred) 
#                dfObj.to_csv("out1.csv")
                #labels_pred=skitleanBirch()
       
            else:
                #if counter % 2 == 0:
                if counter==2 or counter == 5:
                    randomm=randint(1,3)
                    #labels_pred= doBirchClustering(12)
                    labels_pred= do_clustering(dfVPRST,number_cluster)
                else:
                    labels_pred= do_clustering(dfVPRST,number_cluster)
                if len (check_diversity) == 0:
                    ARI=adjusted_rand_score(true,labels_pred) 
                    ARR.append(ARI)
                    check_diversity.append(labels_pred)
                    Statuse = False
                    print(  ARI)
                    
                else:
                    
                    ARI=adjusted_rand_score(true,labels_pred) 
                    print(ARR, ARI)
                    if (ARI in ARR ):
                        Statuse = True
                        stust.append(Statuse)
                    else:
                        Statuse = False
                        ARR.append(ARI)
                        check_diversity.append(labels_pred)
                        stust.append(Statuse)
                        
                        
                 
"""
>>>>>>>>>>>>>>>>>>>>>>>>> Step 2 find POS, BND, NGR for each module>>>>>>>>>>>>>>
"""   
for labels_pred in check_diversity:
       labaledf = pd.DataFrame(labels_pred) # covert the clustering label to df to add 
       dfVPRST['class']=labels_pred # add class coulmn to the df
       dfVPRST['id'] = dfVPRST.index# add index to df 
       equavlance_deci = ( dfVPRST.groupby("class")['id'].apply(list)).tolist() 
       Module_pos=[]
       ARI=adjusted_rand_score(true,labels_pred)
       print("Module ARI",ARI)
       if ARI > 0.60:
           Beta= 0.3
           DD=0.75
       else:
           Beta=0.1
           DD=0.40
       dfVPRST.pop("class")
       for item in attribute: # find which attributes more infultian in consturcting the clu
            equavlance_cond = (dfVPRST.groupby(item)['id'].apply(list)).tolist()
            Module_pos,Module_BND, NG=PVRS(equavlance_cond,equavlance_deci,Beta)
            K= float(len(Module_pos))/ dataSize
            if K >=DD :
                
                print("dependency degree", K)
                pp=intersection1(Module_pos, listt) 
                bb= intersection1(Module_BND, listt) 
                posObjects.append(pp)
                posFeatures.append(item)
                bndObjects.append(bb)
       if len(posObjects) !=0:
            POS_module= intersectionMultiList(posObjects)
            boundry_modul=intersectionMultiList(bndObjects)
            best_Mdule_features.append(posFeatures)
            print("POS", POS_module )
            print(" size", len(POS_module))
            print ("Module Attribute", best_Mdule_features)
       
            all_postive_Module.append(list(POS_module))
            all_boundry_Module.append(list(boundry_modul))        
            for i in range(0, len(labels_pred)):
              if i in list(POS_module):
                  y_pred.append(labels_pred[i])
                  t.append(true[i])
              else:
                  if i not in list(boundry_modul):
                      uncluster.append(i)
            ARI=adjusted_rand_score(t,y_pred)  
            NMI=normalized_mutual_info_score(t,y_pred)
            discoverd_clusters= set (y_pred)
            module_clusters=findPartition(POS_module,y_pred) 
            for key, value in module_clusters.items(): 
                node_name= str(module_number) + str(key)
                node =graph.Vertex(node_name, value, module_number,node_index)
                G.add_node(node)
                allNodes.append(node)
                node_index +=1
            print("Module postive  Accuracy NMI", NMI)
            print("Module postive  Accuracy ARI ", ARI)
            print(y_pred) 
            final_voting.append(y_pred)
            possible_postive.append(uncluster)
            posObjects=[] 
            posFeatures=[]
            y_pred=[]
            t=[]
            best_Mdule_features=[]
            POS_module=[]
            print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
       else:
           #print("No postive resion by this Module")
           POS_module=[]

       
#
#
clustered_data=intersectionMultiList (all_postive_Module)
#clustered_data = set().union(*all_postive_Module)
print("cluster Data  intersection of all postive moduls",clustered_data)
NEG_intersectionBND= intersectionMultiList (all_boundry_Module)
print("Negative region ", NEG_intersectionBND)
BND_regions= (set(listt) - set(clustered_data) - set(NEG_intersectionBND) )
print("BND_regions",BND_regions)

y_pred=[]
t=[]
average_accuracy=0.0
dc=0
for k in range (0, len(final_voting)):
    current_pred_modeul=final_voting[k]
    current_POS_indexes=all_postive_Module[k]
    for i in range(0, len(current_pred_modeul)):
          y_pred.append(current_pred_modeul[i])
          indexx=current_POS_indexes[i]
          t.append(true[indexx])
    ARI=adjusted_rand_score(t,y_pred)  
    founded_cluster= len (list(set(y_pred)))
    if founded_cluster > dc :
        dc= founded_cluster
    average_accuracy+= ARI
    y_pred=[]
    t=[]
#
#
print("average Accuracy=",average_accuracy / float(len(final_voting )))
print ("discoverd_clusters", dc)
if dc < number_cluster:
    print( "remining clusters " , number_cluster- dc)
matrix_adjecny, graphnetwork=adjacencyMatrix(allNodes)
linked=linkage( np.array(matrix_adjecny), 'complete')
postive_AHC_results=(cut_tree(linked, n_clusters = dc).T).tolist() 
print(postive_AHC_results) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>???
allNodes_onjects_list=[]
node_counter=0
for node in allNodes:
    #print (node_counter, ":", node.objects)
    allNodes_onjects_list.append(node.objects)
    node_counter +=1
"""
This step for using any classifeier for the data in the boundery 
in case we want to use KKN or SVM
But Note that KNN is a binar classifier where as SVM can be for multi classiferd
   
dfVPRST.drop(NEG_intersectionBND, inplace= True)
dfVPRST.drop(BND_regions, inplace= True)
data=pd.read_csv("soy_rock.csv",header=None,dtype='int')
data.drop(NEG_intersectionBND, inplace= True)
data.drop(BND_regions, inplace= True)
labels_pred= do_clustering(dfVPRST,dc)
data1=pd.read_csv("soy_rock.csv",header=None,dtype='int')
data1.drop(NEG_intersectionBND, inplace= True)
data1.drop(clustered_data, inplace= True)
findKKN(data,labels_pred,data1)
# for each cluster assign 
""" 
dfVPRST.drop(NEG_intersectionBND, inplace= True)
dfVPRST.drop(BND_regions, inplace= True)
labels_pred= do_clustering(dfVPRST,dc)
t=[]
#print("clustering pos label", labels_pred)
for i in range(0, size_data):
            if i in list(clustered_data):
               t.append(true[i])
ARI=adjusted_rand_score(t,labels_pred)
print("ARI Pos", ARI)   
"""
for each cluster find it's modes 
>>>>>>>>>>>>>>>>>>>>>>>>>>> step   assign negitive and boundery to the neareat cluster
"""
all_clusters=[]
dfVPRST['class'] =  labels_pred 
for clu in range(0, dc):
   df_temp = dfVPRST[dfVPRST['class'] == clu]
   objects_cluster=df_temp["id"].values.tolist()
   df_temp.pop("id")
   df_temp.pop("class")
   clus_mode= (df_temp.apply(most_frequent).tolist())
   cluster=clusters(clu,clus_mode,len(objects_cluster),objects_cluster,[])
   all_clusters.append(cluster)
similariy = -1
RSTData = pd.read_csv(data_set_name)

""" >>>>>2.1
Check if number of didcover cluster == number of cluster
if true:
    check if there is points in nedtive region 
    assign it to the neearest cluster
"""
if dc == number_cluster:
    for bnd_index in NEG_intersectionBND: 
        Obj=(RSTData.loc[ bnd_index , : ]).values.tolist()
        for clus in  all_clusters:
            mode=clus.mode
            sim=compute_nearest_cluster(set(mode), set(Obj))
            #print(sim)
            if sim > similariy:
                similariy= sim 
                candidat_cluster= clus.numcluster
        clus=all_clusters[candidat_cluster]
        clus.add_objects(bnd_index)
        similariy=-1

    for clus in  all_clusters:
          clus.update() 
          clus_number= clus.numcluster
          for item in clus.objects:
              RSTData.loc[item, 'class'] = clus_number
          df_temp = RSTData[RSTData['class'] == clus_number]
          clus_mode = (df_temp.apply(most_frequent).tolist())
          clus.update_mode(clus_mode)
    
""">>>>>>2.2
 assig boundery 
"""
similariy = -1
for bnd_index in BND_regions: 
    Obj=(RSTData.loc[ bnd_index , : ]).values.tolist()
    #print (Obj)
    #print(len(Obj))
    for clus in  all_clusters:
        mode=clus.mode
        #print(mode)
        sim=compute_nearest_cluster(set(mode), set(Obj))
        #print(sim)
        if sim > similariy:
            similariy= sim 
            candidat_cluster= clus.numcluster
    clus=all_clusters[candidat_cluster]
    clus.add_objects(bnd_index)
    similariy=-1

print("Done") 

for clus in  all_clusters:
    clus.update() 
    """ >>>>>>> we need also to update modes """
    clus_number= clus.numcluster
    for item in clus.objects:
       RSTData.loc[item, 'class'] = clus_number
    df_temp = RSTData[RSTData['class'] == clus_number]
    clus_mode= (df_temp.apply(most_frequent).tolist())
    clus.update_mode(clus_mode)
if dc != number_cluster:
    RSTData.drop(NEG_intersectionBND, inplace= True)    
#print(RSTData)   
trueVote= pd.read_csv(file_true_classes,header=None)
trueVote.drop(NEG_intersectionBND, inplace= True) 
true= sum (trueVote.values.tolist(),[]) 
pred= RSTData['class'].values.tolist() 
ARI_withBoundery=adjusted_rand_score(true,pred)
print("ARI after assining boundery to the nearest cluster",ARI_withBoundery )        
print (classification_report(true,pred) )


        
        

