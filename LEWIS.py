# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:17:39 2022

@author: daisu
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

import dowhy
from dowhy import CausalModel
from dowhy.utils import graph_operations
import dowhy.datasets

#バックドア基準を満たす変数を取得
def get_backdoor(data, x, y, graph=None,
                 adjacency_matrix=None, nodes=None,dowhy_data=False):
  #create_graph
  if dowhy_data:
    model=CausalModel(
        data = data,
        treatment=x,
        outcome=y,
        graph=graph
        )      
  else:
    graph_dot = graph_operations.adjacency_matrix_to_graph(adjacency_matrix, nodes)
    graph_dot = graph_operations.str_to_dot(graph_dot.source)
    model = CausalModel(data=data,treatment=x,outcome=y,graph=graph_dot) 
  #Identify causal effect using the optimized backdoor implementation
  identified_estimand = model.identify_effect(optimize_backdoor=True)
  
  backdoor=identified_estimand.get_backdoor_variables()
  return backdoor,model


def X_CF(x, x_base):
    x_CF=[i for i in x.unique() if i != x_base]
    return x_CF  

def X_CF_order(x, x_base):
    x_CF=[i for i in x.unique() if i > x_base]
    return x_CF  

#local levelのsuffecient_score
def Suff_Individual(data, x, k,x_base, x_base_name,backdoor):
    x_cf=X_CF(x, x_base)
    n_x_cf=len(x_cf) #同じ
    suf_sum=0
    pos_con_suf=0
    neg_con_suf=0
    
    for  i , n in zip(x_cf, range(n_x_cf)):
        #sufficiency_score
        #print(i)
        bdp_suf_sum=0
        bd_n=len(backdoor)
            

        if backdoor==[]:
            suf_var=data.loc[(data[x_base_name]==i)]
            suf_var_num=suf_var.loc[suf_var["Model_pred"]==1]
            suf_num, suf_den=len(suf_var_num), len(suf_var)#P(o,c,x,k), P(c,x,k)
            bdp_suf=(suf_num/suf_den)
        else:  
            for j in backdoor:#Σ内の計算
                c_unique=data[j].unique()
                sum=0
                for k in c_unique:
                    #print(k)

                    A=len(data[(data[j]==k)&(data[x_base_name]==x_base)])
                    B=len(data.loc[(data[x_base_name]==x_base)])
                    pr_c_x1_k=A/B
                    suf_var=data.loc[(data[x_base_name]==i)&(data[j]==k)]
                    suf_var_num=suf_var.loc[suf_var["Model_pred"]==1]
                    suf_num, suf_den=len(suf_var_num), len(suf_var)#P(o,c,x,k), P(c,x,k)
                    bdp_suf=(suf_num/suf_den)*pr_c_x1_k
                    #print(A,B, pr_c_x1_k, suf_num, suf_den,bdp_suf)
                    sum=sum+bdp_suf
                bdp_suf_sum+=sum
        
            bdp_suf=bdp_suf_sum/bd_n    
        #print("a2:",a2_p)    
        suf_var2=data.loc[(data[x_base_name]==x_base)]
        suf_var_num2=suf_var2.loc[suf_var2["Model_pred"]==1]
        suf_p2=len(suf_var_num2)/len(suf_var2)
        #print("b2:",len(b2_var_num),len(b2_var), b2_p)    

        suf_var_num3=suf_var2.loc[suf_var2["Model_pred"]==0]
        suf_p3=len(suf_var_num3)/len(suf_var2)
        #print("c2:",b2_p) 


        score=(bdp_suf-suf_p2)/suf_p3
        if score > 1:
            score=1
        if score < 0:
            score=0   
        print(i, "suf_score:",score)
        if i > x_base:
            #print(i)
            if neg_con_suf < score:
                neg_con_suf =score
        else:
            #print(i)
            if pos_con_suf < score:
                pos_con_suf  = score    
        print("-------------------------------")

    return pos_con_suf, neg_con_suf

#local levelのneceesary_score
def Nec_Individual(data, x, k,x_base, x_base_name,backdoor):
    x_cf=X_CF(x, x_base)
    n_x_cf=len(x_cf) #同じ

    nec_sum=0
    pos_con_nec=0
    neg_con_nec=0


    for  i , n in zip(x_cf, range(n_x_cf)):
        #neceesary_score
        #print(i)
        bdp_nec_sum=0
        bd_n=len(backdoor)
        if backdoor==[]:
            nec_var=data.loc[(data[x_base_name]==x_base)]
            nec_var_num=nec_var.loc[nec_var["Model_pred"]==0]
            nec_num, nec_den=len(nec_var_num), len(nec_var)
            bdp_nec=(nec_num/nec_den)
        else:
            for j in backdoor:#Σ内の計算
                c_unique=data[j].unique()
                sum=0
                for k in c_unique:
                    A=len(data[(data[j]==k)&(data[x_base_name]==i)])
                    B=len(data.loc[(data[x_base_name]==i)])
                    #print(A,B)
                    pr_c_x_k=A/B
                    nec_var=data.loc[(data[x_base_name]==x_base)&(data[j]==k)]
                    nec_var_num=nec_var.loc[nec_var["Model_pred"]==0]
                    nec_num, nec_den=len(nec_var_num), len(nec_var)#P(o,c,x,k), P(c,x,k)
                    bdp_nec=(nec_num/nec_den)*pr_c_x_k
                    bdp_nec_sum+=bdp_nec
            bdp_nec=bdp_nec_sum
        #print("a2:",a2_p)    
    
        nec_var2=data.loc[(data[x_base_name]==i)]
        nec_var_num2=nec_var2.loc[nec_var2["Model_pred"]==0]
        nec_p2=len(nec_var_num2)/len(nec_var2)
        #print("b2:",len(b2_var_num),len(b2_var), b2_p)    

        nec_var_num3=nec_var2.loc[nec_var2["Model_pred"]==1]
        nec_p3=len(nec_var_num3)/len(nec_var2)
        #print("c2:",b2_p) 
        score=(bdp_nec-nec_p2)/nec_p3
        if score > 1:
            score=1
        if score < 0:
            score=0   
        #print(i, "nec_score:",score)
        if i > x_base:
            if neg_con_nec < score:
                neg_con_nec =score
        else:
            if pos_con_nec < score:
                pos_con_nec  = score    
        #print("-------------------------------")
    return pos_con_nec, neg_con_nec    


def Suff_Grobal(data, x, x_base_name,backdoor,order=True):    
    suf_sum=0
    x_df=x.unique()
    suf_score_sum=0
    count=0
    for m  in x_df:
        if order:
            x_cf=[j for j in x_df if j > m]
        else:
            x_cf=[j for j in x_df if j != x_base]
        for  i in x_cf:
        #sufficiency_score
            bdp_suf_sum=0
            bdp_nec_sum=0
            bd_n=len(backdoor)

            if backdoor==[]:
                suf_var=data.loc[(data[x_base_name]==i)]
                suf_var_num=suf_var.loc[suf_var["Model_pred"]==1]
                suf_num, suf_den=len(suf_var_num), len(suf_var)#P(o,c,x,k), P(c,x,k)
                bdp_suf=(suf_num/suf_den)

            else:    
                for j in backdoor:#Σ内の計算
                    c_unique=data[j].unique()
                    sum=0
                    for k in c_unique:
                        #print(k)

                        A=len(data[(data[j]==k)&(data[x_base_name]==m)])
                        B=len(data.loc[(data[x_base_name]==m)])
                        pr_c_x1_k=A/B
                        suf_var=data.loc[(data[x_base_name]==i)&(data[j]==k)]
                        suf_var_num=suf_var.loc[suf_var["Model_pred"]==1]
                        suf_num, suf_den=len(suf_var_num), len(suf_var)#P(o,c,x,k), P(c,x,k)
                        if suf_den==0:
                            bdp_suf=0
                        else:
                            bdp_suf=(suf_num/suf_den)*pr_c_x1_k
                        #print(A,B, pr_c_x1_k, suf_num, suf_den,bdp_suf)
                        sum=sum+bdp_suf
                bdp_suf_sum+=sum

                bdp_suf=bdp_suf_sum  
            #print("a2:",bdp_suf)    
            suf_var2=data.loc[(data[x_base_name]==m)]
            suf_var_num2=suf_var2.loc[suf_var2["Model_pred"]==1]
            suf_p2=len(suf_var_num2)/len(suf_var2)
            #print("b2:", suf_p2)    

            suf_var_num3=suf_var2.loc[suf_var2["Model_pred"]==0]
            suf_p3=len(suf_var_num3)/len(suf_var2)
            #print(suf_var2)
            #print("c2:",suf_p3) 


            if suf_p3==0: score==0
            else: score=(bdp_suf-suf_p2)/suf_p3

            #print(m, i, "suf_score:",score)
            if score > 1:
                score=1
            if score < 0:
                score=0
            suf_score_sum+=score
            count+=1
            
            #print(suf_score_sum)
            #print("-------------------------------")
        #print("ave",ave_score)

    ave_score=suf_score_sum/count
    return ave_score

def Nec_Grobal(data, x, x_base_name,backdoor,order=True):
    nec_sum=0
    x_df=x.unique()
    nec_score_sum=0
    count=0
    for m  in x_df:
        suf_score_sum2=0
        if order:
            x_cf=[j for j in x_df if j > m]
        else:
            x_cf=[j for j in x_df if j != m]
        for  i in x_cf:
        #sufficiency_score
        #print(i)
            bd_n=len(backdoor)

            if backdoor==[]:
                nec_var=data.loc[(data[x_base_name]==m)]
                nec_var_num=nec_var.loc[nec_var["Model_pred"]==0]
                nec_num, nec_den=len(nec_var_num), len(nec_var)
                bdp_nec=(nec_num/nec_den)
            else:
                for j in backdoor:#Σ内の計算
                    c_unique=data[j].unique()
                    bdp_nec_sum=0
                    for k in c_unique:
                        A=len(data[(data[j]==k)&(data[x_base_name]==i)])
                        B=len(data.loc[(data[x_base_name]==i)])
                        #print(A,B)
                        pr_c_x_k=A/B
                        nec_var=data.loc[(data[x_base_name]==m)&(data[j]==k)]
                        nec_var_num=nec_var.loc[nec_var["Model_pred"]==0]
                        nec_num, nec_den=len(nec_var_num), len(nec_var)#P(o,c,x,k), P(c,x,k)
                        if nec_den==0:
                            bdp_nec=0
                        else:
                            bdp_nec=(nec_num/nec_den)*pr_c_x_k
                        bdp_nec_sum+=bdp_nec
                bdp_nec=bdp_nec_sum
            #print("a2:",bdp_suf)    
            nec_var2=data.loc[(data[x_base_name]==i)]
            nec_var_num2=nec_var2.loc[nec_var2["Model_pred"]==0]
            nec_p2=len(nec_var_num2)/len(nec_var2)
            #print("b2:", suf_p2)    

            nec_var_num3=nec_var2.loc[nec_var2["Model_pred"]==1]
            nec_p3=len(nec_var_num3)/len(nec_var2)
            #print("c2:",suf_p3) 

            if nec_p3==0: score==0
            else: score=(bdp_nec-nec_p2)/nec_p3
            if score > 1:
                score=1
            if score < 0:
                score=0
            nec_score_sum+=score 
            count+=1
            #print(m, i, "nec_score:",score)
            #print("-------------------------------")

    ave_score=nec_score_sum/count
    return ave_score    


def NeSuf_Grobal(data, x, x_base_name,backdoor, order=True):
    x_df=x.unique()
    count=0
    necsuf_score_sum=0
    for m  in x_df:
        suf_score_sum2=0
        if order:
            x_cf=[j for j in x_df if j > m]
        else:
            x_cf=[j for j in x_df if j != m]
        for  i in x_cf:
            
            bd_n=len(backdoor)

            if backdoor==[]:
                df_x=data.loc[(data[x_base_name]==m)]
                df_o_x=df_x.loc[df_x["Model_pred"]==0]
                pr_o_x=(len(df_o_x))/(len(df_x))
                
                df_xcf=data.loc[(data[x_base_name]==i)]
                df_o_xcf=df_xcf.loc[df_xcf["Model_pred"]==0]
                pr_o_xcf=(len(df_o_xcf))/(len(df_xcf))
                
                pr_necsuf=pr_o_x - pr_o_xcf
            else:
                for j in backdoor:#Σ内の計算
                    c_unique=data[j].unique()
                    c_value=list(data[j].value_counts(dropna=False, normalize=True,sort=False))
                    bdp_necsuf_sum=0
                    sum=0
                    for k , k_value in zip(c_unique, c_value):
        
                        pr_c=k_value
                        df_x_c=data.loc[(data[x_base_name]==m)&(data[j]==k)]
                        df_o_x_c=df_x_c.loc[df_x_c["Model_pred"]==0]
                        if len(df_x_c)==0:
                            pr_o_x_c=0
                        else:
                            pr_o_x_c=(len(df_o_x_c))/(len(df_x_c))
                        
                        df_xcf_c=data.loc[(data[x_base_name]==i)&(data[j]==k)]
                        df_o_xcf_c=df_xcf_c.loc[df_xcf_c["Model_pred"]==0]
                        if len(df_xcf_c)==0:
                            pr_o_xcf_c=0
                        else:
                            pr_o_xcf_c=(len(df_o_xcf_c))/(len(df_xcf_c))
                        
                        pr_necsuf=(pr_o_x_c - pr_o_xcf_c)*pr_c
                        bdp_necsuf_sum+=pr_necsuf
                pr_necsuf=bdp_necsuf_sum      
            
            #print("c2:",suf_p3) 

            
            if pr_necsuf > 1:
                pr_necsuf=1
            if pr_necsuf< 0:
                pr_necsuf=0
            necsuf_score_sum+=pr_necsuf 
            count+=1
            #print(m, i, "nec_score:",score)
            #print("-------------------------------")

    ave_score=necsuf_score_sum/count
    return ave_score



    #bounded
def Nec_Bounds(data, x, x_base_name):
    max_b=[]
    min_b=[]
    x_base=np.sort(x.unique())#ベースの値
    n_x_base=len(x_base)
    df=pd.crosstab(data[x_base_name], data['Model_pred'], normalize=True)#同時確率を求めるためのクロス表
    for i, n in zip(x_base, range(n_x_base)):
        x_cf=np.sort(X_CF(x, i))#ベースに対する反事実値
        n_x_cf=len(x_cf)
        df2=df.T[x_cf]
        pr_o_x=df.iloc[n,1]#pr(o, x)
        for j,m in zip(x_cf,range(n_x_cf)): 
            #上限
            pr_o2_do_x2=len(data.loc[(data[x_base_name]==i)&(data["Model_pred"]==0)])/len(data.loc[(data[x_base_name]==i)])
            pr_o2_x2=df2.T.iloc[m,0]
            max_bound=(pr_o2_do_x2-pr_o2_x2)/pr_o_x
            max_bound=min(max_bound, 1)
            max_b.append(max_bound)

            #下限
            pr_o_do_x2=len(data.loc[(data[x_base_name]==i)&(data["Model_pred"]==1)])/len(data.loc[(data[x_base_name]==i)])
            pr_o_x2=df2.T.iloc[m,1]
            min_bound=(pr_o_x+pr_o_x2-pr_o_do_x2)/pr_o_x
            min_bound=max(min_bound, 0)
            min_b.append(min_bound)

    return np.mean(min_b), np.mean(max_b)


def Suf_Bounds(data, x, x_base_name):
    max_b=[]
    min_b=[]
    x_base=np.sort(x.unique())#ベースの値
    n_x_base=len(x_base)
    df=pd.crosstab(data[x_base_name], data['Model_pred'], normalize=True)#同時確率を求めるためのクロス表
    for i, n in zip(x_base, range(n_x_base)):
        x_cf=np.sort(X_CF(x, i))#ベースに対する反事実値
        n_x_cf=len(x_cf)
        df2=df.T[x_cf]
        pr_o_x=df.iloc[n,1]#pr(o, x)
        pr_o2_x=df.iloc[n,0]#pr(o', x)
        for j,m in zip(x_cf,range(n_x_cf)): 
            #上限
            pr_o_do_x=len(data.loc[(data[x_base_name]==j)&(data["Model_pred"]==1)])/len(data.loc[(data[x_base_name]==j)])
            pr_o2_x2=df2.T.iloc[m,0]
            max_bound=(pr_o_do_x-pr_o_x)/pr_o2_x2
            max_bound=min(max_bound, 1)
            max_b.append(max_bound)

            #下限
            pr_o2_do_x=len(data.loc[(data[x_base_name]==j)&(data["Model_pred"]==0)])/len(data.loc[(data[x_base_name]==j)])
            
            min_bound=(pr_o2_x+pr_o2_x2-pr_o2_do_x)/pr_o2_x2
            min_bound=max(min_bound, 0)
            min_b.append(min_bound)

    return np.mean(min_b), np.mean(max_b)    