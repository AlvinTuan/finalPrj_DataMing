from show_apriori import handleFrequentItemAP, get_rules_ap
from show_fp_growth import handleFrequentItemFP, get_rules_fp
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from mlxtend.frequent_patterns import association_rules, fpgrowth, apriori
import pickle

def load_data():
  with open("./data.pkl", "rb") as file:
    df = pickle.load(file)
  return df

data = load_data()

def get2item_compareDSA():
  st.header("Get itemsets with 2 or more products and compare the results of the 2 algorithms")
  result_ap = handleFrequentItemAP()
  result_fp = handleFrequentItemFP()
  frequent_itemsets_fp = result_fp[result_fp['itemsets'].apply(lambda x: len(x) >= 2)].sort_values(by='support', ascending=False)
  frequent_itemsets_fp = frequent_itemsets_fp.rename(columns={'support': 'FP support', 'itemsets': 'FP itemset'}).reset_index(drop=True)

  frequent_itemsets_ap = result_ap[result_ap['itemsets'].apply(lambda x: len(x) >= 2)].sort_values(by='support', ascending=False)
  frequent_itemsets_ap = frequent_itemsets_ap.rename(columns={'support': 'AP support', 'itemsets': 'AP itemset'}).reset_index(drop=True)

  compare = pd.concat([frequent_itemsets_fp, frequent_itemsets_ap], axis=1)
  st.write(compare)
  
#so sánh thời gian chạy 2 thuật toán
def perform_rule_calculation(transact_items_matrix, rule_type="fpgrowth", min_support=0.001):
    start_time = 0
    total_execution = 0
    
    
    if(not rule_type=="fpgrowth"):
        start_time = time.time()
        rule_items = apriori(transact_items_matrix, min_support=min_support, use_colnames=True)
        total_execution = time.time() - start_time
        # print("Computed Apriori!")
        
    else:
        start_time = time.time()
        rule_items = fpgrowth(transact_items_matrix, min_support=min_support, use_colnames=True)
        total_execution = time.time() - start_time
        # print("Computed Fp Growth!")
    
    rule_items['number_of_items'] = rule_items['itemsets'].apply(lambda x: len(x))
    
    return rule_items, total_execution
  
#biểu đồ giữa lift và confidence
def plot_metrics_relationship(rule_matrix, col1, col2):
    fit = np.polyfit(rule_matrix[col1], rule_matrix[col2], 1)
    fit_funt = np.poly1d(fit)
    plt.plot(rule_matrix[col1], rule_matrix[col2], 'yo', rule_matrix[col1], 
    fit_funt(rule_matrix[col1]))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} vs {}'.format(col1, col2))
    st.pyplot(plt)
    
#biểu đồ so sánh thời gian chạy
def compare_time_exec(algo1=list, algo2=list):
    execution_times = [algo1[1], algo2[1]]
    algo_names = (algo1[0], algo2[0])
    y=np.arange(len(algo_names))
    
    plt.bar(y,execution_times,color=['orange', 'blue'])
    plt.xticks(y,algo_names)
    plt.xlabel('Algorithms')
    plt.ylabel('Time')
    plt.title("Execution Time (seconds) Comparison")
    st.pyplot(plt)
    
def plt_relationship():
  
  
  col1, col2 = st.columns(2)
  with col1:
    st.write("Apriori")
    rules_ap = get_rules_ap()
    plot_metrics_relationship(rules_ap, col1='lift', col2='confidence')
  with col2:
    st.write("FP-Growth")
    rules_fp = get_rules_fp()
    plot_metrics_relationship(rules_fp, col1='lift', col2='confidence')

def plt_compasion_time():
  apriori_matrix, apriori_exec_time = perform_rule_calculation(data, rule_type="apriori")
  fpgrowth_matrix, fp_growth_exec_time = perform_rule_calculation(data) # Run the algorithm
  algo1 = ['Fp Growth', fp_growth_exec_time]
  algo2 = ['Apriori', apriori_exec_time]
  st.header("Execution Time (seconds) Comparison")
  compare_time_exec(algo1, algo2)
  
def main_evaluate():
  options = ["Execution Time (seconds) Comparison", "Relationship attribute", "Get itemsets with 2 or more products and compare the results of the 2 algorithms"]
  choose = st.selectbox("Items", options)
  if ("Comparison" in choose):
    plt_compasion_time()
  elif ("algorithms" in choose):
    get2item_compareDSA()
  else:
    plt_relationship()
  
  