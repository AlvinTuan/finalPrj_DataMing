import streamlit as st
import numpy as np
import pandas as pd
import pickle
from mlxtend.frequent_patterns import association_rules, apriori, fpgrowth
from streamlit_modal import Modal
import time

def load_data():
  with open("./data.pkl", "rb") as file:
    df = pickle.load(file)
  return df

data = load_data()
items = ['Adjustment', 'Afternoon with the baker', 'Alfajores','Argentina Night', 'Art Tray', 'Bacon', 'Baguette', 'Bakewell','Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread', 'Bread Pudding','Brioche and salami', 'Brownie', 'Cake', 'Caramel bites','Cherry me Dried fruit', 'Chicken Stew', 'Chicken sand','Chimichurri Oil', 'Chocolates', 'Christmas common', 'Coffee','Coffee granules ', 'Coke', 'Cookies', 'Crepes', 'Crisps','Drinking chocolate spoons ', 'Duck egg', 'Dulce de Leche', 'Eggs',"Ella's Kitchen Pouches", 'Empanadas', 'Extra Salami or Feta','Fairy Doors', 'Farm House', 'Focaccia', 'Frittata', 'Fudge','Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack','Half slice Monster ', 'Hearty & Seasonal', 'Honey', 'Hot chocolate','Jam', 'Jammie Dodgers', 'Juice', 'Keeping It Local', 'Kids biscuit','Lemon and coconut', 'Medialuna', 'Mighty Protein', 'Mineral water','Mortimer', 'Muesli', 'Muffin', 'My-5 Fruit Shoot', 'Nomad bag','Olum & polenta', 'Panatone', 'Pastry', 'Pick and Mix Bowls', 'Pintxos','Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars','Salad', 'Sandwich', 'Scandinavian', 'Scone', 'Siblings', 'Smoothies','Soup', 'Spanish Brunch', 'Spread', 'Tacos/Fajita', 'Tartine', 'Tea','The BART', 'The Nomad', 'Tiffin', 'Toast', 'Truffles', 'Tshirt',"Valentine's card", 'Vegan Feast', 'Vegan mincepie','Victorian Sponge']

def remove_frozenset(item):
    # Chuyển đổi frozenset thành chuỗi thường
    if isinstance(item, frozenset):
        return ', '.join(item)
    return item
    
def handleFrequentItemAP():
  result_ap = apriori(data, min_support=0.01, use_colnames=True)
  # result_ap.head(number_get).sort_values('support', ascending=False)
  result_ap = result_ap.sort_values('support', ascending=False)
  return result_ap

#so sánh thời gian chạy 2 thuật toán
def perform_rule_calculation(transact_items_matrix, rule_type="fpgrowth", min_support=0.001):
    start_time = 0
    total_execution = 0
    
    
    if(not rule_type=="fpgrowth"):
        start_time = time.time()
        rule_items = apriori(transact_items_matrix, min_support=min_support, use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Apriori!")
        
    else:
        start_time = time.time()
        rule_items = fpgrowth(transact_items_matrix, min_support=min_support, use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Fp Growth!")
    
    rule_items['number_of_items'] = rule_items['itemsets'].apply(lambda x: len(x))
    
    return rule_items, total_execution

def get_rules_ap():
  result_ap = handleFrequentItemAP()
  rules_ap = association_rules(result_ap, metric='lift', min_threshold=1.0)
  return rules_ap

def get_itemset(frequent_itemsets, min_support=0.01, min_confidence=0.1, min_lift=1, min_leverage=0.001, min_zhangs=0.02):
    rules = association_rules(frequent_itemsets, metric='lift')
    result = rules[(rules['support'] >= min_support) & (rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift) & (rules['leverage'] >= min_leverage) & (rules['zhangs_metric'] >= min_zhangs)]
    return result
  
def recommend(freq_itemset, ante, min_support=0.01, min_confidence=0.1, min_lift=1, min_leverage=0.001, min_zhangs=0.02):
    itemset = get_itemset(freq_itemset, min_support, min_confidence, min_lift, min_leverage, min_zhangs)
    for row in range(itemset.shape[0]):
        if ante in list(itemset.iloc[row, 0]):
            st.write("If you buy {0}, you should buy {1}".format(list(itemset.iloc[row, 0]), list(itemset.iloc[row, 1])))
            



def show_apriori():
  st.title("Custom Behaviour Detection")
  
  st.header("Frequent itemsets")
  
  min_support = st.number_input("Min support",min_value=0.00, value=0.01, step=0.01)
  number_get = st.number_input("Quantity want to get", min_value=0, value=5, format="%d")
  if (st.button("Show")):
    frequent_itemsets = handleFrequentItemAP()
    if (min_support):
      frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= min_support]
    st.dataframe(frequent_itemsets.head(number_get))
  
  
  """
  Association rule
  """
  st.header("Association rule")
  #phần thông số
  min_support = st.number_input("Min support", min_value=0.00, value=0.01, step=0.01, key="suppRule")
  min_confidence = st.number_input("Min confidence",min_value=0.0, value=0.1, step=0.1)
  min_lift = st.number_input("Min lift",min_value=0, value=1, step=1)
  min_leverage = st.number_input("Min leverage",min_value=0.0000, value=0.005, step=0.001)
  
  if (st.button("Find")):
    frequent_itemsets = handleFrequentItemAP()
    apriori_matrix, apriori_exec_time = perform_rule_calculation(data, rule_type="apriori")
    st.write("Apriori Execution took: {} seconds".format(apriori_exec_time))
    # rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    rules_new = get_itemset(frequent_itemsets, min_support, min_confidence, min_lift, min_leverage)
    st.dataframe(rules_new)
    
  st.header("Recommendation")
  item = st.selectbox("Choose items", items, index=15)
  if (st.button("Recommend")):
    result_ap = handleFrequentItemAP()
    recommend(freq_itemset=result_ap, ante=item)
  