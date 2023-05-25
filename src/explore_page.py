import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def load_data():
  with open("./data_visualization.pkl", "rb") as file:
    df = pickle.load(file)
  return df

df = load_data()

def show_items_purchased(top_item):
  plt.figure(figsize=(15,5))
  sns.barplot(x = df.Item.value_counts().head(top_item).index, y = df.Item.value_counts().head(top_item).values, palette = 'gnuplot')
  plt.xlabel('Items', size = 15)
  plt.xticks(rotation=45)
  plt.ylabel('Count of Items', size = 15)
  plt.title(f'Top {top_item} Items purchased by customers', color = 'green', size = 20)
  st.pyplot(plt)
  
def show_orders_month(number_month=12):
  monthTran = df.groupby('month')['Transaction'].count().reset_index()
  monthTran.loc[:,"monthorder"] = [4,8,12,2,1,7,6,3,5,11,10,9]
  monthTran.sort_values("monthorder",inplace=True)

  plt.figure(figsize=(12,5))
  sns.barplot(data = monthTran[:number_month], x = "month", y = "Transaction")
  plt.xlabel('Months', size = 15)
  plt.ylabel('Orders per month', size = 15)
  plt.title('Number of orders received each month', color = 'green', size = 20)
  st.pyplot(plt)
  
def show_orders_day():
  weekTran = df.groupby('weekday')['Transaction'].count().reset_index()
  weekTran.loc[:,"weekorder"] = [4,0,5,6,3,1,2]
  weekTran.sort_values("weekorder",inplace=True)

  plt.figure(figsize=(12,5))
  sns.barplot(data = weekTran, x = "weekday", y = "Transaction")
  plt.xlabel('Week Day', size = 15)
  plt.ylabel('Orders per day', size = 15)
  plt.title('Number of orders received each day', color = 'green', size = 20)
  st.pyplot(plt)
  
def show_orders_hour():
  hourTran = df.groupby('hour')['Transaction'].count().reset_index()
  hourTran.loc[:,"hourorder"] = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,7,8,9]
  hourTran.sort_values("hourorder",inplace=True)

  plt.figure(figsize=(12,5))
  sns.barplot(data = hourTran, x = "Transaction", y = "hour")
  plt.ylabel('Hours', size = 15)
  plt.xlabel('Orders each hour', size = 15)
  plt.title('Count of orders received each hour', color = 'green', size = 20)
  st.pyplot(plt)
  
def show_order_part():
  dayTran = df.groupby('period_day')['Transaction'].count().reset_index()
  # dayTran.loc[:,"hourorder"] = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,7,8,9]
  # dayTran.sort_values("hourorder",inplace=True)

  plt.figure(figsize=(12,5))
  sns.barplot(data = dayTran, x = "Transaction", y = "period_day")
  plt.ylabel('Period', size = 15)
  plt.xlabel('Orders each period of a day', size = 15)
  plt.title('Count of orders received each period of a day', color = 'green', size = 20)
  st.pyplot(plt)
  
def show_item_people_like(day):
  data = df.groupby(['period_day','Item'])['Transaction'].count().reset_index().sort_values(['period_day','Transaction'],ascending=False)
  # day = ['morning','afternoon','evening','night']

  plt.figure(figsize=(15,8))
  for i,j in enumerate(day):
      plt.subplot(2,2,i+1)
      df1 = data[data.period_day==j].head(10)
      sns.barplot(data=df1, y=df1.Item, x=df1.Transaction, color='pink')
      plt.xlabel('')
      plt.ylabel('')
      plt.title('Top 10 items people like to order in "{}"'.format(j), size=13)
  st.pyplot(plt)

def show_explore_page():
  options = ["Top items purchased by customers", "Number of orders received each month", "Number of orders received each day", "Count of orders received each hour", "Top items people like to order"]
  choose = st.selectbox("Items", options)
  
  if ("purchased" in choose):
    top_item = st.number_input("Input number",value=20, format="%d")
    
  # if ("month" in choose):
  #   number_month = st.number_input("Input month",min_value=1, max_value=12 ,value=12, format="%d")
  
  if ("people" in choose):
    day = st.multiselect("Days", ['morning','afternoon','evening','night'])
    
  if (st.button("Show")):
    if ("purchased" in choose):
      show_items_purchased(top_item) if top_item <= len(df.Item.value_counts()) else st.warning('Error input number', icon="⚠️")
    elif ("month" in choose):
      show_orders_month()
    elif ("day" in choose):
      show_orders_day()
    elif ("hour" in choose):
      show_orders_hour()
    # elif ("part" in choose):
    #   show_order_part()
    if ("people" in choose):
      show_item_people_like(day)