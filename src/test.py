import streamlit as st
from show_apriori import show_apriori
from explore_page import show_explore_page
from show_fp_growth import show_fp_growth
from show_evaluate import main_evaluate

page = st.sidebar.selectbox("Menu", ("Apriori", "FP-Growth", "Visualization", "Evaluate"))

if (page == "Apriori"):
  show_apriori()
elif (page == "FP-Growth"):
  show_fp_growth()
elif (page == "Evaluate"):
  main_evaluate()
else:
  show_explore_page()