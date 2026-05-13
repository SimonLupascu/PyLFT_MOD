import streamlit as st

st.title("MO Diagram Finder for Coordination Complexes")
st.write("Package to find the MO diagram of a given complex coordinated by a ligand field.")

with st.sidebar: 
    st.subheader("Input")
    metal = st.selectbox("Select a metal", ["Fe", "Co", "Ni"])
    ligand1 = st.selectbox("Select first ligand", ["H2O", "NH3", "Cl-"])
    ligand2 = st.selectbox("Select second ligand", ["H2O", "NH3", "Cl-"])
    ligand3 = st.selectbox("Select third ligand", ["H2O", "NH3", "Cl-"])
    ligand4 = st.selectbox("Select fourth ligand", ["H2O", "NH3", "Cl-"])
    ligand5 = st.selectbox("Select fifth ligand", ["H2O", "NH3", "Cl-"])
    ligand6 = st.selectbox("Select sixth ligand", ["H2O", "NH3", "Cl-"])
    ligand = f"{ligand1}, {ligand2}, {ligand3}, {ligand4}, {ligand5}, {ligand6}"
    complex = f"{metal}({ligand})"
    st.write(f"You selected: {complex}")

tab1, tab2 = st.tabs(["Complex Visualiser","MO Diagram"])
with tab1:
    st.subheader("Complex Visualiser")
    st.write("Here you would visualise the complex you have selected.")
with tab2:
    st.subheader("MO Diagram")
    st.write("Here you would see the MO diagram for the selected complex.")

