
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("star_classifier.pkl")

# Title
st.title("🌌 Star Evolution Analyzer")

st.write("Predict star types using stellar properties.")

# Inputs
temp = st.number_input("Temperature (K)", min_value=1000)

luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0)

radius = st.number_input("Radius (R/Ro)", min_value=0.0)

magnitude = st.number_input("Absolute Magnitude")

star_color = st.number_input("Star Color Encoded")

spectral_class = st.number_input("Spectral Class Encoded")

# Predict button
if st.button("Predict Star Type"):

    features = np.array([[
        temp,
        luminosity,
        radius,
        magnitude,
        star_color,
        spectral_class
    ]])

    prediction = model.predict(features)

    star_types = {
        0: "Brown Dwarf",
        1: "Red Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Supergiant",
        5: "Hypergiant"
    }

    result = star_types[prediction[0]]

    st.success(f"Predicted Star Type: {result}")

    # HR Diagram Visualization
    fig, ax = plt.subplots()

    ax.scatter(temp, luminosity)

    ax.set_xscale('linear')
    ax.set_yscale('log')

    ax.invert_xaxis()

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Luminosity")

    ax.set_title("Hertzsprung-Russell Diagram")

    st.pyplot(fig)
