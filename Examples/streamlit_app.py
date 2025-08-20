import streamlit as st

# title
st.title("Streamlit Spike")

# Text input 
name = st.text_input("Enter your name")

# Selectbox
color = st.selectbox("Pick your favorite color:", ["Red","Green","Blue"])

# Slider 
age = st.slider("How old are you?", 0 , 100, 25)

# checkbox
show_message = st.checkbox("Show greeting Message")

# Button to trigger action
if st.button("Submit"):
    st.success(f"Hello {name}! You are {age} years old and like {color}.")

# Dynamic Live Display
st.write("**Live Preview**")
st.write(f"Name: {name}")
st.write(f"Color: {color}")
st.write(f"Age: {age}")
st.write(f"Greeting Visible: {show_message}")