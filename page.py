import streamlit as st

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Function to navigate to the next page
def next_page():
    st.session_state.current_page = 2

# Function to navigate to the previous page
def previous_page():
    st.session_state.current_page = 1

# Page 1 content
if st.session_state.current_page == 1:
    st.title("Page 1")
    st.write("Welcome to the first page!")
    if st.button("Next"):
        next_page()

# Page 2 content
elif st.session_state.current_page == 2:
    st.title("Page 2")
    st.write("This is the second page!")
    if st.button("Previous"):
        previous_page()


