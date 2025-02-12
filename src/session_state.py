import streamlit as st

def initialize_session_state():
    """Ensure session state variables are available across all pages"""
    
    if 'output' not in st.session_state:
        st.session_state.output = None
        st.session_state.output_df = None

    if "blocking_metrics" not in st.session_state:
        st.session_state.blocking_metrics = {
            "reduction_ratio": None,
            "recall": None,
            "f1": None
        }
        
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
        st.session_state.cols_a = None
        st.session_state.cols_b = None