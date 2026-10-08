import streamlit as st

def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Hide default Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Add some top padding since header is hidden */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Modern buttons */
        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transform: translateY(-1px);
        }
        
        /* Metric cards styling */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            border: 1px solid #E2E8F0;
        }
        
        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
