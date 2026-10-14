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

        /* ==================================================== */
        /* SIDEBAR ENHANCEMENTS & TYPOGRAPHY                    */
        /* ==================================================== */
        /* Sidebar container spacing & subtle border */
        [data-testid="stSidebar"] {
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Brand title & caption in sidebar */
        [data-testid="stSidebar"] h1 {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #1E3A8A !important;
            margin-bottom: 0.15rem !important;
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] .stCaption {
            font-size: 0.8rem !important;
            color: #64748B !important;
            margin-bottom: 1.5rem !important;
        }

        /* Section headers in navigation (e.g., Quản lý, Nhận diện AI) */
        [data-testid="stSidebarNav"] span[data-testid="stSidebarNavSectionHeader"] {
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #64748B !important;
            margin-top: 1rem !important;
            margin-bottom: 0.35rem !important;
            padding-left: 0.5rem !important;
        }

        /* Sidebar navigation links */
        [data-testid="stSidebarNav"] ul {
            padding: 0 !important;
        }

        [data-testid="stSidebarNav"] li {
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stSidebarNav"] a {
            padding: 0.55rem 0.85rem !important;
            border-radius: 8px !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            color: #334155 !important;
            transition: all 0.15s ease-in-out !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
        }

        /* Hover effect on menu item */
        [data-testid="stSidebarNav"] a:hover {
            background-color: #F1F5F9 !important;
            color: #1E3A8A !important;
            transform: translateX(2px);
        }

        /* Active / Selected menu item */
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background-color: #EEF2FF !important;
            color: #1E3A8A !important;
            font-weight: 600 !important;
            border-left: 3.5px solid #1E3A8A !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
