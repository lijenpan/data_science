import streamlit as st


def display_header(header):
    """
    function to display minor headers at user interface main panel

    Parameters
    ----------
    header: str -> the major text to be displayed
    """

    # view clean data
    html_temp = f"""
    <div style="background.color:#fc7f03; padding:10px">
    <h4 style="color:white;text_align:center;">{header}</h5>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


def space_header():
    """
    function to create space using html

    Parameters
    ----------
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    function to display major headers at user interface

    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style="background.color:#fc7f03; padding:15px">
    <h2 style="color:white; text_align:center;">{main_txt}</h2>
    <p style="color:white; text_align:center;">{sub_txt}</p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)


def display_side_panel_header(txt):
    """
    function to display minor headers at side panel

    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')