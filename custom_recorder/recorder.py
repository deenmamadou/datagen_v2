import os
import streamlit as st
import streamlit.components.v1 as components

# Tell Streamlit where the frontend lives
_component_func = components.declare_component(
    "audio_recorder_5s",
    path=os.path.join(os.path.dirname(__file__), "frontend")
)

def audio_recorder_5s(key=None):
    """Returns recorded WAV bytes after user stops speaking."""
    return _component_func(key=key, default=None)

