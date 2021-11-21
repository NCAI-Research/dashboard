import streamlit as st
import wandb
from streamlit_observable import observable

from dashboard_utils.bubbles import get_new_bubble_data

st.title("Training transformers together dashboard")
st.write("test")
wandb.login(anonymous="must")

serialized_data, profiles = get_new_bubble_data()


observers = observable(
    "Participants",
    notebook="d/9ae236a507f54046",  # "@huggingface/participants-bubbles-chart",
    targets=["c_noaws"],
    # observe=["selectedCounties"]
    redefine={"serializedData": serialized_data, "profileSimple": profiles},
)
