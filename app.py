import pandas as pd
import streamlit as st
import wandb

from dashboard_utils.bubbles import get_new_bubble_data
from dashboard_utils.main_metrics import get_main_metrics
from streamlit_observable import observable

# Only need to set these here as we are add controls outside of Hydralit, to customise a run Hydralit!
st.set_page_config(page_title="Dashboard", layout="centered")

wandb.login(anonymous="must")

st.markdown("<h1 style='text-align: center;'>Dashboard</h1>", unsafe_allow_html=True)
st.caption("Training Loss")

steps, dates, losses, alive_peers = get_main_metrics()
source = pd.DataFrame({"steps": steps, "loss": losses, "alive participants": alive_peers, "date": dates})


st.vega_lite_chart(
    source,
    {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Training Loss",
        "mark": {"type": "line", "point": {"tooltip": True, "filled": False, "strokeOpacity": 0}},
        "encoding": {"x": {"field": "date", "type": "temporal"}, "y": {"field": "loss", "type": "quantitative"}},
        "config": {"axisX": {"labelAngle": -40}},
    },
    use_container_width=True,
)

st.caption("Number of alive runs over time")
st.vega_lite_chart(
    source,
    {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Alive participants",
        "mark": {"type": "line", "point": {"tooltip": True, "filled": False, "strokeOpacity": 0}},
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "alive participants", "type": "quantitative"},
        },
        "config": {"axisX": {"labelAngle": -40}},
    },
    use_container_width=True,
)
st.caption("Number of steps")
st.vega_lite_chart(
    source,
    {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Training Loss",
        "mark": {"type": "line", "point": {"tooltip": True, "filled": False, "strokeOpacity": 0}},
        "encoding": {"x": {"field": "date", "type": "temporal"}, "y": {"field": "steps", "type": "quantitative"}},
        "config": {"axisX": {"labelAngle": -40}},
    },
    use_container_width=True,
)

st.header("Collaborative training participants")
serialized_data, profiles = get_new_bubble_data()
observable(
    "Participants",
    notebook="d/9ae236a507f54046",  # "@huggingface/participants-bubbles-chart",
    targets=["c_noaws"],
    redefine={"serializedData": serialized_data, "profileSimple": profiles},
)
