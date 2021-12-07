import pandas as pd
import streamlit as st
import wandb

from dashboard_utils.bubbles import get_global_metrics, get_new_bubble_data, get_leaderboard
from dashboard_utils.main_metrics import get_main_metrics
from streamlit_observable import observable
import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Only need to set these here as we are add controls outside of Hydralit, to customise a run Hydralit!
st.set_page_config(page_title="Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'>Dashboard</h1>", unsafe_allow_html=True)

key_figures_margin_left, key_figures_c1, key_figures_c2, key_figures_c3, key_figures_margin_right = st.columns(
    (2, 1, 1, 1, 2)
)
chart_c1, chart_c2 = st.columns((3, 2))

lottie_url_loading = "https://assets5.lottiefiles.com/packages/lf20_OdNgAj.json"
lottie_loading = load_lottieurl(lottie_url_loading)


with key_figures_c1:
    st.caption("\# of contributing users")
    placeholder_key_figures_c1 = st.empty()
    with placeholder_key_figures_c1:
        st_lottie(lottie_loading, height=100, key="loading_key_figure_c1")

with key_figures_c2:
    st.caption("\# active users")
    placeholder_key_figures_c2 = st.empty()
    with placeholder_key_figures_c2:
        st_lottie(lottie_loading, height=100, key="loading_key_figure_c2")

with key_figures_c3:
    st.caption("Total runtime")
    placeholder_key_figures_c3 = st.empty()
    with placeholder_key_figures_c3:
        st_lottie(lottie_loading, height=100, key="loading_key_figure_c3")

with chart_c1:
    st.subheader("Metrics over time")
    st.caption("Training Loss")
    placeholder_chart_c1_1 = st.empty()
    with placeholder_chart_c1_1:
        st_lottie(lottie_loading, height=100, key="loading_c1_1")

    st.caption("Number of alive runs over time")
    placeholder_chart_c1_2 = st.empty()
    with placeholder_chart_c1_2:
        st_lottie(lottie_loading, height=100, key="loading_c1_2")

    st.caption("Number of steps")
    placeholder_chart_c1_3 = st.empty()
    with placeholder_chart_c1_3:
        st_lottie(lottie_loading, height=100, key="loading_c1_3")

with chart_c2:
    st.subheader("Global metrics")
    st.caption("Collaborative training participants")
    placeholder_chart_c2_1 = st.empty()
    with placeholder_chart_c2_1:
        st_lottie(lottie_loading, height=100, key="loading_c2_1")
    
    st.write("Chart showing participants of the collaborative-training. Circle radius is relative to the total number of "
    "processed batches, the profile picture is circled in purple if the participant is active. Every purple square represents an "
    "active device.")

    st.caption("Leaderboard")
    placeholder_chart_c2_3 = st.empty()
    with placeholder_chart_c2_3:
        st_lottie(lottie_loading, height=100, key="loading_c2_2")


wandb.login(anonymous="must")


steps, dates, losses, alive_peers = get_main_metrics()
source = pd.DataFrame({"steps": steps, "loss": losses, "alive participants": alive_peers, "date": dates})


placeholder_chart_c1_1.vega_lite_chart(
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

placeholder_chart_c1_2.vega_lite_chart(
    source,
    {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Alive sessions",
        "mark": {"type": "line", "point": {"tooltip": True, "filled": False, "strokeOpacity": 0}},
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "alive participants", "type": "quantitative"},
        },
        "config": {"axisX": {"labelAngle": -40}},
    },
    use_container_width=True,
)
placeholder_chart_c1_3.vega_lite_chart(
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

serialized_data, profiles = get_new_bubble_data()
df_leaderboard = get_leaderboard(serialized_data)
observable(
    "_",
    notebook="d/9ae236a507f54046",  # "@huggingface/participants-bubbles-chart",
    targets=["c_noaws"],
    redefine={"serializedData": serialized_data, "profileSimple": profiles, "width": 0},
    render_empty=True,
)
placeholder_chart_c2_3.dataframe(df_leaderboard[["User", "Total time contributed"]])

global_metrics = get_global_metrics(serialized_data)

placeholder_key_figures_c1.write(f"<b>{global_metrics['num_contributing_users']}</b>", unsafe_allow_html=True)
placeholder_key_figures_c2.write(f"<b>{global_metrics['num_active_users']}</b>", unsafe_allow_html=True)
placeholder_key_figures_c3.write(f"<b>{global_metrics['total_runtime']}</b>", unsafe_allow_html=True)

with placeholder_chart_c2_1:
    observable(
        "Participants",
        notebook="d/9ae236a507f54046",  # "@huggingface/participants-bubbles-chart",
        targets=["c_noaws"],
        redefine={"serializedData": serialized_data, "profileSimple": profiles},
    )
