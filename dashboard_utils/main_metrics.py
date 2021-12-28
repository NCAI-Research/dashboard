import datetime

import streamlit as st
import wandb

from dashboard_utils.time_tracker import _log, simple_time_tracker

WANDB_RUN_URL = st.secrets["WANDB_RUN_URL_MAIN_METRICS"] 
CACHE_TTL = 100


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_main_metrics():
    api = wandb.Api()
    run = api.run(WANDB_RUN_URL)
    history = run.history(keys=["step", "loss", "alive peers", "_timestamp"])

    steps = []
    losses = []
    alive_peers = []
    dates = []
    for _, row in history.iterrows():
        steps.append(row["step"])
        losses.append(row["loss"])
        alive_peers.append(row["alive peers"])
        dates.append(datetime.datetime.utcfromtimestamp(row["_timestamp"]))

    return steps, dates, losses, alive_peers
