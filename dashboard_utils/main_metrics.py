import datetime

import streamlit as st
import wandb

from dashboard_utils.time_tracker import _log, simple_time_tracker

WANDB_REPO = "learning-at-home/dalle-hivemind"
CACHE_TTL = 100


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_main_metrics():
    api = wandb.Api()
    runs = api.runs(WANDB_REPO)
    run = runs[0]
    history = run.scan_history(keys=["step", "loss", "alive peers", "_timestamp"])

    steps = []
    losses = []
    alive_peers = []
    dates = []
    for row in history:
        steps.append(row["step"])
        losses.append(row["loss"])
        alive_peers.append(row["alive peers"])
        dates.append(datetime.datetime.utcfromtimestamp(row["_timestamp"]))

    return steps, dates, losses, alive_peers
