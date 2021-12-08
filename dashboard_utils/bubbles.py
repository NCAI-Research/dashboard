import datetime
from concurrent.futures import as_completed
from requests.adapters import HTTPAdapter
from urllib import parse
import requests
import json
import pandas as pd

import streamlit as st
import wandb
from requests_futures.sessions import FuturesSession

from dashboard_utils.time_tracker import _log, simple_time_tracker

URL_QUICKSEARCH = "https://huggingface.co/api/quicksearch?"
WANDB_REPO = st.secrets["WANDB_REPO_INDIVIDUAL_METRICS"]  
CACHE_TTL = 100
MAX_DELTA_ACTIVE_RUN_SEC = 60 * 5


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_new_bubble_data():
    serialized_data_points, latest_timestamp = get_serialized_data_points()
    serialized_data = get_serialized_data(serialized_data_points, latest_timestamp)

    usernames = []
    for item in serialized_data["points"][0]:
        usernames.append(item["profileId"])

    profiles = get_profiles(usernames)

    return serialized_data, profiles


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_profiles(usernames):
    profiles = []
    with FuturesSession() as session:
        futures = []
        for username in usernames:
            future = session.get(URL_QUICKSEARCH + parse.urlencode({"type": "user", "q": username}))
            future.username = username
            futures.append(future)
        for future in as_completed(futures):
            resp = future.result()
            username = future.username
            response = resp.json()
            avatarUrl = None
            if response["users"]:
                for user_candidate in response["users"]:
                    if user_candidate["user"] == username:
                        avatarUrl = response["users"][0]["avatarUrl"]
                        break
            if not avatarUrl:
                avatarUrl = "/avatars/57584cb934354663ac65baa04e6829bf.svg"

            if avatarUrl.startswith("/avatars/"):
                avatarUrl = f"https://huggingface.co{avatarUrl}"

            profiles.append(
                {"id": username, "name": username, "src": avatarUrl, "url": f"https://huggingface.co/{username}"}
            )
    return profiles


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_serialized_data_points():
    url = "https://api.wandb.ai/graphql"
    api = wandb.Api()

    # Get the run ids
    json_query_run_names = {
        "operationName":"WandbConfig",
        "variables":{"limit":100000,"entityName":"learning-at-home","projectName":"dalle-hivemind-trainers","filters":"{\"$and\":[{\"$or\":[{\"$and\":[]}]},{\"$and\":[]},{\"$or\":[{\"$and\":[{\"$or\":[{\"$and\":[]}]},{\"$and\":[{\"name\":{\"$ne\":null}}]}]}]}]}","order":"-state"},
        "query": """ query WandbConfig($projectName: String!, $entityName: String!, $filters: JSONString, $limit: Int = 100, $order: String) {
    project(name: $projectName, entityName: $entityName) {
    id
    runs(filters: $filters, first: $limit, order: $order) {
    edges {
    node {
    id
    name
    __typename
    }
    __typename
    }
    __typename
    }
    __typename
    }
    }
    """}

    s = requests.Session()
    s.mount(url, HTTPAdapter(max_retries=5))

    resp = s.post(
        headers={"User-Agent": api.user_agent, "Use-Admin-Privileges": "true", 'content-type': 'application/json'},
        auth=("api", api.api_key),
        url=url,
        data=json.dumps(json_query_run_names)
    )
    json_metrics = resp.json()
    run_names = [run['node']["name"] for run in json_metrics['data']['project']["runs"]['edges']]

    # Get info of each run
    with FuturesSession() as session:
        futures = []
        for run_name in run_names:
            json_query_by_run = {
                "operationName":"Run",
                "variables":{"entityName":"learning-at-home","projectName":"dalle-hivemind-trainers", "runName":run_name},
                "query":"""query Run($projectName: String!, $entityName: String, $runName: String!) {
                    project(name: $projectName, entityName: $entityName) {
                        id
                        name
                        createdAt
                        run(name: $runName) {
                        id
                        name
                        displayName
                        state
                        summaryMetrics
                        runInfo {
                            gpu
                            }
                            __typename
                        }
                        __typename
                        }
                    }
                    """}

            future = session.post(
                headers={"User-Agent": api.user_agent, "Use-Admin-Privileges": "true", 'content-type': 'application/json'},
                auth=("api", api.api_key),
                url=url,
                data=json.dumps(json_query_by_run)
            )
            futures.append(future)
        
        serialized_data_points = {}
        latest_timestamp = None
        for future in as_completed(futures):
            resp = future.result()
            json_metrics = resp.json()

            data = json_metrics.get("data", None)
            if data is None:
                continue
            
            project = data.get("project", None)
            if project is None:
                continue

            run = project.get("run", None)
            if run is None:
                continue

            runInfo = run.get("runInfo", None) 
            if runInfo is None:
                gpu_type = None
            else:
                gpu_type = runInfo.get("gpu", None)

            summaryMetrics = run.get("summaryMetrics", None)
            if summaryMetrics is not None:
                run_summary = json.loads(summaryMetrics)

            state = run.get("state", None)
            if state is None:
                continue

            displayName = run.get("displayName", None)
            if displayName is None:
                continue

            if displayName in serialized_data_points:
                if "_timestamp" in run_summary and "_step" in run_summary:
                    timestamp = run_summary["_timestamp"]
                    serialized_data_points[displayName]["Runs"].append(
                        {
                            "batches": run_summary["_step"],
                            "runtime": run_summary["_runtime"],
                            "loss": run_summary["train/loss"],
                            "gpu_type": gpu_type,
                            "state": state,
                            "velocity": run_summary["_step"] / run_summary["_runtime"],
                            "date": datetime.datetime.utcfromtimestamp(timestamp),
                        }
                    )
                    if not latest_timestamp or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
            else:
                if "_timestamp" in run_summary and "_step" in run_summary:
                    timestamp = run_summary["_timestamp"]
                    serialized_data_points[displayName] = {
                        "profileId": displayName,
                        "Runs": [
                            {
                                "batches": run_summary["_step"],
                                "gpu_type": gpu_type,
                                "state": state,
                                "runtime": run_summary["_runtime"],
                                "loss": run_summary["train/loss"],
                                "velocity": run_summary["_step"] / run_summary["_runtime"],
                                "date": datetime.datetime.utcfromtimestamp(timestamp),
                            }
                        ],
                    }
                    if not latest_timestamp or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
        latest_timestamp = datetime.datetime.utcfromtimestamp(latest_timestamp)
    return serialized_data_points, latest_timestamp


@st.cache(ttl=CACHE_TTL, show_spinner=False)
@simple_time_tracker(_log)
def get_serialized_data(serialized_data_points, latest_timestamp):
    serialized_data_points_v2 = []
    max_velocity = 1
    for run_name, serialized_data_point in serialized_data_points.items():
        activeRuns = []
        loss = 0
        runtime = 0
        batches = 0
        velocity = 0
        for run in serialized_data_point["Runs"]:
            if run["state"] == "running":
                run["date"] = run["date"].isoformat()
                activeRuns.append(run)
                loss += run["loss"]
                velocity += run["velocity"]
            loss = loss / len(activeRuns) if activeRuns else 0
            runtime += run["runtime"]
            batches += run["batches"]
        new_item = {
            "date": latest_timestamp.isoformat(),
            "profileId": run_name,
            "batches": batches,
            "runtime": runtime,
            "activeRuns": activeRuns,
        }
        serialized_data_points_v2.append(new_item)
    serialized_data = {"points": [serialized_data_points_v2], "maxVelocity": max_velocity}
    return serialized_data


def get_leaderboard(serialized_data):
    data_leaderboard = {"user": [], "runtime": []}

    for user_item in serialized_data["points"][0]:
        data_leaderboard["user"].append(user_item["profileId"])
        data_leaderboard["runtime"].append(user_item["runtime"])

    df = pd.DataFrame(data_leaderboard)
    df = df.sort_values("runtime", ascending=False)
    df["runtime"] = df["runtime"].apply(lambda x: datetime.timedelta(seconds=x))
    df["runtime"] = df["runtime"].apply(lambda x: str(x))

    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"user": "User", "runtime": "Total time contributed"}, inplace=True)
    df["Rank"] = df.index + 1
    df = df.set_index("Rank")
    return df


def get_global_metrics(serialized_data):
    current_time = datetime.datetime.utcnow()
    num_contributing_users = len(serialized_data["points"][0])
    num_active_users = 0
    total_runtime = 0

    for user_item in serialized_data["points"][0]:
        for run in user_item["activeRuns"]:
            date_run = datetime.datetime.fromisoformat(run["date"])
            delta_time_sec = (current_time - date_run).total_seconds()
            if delta_time_sec < MAX_DELTA_ACTIVE_RUN_SEC:
                num_active_users += 1
                break

        total_runtime += user_item["runtime"]

    total_runtime = datetime.timedelta(seconds=total_runtime)
    return {
        "num_contributing_users": num_contributing_users,
        "num_active_users": num_active_users,
        "total_runtime": total_runtime,
    }
