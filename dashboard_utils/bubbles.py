import datetime
from urllib import parse

import requests
import wandb

URL_QUICKSEARCH = "https://huggingface.co/api/quicksearch?"
WANDB_REPO = "learning-at-home/Worker_logs"


def get_new_bubble_data():
    serialized_data_points, latest_timestamp = get_serialized_data_points()
    serialized_data = get_serialized_data(serialized_data_points, latest_timestamp)
    profiles = get_profiles(serialized_data_points)

    return serialized_data, profiles


def get_profiles(serialized_data_points):
    profiles = []
    for username in serialized_data_points.keys():
        params = {"type": "user", "q": username}
        new_url = URL_QUICKSEARCH + parse.urlencode(params)
        r = requests.get(new_url)
        response = r.json()
        try:
            avatarUrl = response["users"][0]["avatarUrl"]
        except:
            avatarUrl = "/avatars/57584cb934354663ac65baa04e6829bf.svg"
        if avatarUrl.startswith("/avatars/"):
            avatarUrl = f"https://huggingface.co{avatarUrl}"
        profiles.append(
            {"id": username, "name": username, "src": avatarUrl, "url": f"https://huggingface.co/{username}"}
        )
    return profiles


def get_serialized_data_points():

    api = wandb.Api()
    runs = api.runs(WANDB_REPO)

    serialized_data_points = {}
    latest_timestamp = None
    print("**start api call")
    for run in runs:
        run_summary = run.summary._json_dict
        run_name = run.name

        if run_name in serialized_data_points:
            try:
                timestamp = run_summary["_timestamp"]
                serialized_data_points[run_name]["Runs"].append(
                    {
                        "batches": run_summary["_step"],
                        "runtime": run_summary["_runtime"],
                        "loss": run_summary["train/loss"],
                        "velocity": run_summary["_step"] / run_summary["_runtime"],
                        "date": datetime.datetime.utcfromtimestamp(timestamp),
                    }
                )
                if not latest_timestamp or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
            except Exception as e:
                pass
                # print(e)
                # print([key for key in list(run_summary.keys()) if "gradients" not in key])
        else:
            try:
                timestamp = run_summary["_timestamp"]
                serialized_data_points[run_name] = {
                    "profileId": run_name,
                    "Runs": [
                        {
                            "batches": run_summary["_step"],
                            "runtime": run_summary["_runtime"],
                            "loss": run_summary["train/loss"],
                            "velocity": run_summary["_step"] / run_summary["_runtime"],
                            "date": datetime.datetime.utcfromtimestamp(timestamp),
                        }
                    ],
                }
                if not latest_timestamp or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
            except Exception as e:
                pass
                # print(e)
                # print([key for key in list(run_summary.keys()) if "gradients" not in key])
    latest_timestamp = datetime.datetime.utcfromtimestamp(latest_timestamp)
    print("**finish api call")
    return serialized_data_points, latest_timestamp


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
            if run["date"] == latest_timestamp:
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
            "loss": loss,
            "velocity": velocity,
            "activeRuns": activeRuns,
        }
        serialized_data_points_v2.append(new_item)
    serialized_data = {"points": [serialized_data_points_v2], "maxVelocity": max_velocity}
    return serialized_data
