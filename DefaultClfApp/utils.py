"""Utility functions for the streamlit app"""

from google.cloud import aiplatform
import pandas as pd
import json


def endpoint_predict(
    PROJECT: str, REGION: str, INSTANCES: list, ENDPOINT: str
):
    """Request predictions from the deployed custom container model."""
    endpoint = aiplatform.Endpoint(ENDPOINT, project=PROJECT, location=REGION)
    return endpoint.predict(instances=INSTANCES)


def get_test_input(path_json: str = 'test_instances.json') -> pd.DataFrame:
    """Return test input for selected model."""
    try:
        with open(path_json, 'r') as json_file:
            json_data = json.load(json_file)
            instances = json_data.get("instances", [])
            if not instances:
                return None
            df = pd.DataFrame(instances)
            return df
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return None
