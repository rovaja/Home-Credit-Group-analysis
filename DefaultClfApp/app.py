"""User interface streamlit app"""
import streamlit as st
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
from utils import endpoint_predict, get_test_input


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rovaja-ml-playground-470e2d0c0dc6.json"
PROJECT_ID: str = "rovaja-ml3-captsone"
REGION: str = "europe-west1"
ENDPOINT_LOAN_DEFAULT_CAT: str = "projects/8393548517/locations/europe-west1/endpoints/2198785761040400384"
ENDPOINT_LOAN_DEFAULT_XGB: str = "projects/8393548517/locations/europe-west1/endpoints/638851440109944832"


def select_model() -> str:
    """Select which model to use."""
    model_str: str = st.selectbox("Select model to use:", options=[
                                  'Loan default risk classifier CatBoost', 'Loan default risk classifier XGBBoost', 'Loan default risk classifier Stacked'])

    title = "Predict if the loan application has risk of default."
    if model_str == 'Loan default risk classifier CatBoost':
        subtitle = "\nCatBoost model."
    elif model_str == 'Loan default risk classifier XGBBoost':
        subtitle = "\nXGBoost model."
    elif model_str == 'Loan default risk classifier Stacked':
        subtitle = "\nStacked XGBoost and CatBoost model."
    st.subheader(title)
    st.markdown("Selected model: "+subtitle)

    return model_str


# @st.cache_data
def make_prediction(input_data: pd.DataFrame, model: str = ENDPOINT_LOAN_DEFAULT_CAT) -> npt.NDArray:
    """Makes prediction from the input data"""
    prediction_json = endpoint_predict(PROJECT_ID,
                                       REGION,
                                       input_data,
                                       model)
    return prediction_json.predictions[0]


def predict_default_risk(result_proba: dict[str, float]):
    """Creates user output from loan default risk prediction results."""
    preds = result_proba['probability']
    label = result_proba['label']
    if label == 1:
        st.markdown(
            f"""The loan application has been evaluated as risky. The probability of defaulting is {(preds):.2%}."""
        )
    else:
        st.markdown(
            f"""The loan application has been evaluated as not risky. The probability of defaulting is {(preds):.2%}."""
        )


def stacked_prediction(result_proba_1: dict[str, float], result_proba_2: dict[str, float]):
    """Creates user output from stacked default risk prediction results."""
    preds = 0.5*(result_proba_1['probability'] + result_proba_2['probability'])
    label = int(preds > 0.5)
    if label == 1:
        st.markdown(
            f"""The loan application has been evaluated as risky. The probability of defaulting is {(preds):.2%}."""
        )
    else:
        st.markdown(
            f"""The loan application has been evaluated as not risky. The probability of defaulting is {(preds):.2%}."""
        )


def app():
    st.title("Default risk estimator")
    model_str = select_model()
    test_index: int = st.slider(
        "Select test instance for model:", min_value=0, max_value=9, value=0)
    if model_str == 'Loan default risk classifier CatBoost':
        test_input = get_test_input()
        test_input = test_input.iloc[[test_index], :]
        st.markdown(
            """
            The Default risk estimator is in testing phase and users can not give inputs for this model due to large number of input features.
            The Default risk estimator has premade inputs that can be used to test the models.
            """)
        if st.button('Make prediction'):
            st.markdown("## Given input:")
            st.dataframe(test_input)
            st.markdown("## Prediction")
            result_proba = make_prediction(test_input.to_dict(
                orient='records'), model=ENDPOINT_LOAN_DEFAULT_CAT)
            predict_default_risk(result_proba)

    elif model_str == 'Loan default risk classifier XGBBoost':
        test_input = get_test_input()
        test_input = test_input.iloc[[test_index], :]
        st.markdown(
            """
            The Default risk estimator is in testing phase and users can not give inputs for this model due to large number of input features.
            The Default risk estimator has premade inputs that can be used to test the models.
            """)
        if st.button('Make prediction'):
            st.markdown("## Given input:")
            st.dataframe(test_input)
            st.markdown("## Prediction")
            result_proba = make_prediction(test_input.to_dict(
                orient='records'), model=ENDPOINT_LOAN_DEFAULT_XGB)
            predict_default_risk(result_proba)

    else:
        test_input = get_test_input()
        test_input = test_input.iloc[[test_index], :]
        st.markdown(
            """
            The Default risk estimator is in testing phase and users can not give inputs for this model due to large number of input features.
            The Default risk estimator has premade inputs that can be used to test the models.
            """)
        if st.button('Make prediction'):
            st.markdown("## Given input:")
            st.dataframe(test_input)
            st.markdown("## Prediction")
            result_proba_1 = make_prediction(test_input.to_dict(
                orient='records'), model=ENDPOINT_LOAN_DEFAULT_CAT)
            result_proba_2 = make_prediction(test_input.to_dict(
                orient='records'), model=ENDPOINT_LOAN_DEFAULT_XGB)
            stacked_prediction(result_proba_1, result_proba_2)


app()
