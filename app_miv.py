# Most Important Variables – MVP ver_1.xx
# Basic version with core functionality (upload → model → results)
# No advanced options or UI refinements yet

# ver_1.09

import streamlit as st
import pandas as pd
from PIL import Image
from pycaret.classification import (
    setup as setup_clf,
    compare_models as compare_models_clf,
    plot_model as plot_model_clf,
    pull as pull_clf,
    predict_model as predict_model_clf,
)
from pycaret.regression import (
    setup as setup_reg,
    compare_models as compare_models_reg,
    plot_model as plot_model_reg,
    pull as pull_reg,
    predict_model as predict_model_reg,
)


# Helper function to detect separator
def detect_separator(uploaded_file, header_option):
    """Try common separators and choose the one producing most columns."""
    separators = [",", ";", "\t", "|"]
    best_separator = ","
    max_cols = 0

    for separator in separators:
        uploaded_file.seek(0)  # reset file pointer
        try:
            df_test = pd.read_csv(uploaded_file, sep=separator, header=header_option, nrows=6)
            num_cols = df_test.shape[1]
            if num_cols > max_cols:
                max_cols = num_cols
                best_separator = separator
        except Exception:
            continue

    uploaded_file.seek(0)  # reset again for actual read
    return best_separator


# Function to upload data
def upload_data_from_user():
    st.subheader("Select Data Source")
    uploaded_file = st.file_uploader("Upload a CSV or DATA file", type=["csv", "data"])

    if "header_present" not in st.session_state:
        st.session_state["header_present"] = True

    header_present = st.checkbox(
        "First row contains column names",
        value=st.session_state["header_present"],
        key="header_present"
    )

    header_option = 0 if header_present else None

    if uploaded_file is not None:
        try:
            detected_sep = detect_separator(uploaded_file, header_option)
            st.info(f"Auto-detected separator: '{detected_sep}'")
            dataset = pd.read_csv(uploaded_file, sep=detected_sep, header=header_option)
            st.success("File loaded successfully.")
            st.write(dataset.head())  # display preview of data
            st.session_state["dataset"] = dataset  # save dataset in session state

            # Reset model-related session states on new upload
            st.session_state.pop("best_model", None)
            st.session_state.pop("leaderboard", None)
            st.session_state.pop("show_results_clicked", None)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a file to proceed.")


# Function to configure model options and generate model
def configure_model_options():
    
    st.subheader("Model Configuration Options")

    # Placeholder for future advanced options
    model_conf_text = """
    This is a placeholder for future model configurations.
    Additional options like dropping columns, transforming features,
    handling missing values, and other advanced settings will be
    introduced in upcoming versions.
    """

    st.info(model_conf_text)

    # TODO: Replace placeholder with configurations of model (v3.x)

    if "dataset" in st.session_state:
        dataset = st.session_state["dataset"]

        if "target_column" not in st.session_state:
            st.session_state["target_column"] = dataset.columns[0]

        target_column = st.selectbox(
            "Select the target column for your model:",
            options=dataset.columns,
            index=dataset.columns.get_loc(st.session_state["target_column"]),
            key="target_column"
        )

        # Determine problem type
        target_dtype = dataset[target_column].dtype
        unique_values = dataset[target_column].nunique()
        threshold = 12

        if target_dtype in ["object", "category"]:
            problem_type = "Classification"
        elif pd.api.types.is_numeric_dtype(target_dtype):
            if unique_values <= threshold:
                problem_type = "Classification"
            else:
                problem_type = "Regression"
        else:
            problem_type = "Unknown"

        st.session_state["problem_type"] = problem_type
        st.info(f"Detected problem type: **{problem_type}**")

        if st.button("Generate Best Model"):
            with st.spinner("Training models... this may take a while"):
                if problem_type == "Classification":
                    s = setup_clf(data=dataset, target=target_column, verbose=False, session_id=123)
                    best_model = compare_models_clf()
                    leaderboard_df = pull_clf("leaderboard")
                elif problem_type == "Regression":
                    s = setup_reg(data=dataset, target=target_column, verbose=False, session_id=123)
                    best_model = compare_models_reg()
                    leaderboard_df = pull_reg("leaderboard")
                else:
                    st.error("Unknown problem type. Cannot generate model.")
                    return

                st.session_state["best_model"] = best_model
                st.session_state["leaderboard"] = leaderboard_df
                st.success("Best model generated successfully.")
                st.write(best_model)
    else:
        st.warning("No dataset loaded. Please upload data first in 'Load Data' section.")


# Function to show model summary (feature importance, leaderboard, predictions)
def show_model_summary():
    st.subheader("Model Summary")

    if "best_model" in st.session_state:
        if st.button("Show Model Results"):
            model = st.session_state["best_model"]

            # Feature Importance
            try:
                if st.session_state["problem_type"] == "Classification":
                    plot_model_clf(model, plot="feature", save=True)
                elif st.session_state["problem_type"] == "Regression":
                    plot_model_reg(model, plot="feature", save=True)
                else:
                    st.error("Unknown problem type. Cannot plot feature importance.")
                    return

                img = Image.open("Feature Importance.png")
                st.image(img, caption="Feature Importance", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating feature importance plot: {e}")

            # Leaderboard
            st.write("### Model Leaderboard")
            leaderboard_df = st.session_state.get("leaderboard")
            if leaderboard_df is not None:
                st.dataframe(leaderboard_df)
            else:
                st.warning("Leaderboard not available.")

            # Predictions
            st.write("### Predictions on Dataset")
            dataset = st.session_state["dataset"]
            try:
                if st.session_state["problem_type"] == "Classification":
                    predictions_df = predict_model_clf(model, data=dataset)
                else:
                    predictions_df = predict_model_reg(model, data=dataset)
                st.dataframe(predictions_df.head(10))
            except Exception as e:
                st.error(f"Error generating predictions: {e}")

            # Summary Text (placeholder)
            st.write("### Summary")
            summary_text = """
            This is a placeholder for the model summary.
            The actual description will be generated by an AI assistant in future versions.
            """
            st.info(summary_text)

            # TODO: Replace placeholder with LLM-generated summary (v2.x)
    else:
        st.warning("No model generated. Please train a model in 'Model Options' section.")


# Section content dictionary
section_content = {
    "Load Data": [":open_file_folder:", upload_data_from_user],
    "Model Options": [":gear:", configure_model_options],
    "Model Summary": [":bar_chart:", show_model_summary]
}


# Main run block
with st.sidebar:
    st.header("Navigation Panel")
    navigation_panel = st.radio("Select Action:", list(section_content.keys()))

emoji, render_func = section_content[navigation_panel]
st.title(f"{navigation_panel} {emoji}")
render_func()
