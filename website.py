import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to calculate PCA-based Fraudulent Score
def calculate_pca_fraud_score(data, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    pca = PCA(n_components=1)
    fraud_score = pca.fit_transform(scaled_data).flatten()
    return fraud_score

# Load the dataset
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Predict using the pre-trained XGBoost model
def predict_fraud_score(model_path, features, data):
    try:
        dmatrix = xgb.DMatrix(data[features])
        model = xgb.Booster()
        model.load_model(model_path)
        return model.predict(dmatrix)
    except Exception as e:
        st.error(f"Error loading or predicting with the model: {e}")
        return None

# Streamlit app
st.title("Fraudulent Score Predictor")
st.write("Upload a dataset and select attributes to calculate PCA Fraudulent Scores and predict using XGBoost.")

# File upload option
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # List of attributes
    attributes = data.columns.tolist()

    # Select features for PCA and model prediction
    selected_features = st.multiselect("Select attributes for PCA and prediction:", attributes)

    if selected_features:
        st.write("Selected Attributes:", selected_features)

        # Calculate PCA-based Fraudulent Score
        pca_scores = calculate_pca_fraud_score(data, selected_features)
        data['PCA_Fraudulent_Score'] = pca_scores
        st.write("Calculated PCA Fraudulent Score:")
        st.dataframe(data[['PCA_Fraudulent_Score']].head())

        # Predict Fraudulent Score using the pre-trained XGBoost model
        model_path = "xgboost_model.model"
        predicted_scores = predict_fraud_score(model_path, selected_features, data)
        if predicted_scores is not None:
            data['Predicted_Fraudulent_Score'] = predicted_scores
            st.write("Predicted Fraudulent Score using XGBoost:")
            st.dataframe(data[['Predicted_Fraudulent_Score']].head())

            # Download option for updated dataset
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Updated Dataset",
                data=csv,
                file_name="updated_dataset_with_scores.csv",
                mime="text/csv",
            )
    else:
        st.warning("Please select at least one attribute to proceed.")
