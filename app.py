import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Academic Performance Change Predictor", layout="centered")
st.title("📚 Academic Performance Change Predictor")
st.write("Predict the likelihood of academic performance changes based on mental health and lifestyle factors.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Garvitpujari/Depression-/main/Student%20Mental%20Health%20Analysis%20During%20Online%20Learning.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df = df.drop(columns=["Name"])  # Drop the 'Name' column
    return df

# Load and prepare data
df = load_data()
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=["Academic Performance Change"])
y = df_encoded["Academic Performance Change"]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# User input form
st.subheader("📝 Fill in your information:")
user_input = {}
for col in X.columns:
    if df[col].dtype == 'object':
        user_input[col] = st.selectbox(col, sorted(df[col].unique()))
    else:
        user_input[col] = st.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].mean()))

# Predict
if st.button("🔮 Predict Academic Change"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = le.fit_transform(input_df[col])

    input_df = input_df[X.columns]
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
