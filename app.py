import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Load dataset
csv_url = "https://raw.githubusercontent.com/Garvitpujari/Mental_Health_/main/cleaned_mental_health_dataset%20(1).csv"
df = pd.read_csv(csv_url)

# Target column
target_column = "Depression"
features = [col for col in df.columns if col != target_column]

X = df[features]
y = df[target_column]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# Prediction function
def predict_depression(Gender, Age, Sleep_Duration, Dietary_Habits,
                       Suicidal_Thoughts, Financial_Stress,
                       Family_History):
    input_dict = {
        'Gender_Male': 1 if Gender == "Male" else 0,
        'Age': Age,
        'Sleep Duration_7-8 hours': 1 if Sleep_Duration == "7-8 hours" else 0,
        'Sleep Duration_Less than 5 hours': 1 if Sleep_Duration == "Less than 5 hours" else 0,
        'Sleep Duration_More than 8 hours': 1 if Sleep_Duration == "More than 8 hours" else 0,
        'Dietary Habits_Moderate': 1 if Dietary_Habits == "Moderate" else 0,
        'Dietary Habits_Unhealthy': 1 if Dietary_Habits == "Unhealthy" else 0,
        'Have you ever had suicidal thoughts ?_Yes': 1 if Suicidal_Thoughts == "Yes" else 0,
        'Financial Stress': {"No stress":0, "Some stress":1, "High stress":2}[Financial_Stress],
        'Family History of Mental Illness_Yes': 1 if Family_History == "Yes" else 0
    }
    input_df = pd.DataFrame([input_dict]).reindex(columns=X_encoded.columns, fill_value=0)
    pred = model.predict(input_df)[0]
    return "Result: 0 ✅ No significant depression" if pred==0 else "Result: 1 ⚠️ Possible depression"

# Gradio interface
iface = gr.Interface(
    fn=predict_depression,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Radio(["Less than 5 hours", "7-8 hours", "More than 8 hours"], label="Sleep Duration"),
        gr.Radio(["Healthy", "Moderate", "Unhealthy"], label="Dietary Habits"),
        gr.Radio(["Yes", "No"], label="Have you ever had suicidal thoughts?"),
        gr.Radio(["No stress", "Some stress", "High stress"], label="Financial Stress"),
        gr.Radio(["Yes", "No"], label="Family History of Mental Illness")
    ],
    outputs="text",
    title="Depression Predictor"
)

if __name__ == "__main__":
    iface.launch()
