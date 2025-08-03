import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess dataset
def load_data():
    url = "https://raw.githubusercontent.com/Garvitpujari/Depression-/main/Student%20Mental%20Health%20Analysis%20During%20Online%20Learning.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df = df.drop(columns=["Name"])  # Drop Name column
    return df

df = load_data()

# Encode categorical features
le_dict = {}
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

X = df_encoded.drop(columns=["Academic Performance Change"])
y = df_encoded["Academic Performance Change"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction function
def predict(gender, age, edu, screen_time, sleep, activity, stress, anxiety):
    input_data = {
        "Gender": [gender],
        "Age": [int(age)],
        "Education Level": [edu],
        "Screen Time (hrs/day)": [float(screen_time)],
        "Sleep Duration (hrs)": [float(sleep)],
        "Physical Activity (hrs/week)": [float(activity)],
        "Stress Level": [int(stress)],
        "Anxious Before Exams": [anxiety],
    }

    input_df = pd.DataFrame(input_data)

    # Encode string fields
    for col in input_df.select_dtypes(include="object").columns:
        le = le_dict[col]
        input_df[col] = le.transform(input_df[col])

    prediction = model.predict(input_df)[0]
    return f"📊 Predicted Academic Performance Change: {prediction}"

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(choices=df["Gender"].unique().tolist(), label="Gender"),
        gr.Slider(minimum=int(df["Age"].min()), maximum=int(df["Age"].max()), value=int(df["Age"].mean()), label="Age"),
        gr.Radio(choices=df["Education Level"].unique().tolist(), label="Education Level"),
        gr.Slider(0, 24, step=0.5, value=4, label="Screen Time (hrs/day)"),
        gr.Slider(0, 24, step=0.5, value=6, label="Sleep Duration (hrs)"),
        gr.Slider(0, 20, step=1, value=3, label="Physical Activity (hrs/week)"),
        gr.Slider(1, 10, step=1, value=5, label="Stress Level"),
        gr.Radio(choices=df["Anxious Before Exams"].unique().tolist(), label="Anxious Before Exams")
    ],
    outputs="text",
    title="🎓 Academic Performance Change Predictor",
    description="Enter your lifestyle and mental health stats to predict if your academic performance may change.",
)

if __name__ == "__main__":
    demo.launch()
