import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Title
st.set_page_config(page_title="Iris Logistic Classification", layout="centered")

st.title("🌸 Iris Flower Classification")
st.write("Logistic Regression Model using Iris Dataset")

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Flower Details")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width", 0.1, 3.0, 0.2)

# Prediction
input_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

flower_names = iris.target_names

# Result
st.subheader("Prediction Result")

st.success(f"Predicted Flower: {flower_names[prediction[0]].title()}")

# Probability Scores
st.subheader("Prediction Probability")

for i, flower in enumerate(flower_names):
    st.write(f"{flower.title()}: {prediction_proba[0][i]:.2f}")

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.info(f"Accuracy: {accuracy:.2f}")

# Dataset Info
st.subheader("Dataset Information")

st.write("""
- Dataset: Iris Dataset
- Algorithm: Logistic Regression
- Classes:
    - Setosa
    - Versicolor
    - Virginica
""")