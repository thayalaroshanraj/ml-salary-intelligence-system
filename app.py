import streamlit as st
import joblib
import numpy as np

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #c084fc, #fbcfe8, #ffffff);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("💼 Data Science Salary Predictor")


# Top heading (move this ABOVE everything)
st.markdown("## 🚀 Predict your salary instantly using Machine Learning")

st.write("💡 Select your role, experience, and location to estimate your salary and required skills.")
st.write("---")


# Load model
model = joblib.load("model.pkl")

# Mappings (IMPORTANT)
experience_map = {"Entry": 0, "Mid": 1, "Senior": 2, "Executive": 3}
job_map = {
    "Data Scientist": 0,
    "ML Engineer": 1,
    "Data Analyst": 2,
    "AI Engineer": 3
}
location_map = {
    "USA": 84,
    "India": 29,
    "UK": 77,
    "Germany": 22
}

skill_map = {
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL"],
    "Data Analyst": ["Excel", "SQL", "Power BI", "Python"],
    "ML Engineer": ["Python", "TensorFlow", "Deep Learning", "MLOps"],
    "AI Engineer": ["Python", "NLP", "Deep Learning", "PyTorch"]
}



# UI Inputs
experience = st.selectbox("Experience Level", list(experience_map.keys()))
job = st.selectbox("Job Title", list(job_map.keys()))
location = st.selectbox("Company Location", list(location_map.keys()))
remote = st.slider("Remote Ratio (%)", 0, 100)

# Convert to model input
input_data = np.array([[
    2025,
    experience_map[experience],
    0,
    job_map[job],
    location_map[location],
    remote,
    1,
    1
]])

if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    
    st.markdown("### 💡 Use this tool to estimate market salary before applying for jobs.")
    
    st.success(f"💰 Estimated Salary: ${int(prediction[0])}")

    st.markdown("### 📊 This model predicts salary based on experience, role, and location.")

    st.info("📌 Note: Prediction is based on historical data trends. Actual salary may vary based on company and skills.")

    st.markdown("### 📊 Insights")
    st.write("- Job role and experience have highest impact on salary")
    st.write("- Remote work has smaller influence compared to role")

    st.markdown("### 🧠 What this means")
    st.write(f"For a {experience} level {job} in {location}, the expected salary is around ${int(prediction[0])}.")

    st.markdown("### 🧠 Recommended Skills")
    skills = skill_map.get(job, ["Python", "SQL"])
    for skill in skills:
        st.write(f"✔ {skill}")

    low = int(prediction[0] * 0.9)
    high = int(prediction[0] * 1.1)

    st.markdown("### 💰 Expected Salary Range")
    st.write(f"${low} - ${high}")