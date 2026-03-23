# ML Salary Intelligence System

An end-to-end machine learning system that predicts data science salaries, analyzes key influencing factors, and provides actionable career insights including skill recommendations through an interactive web application.

---

## Features

- Salary prediction using machine learning models  
- Salary range estimation for better decision-making  
- Skill recommendations based on selected job roles  
- Feature importance analysis to identify key salary drivers  
- Interactive web application built with Streamlit  

---

## Problem Statement

Understanding salary expectations and required skills in data science roles can be challenging.  
This project provides a data-driven approach to estimate salaries and guide users toward relevant skills based on role, experience, and location.

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## Project Workflow

1. Data collection and preprocessing  
2. Feature engineering and encoding  
3. Model training using Random Forest and Linear Regression  
4. Model evaluation using MAE and MSE  
5. Deployment as an interactive web application  

---

## Key Insights

- Job role and experience level are the most influential factors  
- Location significantly impacts salary  
- Remote work has comparatively lower influence  

---

## Model Performance

- Mean Absolute Error (MAE): ~45,000  
- Mean Squared Error (MSE): ~3.8 Billion  

---

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
