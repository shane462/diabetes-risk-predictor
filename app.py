import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="糖尿病风险预测", page_icon="🩺", layout="centered")
# === 1. 加载模型与预处理器 ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("diabetes_model_tf.h5")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_model()



# === 2. 页面标题 ===
st.title("🩺 糖尿病预测系统（基于神经网络模型）")
st.write("请输入以下信息，系统将预测患糖尿病的概率。")

# === 3. 构建输入界面 ===

# 数值型特征
age = st.slider("年龄 (age)", 18, 100, 40)
bmi = st.slider("BMI（体重指数）", 10.0, 50.0, 22.0)
hbA1c = st.slider("HbA1c 水平（糖化血红蛋白）", 3.0, 15.0, 5.5)
glucose = st.slider("血糖水平 (blood_glucose_level)", 50, 300, 100)

# 二进制特征
st.subheader("🫀 健康状况")
hypertension = st.selectbox("是否有高血压", ["否", "是"])
heart_disease = st.selectbox("是否有心脏病", ["否", "是"])

# 种族（独热编码）
st.subheader("🌍 种族（Race）")
race = st.selectbox("请选择种族", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"])

# 类别特征
st.subheader("🚻 其他信息")
gender = st.selectbox("性别", ["Male", "Female", "Other"])
location = st.selectbox("所在地区", ["Urban", "Rural"])
smoking_history = st.selectbox("吸烟史", ["never", "current", "former", "ever", "not current", "No Info"])

# === 4. 整理输入成 DataFrame ===
input_dict = {
    'age': [age],
    'bmi': [bmi],
    'hbA1c_level': [hbA1c],
    'blood_glucose_level': [glucose],
    'hypertension': [1 if hypertension == "是" else 0],
    'heart_disease': [1 if heart_disease == "是" else 0],
    'gender': [gender],
    'location': [location],
    'smoking_history': [smoking_history],
    'race:AfricanAmerican': [1 if race == "AfricanAmerican" else 0],
    'race:Asian': [1 if race == "Asian" else 0],
    'race:Caucasian': [1 if race == "Caucasian" else 0],
    'race:Hispanic': [1 if race == "Hispanic" else 0],
    'race:Other': [1 if race == "Other" else 0],
}

df_input = pd.DataFrame(input_dict)

# === 5. 预测逻辑 ===
if st.button("🔍 预测糖尿病概率"):
    try:
        Xp = preprocessor.transform(df_input)
        prob = model.predict(Xp).ravel()[0]
        st.success(f"模型预测患糖尿病的概率为：**{prob:.2%}**")

        if prob > 0.7:
            st.error("⚠️ 高风险，请考虑进行健康检查。")
        elif prob > 0.4:
            st.warning("🟠 中等风险，请保持健康生活方式。")
        else:
            st.info("🟢 风险较低，请继续保持。")

    except Exception as e:
        st.error(f"预测出错：{e}")

# === 6. 侧边栏说明 ===
st.sidebar.header("📘 使用说明")
st.sidebar.write("""
该模型基于神经网络（TensorFlow）构建，使用多个健康特征预测糖尿病风险。  
输入值仅用于演示，不构成医学诊断。
""")
