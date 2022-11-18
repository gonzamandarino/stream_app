# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:26:17 2022

@author: tomas.maguire
"""

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import shap
from streamlit_shap import st_shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

## Titulo

st.header('MedXplain :medical_symbol:')

# Cuerpo

st.write('Please, select your patient:')

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df
df = pd.read_csv('file1.csv').drop(columns=['Unnamed: 0'])
st.dataframe(filter_dataframe(df))
scaler = StandardScaler()
scaler.fit(df)
model = tf.keras.models.load_model('saved_model/my_model')
explainer = shap.KernelExplainer(model.predict,scaler.transform(df))
#shap_values = explainer.shap_values(scaler.transform(df),nsamples=20)
#shap_final = shap_values[0]


shap_final = pd.read_csv('shap.csv').to_numpy()
#explainer = pd.read_csv('explainer.csv').to_numpy()

features = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'radius_se',
 'perimeter_se',
 'area_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']


form = st.form("template_form")
student = form.text_input("Enter patient name")
submit = form.form_submit_button("Search")

if submit:
	student = int(student)
	#explainer = explainer
	#shap_final = shap_final
	try:
		st.dataframe(df.iloc[[student]])
		st.success("Your patient was found")
		shap_final = shap_final
		st.write('*Overall model explainability:*')
		st_shap(shap.summary_plot(shap_final,df,feature_names=features))
		st.write('*Patient outcome explainability:*')
		st_shap(shap.force_plot(explainer.expected_value, shap_final[student,:],df.values[student,:],feature_names=features,matplotlib=False,link="logit"))
	except:
		st.error('Patient not found')
		
		   
