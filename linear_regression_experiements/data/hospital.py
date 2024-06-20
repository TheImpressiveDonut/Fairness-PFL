import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def pre_process(df: pd.DataFrame) -> pd.DataFrame:
    int_column = ["Length of Stay"]

    for col in int_column:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        df.dropna(subset=[col], inplace=True)

    category_column = ["Age Group", "Gender", "Race", "Ethnicity", "Patient Disposition", "Discharge Year",
                       "Type of Admission", "APR Severity of Illness Description", "APR Risk of Mortality",
                       "Abortion Edit Indicator", "Emergency Department Indicator", "APR Risk of Mortality",
                       "APR Medical Surgical Description", "Payment Typology 1", "Payment Typology 2",
                       "Payment Typology 3"]
    for col in category_column:
        df.dropna(subset=[col], inplace=True)
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

    df = df.drop(
        ["Hospital County", "Operating Certificate Number", "Facility Id",
         "Zip Code - 3 digits",
         "CCS Diagnosis Description", "CCS Procedure Description", "APR MDC Description", "APR DRG Description",
         "Total Charges", "Total Costs"], axis=1)

    cols = list(df.columns)
    cols.remove("Facility Name")
    cols.remove("Health Service Area")
    scaler = MinMaxScaler()
    df[cols] = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

    return df


def clean(df_h: pd.DataFrame) -> pd.DataFrame:
    df_h = df_h.drop(
        ["Health Service Area", "Facility Name"], axis=1)

    return df_h


def get_hospital_dataset() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if not os.path.exists('hospital_data'):
        os.makedirs('hospital_data')

        df = pd.read_csv(
            'data/datasets/hospital_discharge/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015_20240615.csv',
            delimiter=',', low_memory=False)
        df = pre_process(df)
        df_per_hospital = {}

        for name, val in df.groupby(["Health Service Area", "Facility Name"]):
            val = clean(val)
            if val.iloc[:, 0].count() > 500:
                df_per_hospital[name] = val
        keys = list(df_per_hospital.keys())
        num_data = list(map(lambda x: x["Length of Stay"].count(), df_per_hospital.values()))
        indices = np.argsort(num_data)
        keys = [keys[idx] for idx in indices[::-1]]
        N = len(keys)
        train_size = 0.8

        Xs_train = []
        Xs_test = []
        ys_train = []
        ys_test = []

        for idx in range(N):
            data = df_per_hospital[keys[idx]]
            data = data.sample(frac=1, random_state=42).reset_index(drop=True)
            train_size_idx = int(train_size * len(data))
            #print(keys[idx], train_size_idx)
            y = data["Length of Stay"]
            X = data.drop(["Length of Stay"], axis=1)
            ys_train.append(y.iloc[:train_size_idx].to_numpy())
            ys_test.append(y.iloc[train_size_idx:].to_numpy())
            Xs_train.append(X.iloc[:train_size_idx, :].to_numpy())
            Xs_test.append(X.iloc[train_size_idx:, :].to_numpy())

        with open('hospital_data/hospital_data.pkl', 'wb') as f:
            pickle.dump((Xs_train, Xs_test, ys_train, ys_test), f)

    with open('hospital_data/hospital_data.pkl', 'rb') as f:
        Xs_train, Xs_test, ys_train, ys_test = pickle.load(f)

    return Xs_train, Xs_test, ys_train, ys_test
