import pandas as pd
import os
import sqlite3 as sl
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# import optuna


def get_df_list(data_names: str) -> list[pd.DataFrame]:
    data_names_list = data_names.split('\n')
    for idx_name in range(len(data_names_list)):
        data_names_list[idx_name] = os.path.join('uploads', data_names_list[idx_name])
    df_list = []
    for path_name in data_names_list:
        df = pd.read_csv(path_name)
        df_list.append(df)
    del df


# my_db = sl.connect(os.path.join('database', 'drilling_mixture.db'))
# print(my_db.__sizeof__())
#
# try:
#     with my_db:
#         my_db.execute("""
#         CREATE TABLE USER (
#         id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
#         name TEXT
#         );
#         """)
# except sl.OperationalError:
#     pass
# print(my_db.__sizeof__())
#
# sql = 'INSERT INTO USER (id, name) values(?, ?)'
# data = [
#     (1, 'Alice'),
#     (2, 'Bob'),
#     (3, 'Chris')
# ]
# with my_db:
#     my_db.executemany(sql, data)
#
# print(my_db.__sizeof__())

# "SELECT id FROM USER ORDER BY id DESC LIMIT 1" - lastID
# with my_db:
#     data = my_db.execute("SELECT id FROM USER ORDER BY id DESC LIMIT 1")
#     for row in data:
#         print(row[0])

def get_linear_reg_score(df: pd.DataFrame, x_names: list, y_name: str) -> float:
    df = df.loc[1:]
    X = df[x_names]
    y = df[y_name]
    # X, X_test, y, y_test = train_test_split(X, y, random_state=19, test_size=0.2)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    reg = DecisionTreeRegressor(max_depth=3).fit(X, y)
    # reg = DecisionTreeRegressor(max_depth=4)
    cv_score = cross_val_score(reg, X, y, cv=3, scoring='r2')
    print(f'cross_val score: {cv_score}')

    print(f'Train R2: {reg.score(X, y)}')
    print(f'Train R: {np.sqrt(reg.score(X, y))}')

    # X_test = scaler.transform(X_test)
    # print(f'Test R2: {reg.score(X_test, y_test)}')
    # print(f'Test R: {np.sqrt(reg.score(X_test, y_test))}')

    # print(reg.predict(X))
    # print(reg.predict(X_test))

    return reg


if __name__ == '__main__':
    # data = pd.read_excel('uploads/БАШ_НИПИ_новая_форма.xlsx', skiprows=0, sheet_name=0)
    # x = ['Пластовая нефть, %', 'Хлорид натрия NaCl, %', 'Ангидрит (CaSO4), %', 'Модельная пластовая вода (МПВ)']
    # y = 'Фильтрация API, мл/30 мин'  # 'Фильтрация API, мл/30 мин'
    # y = data.columns[-8:].to_list()
    # res_model = get_linear_reg_score(data, x, y)
    df = pd.read_excel("C:\\Users\\Aleksei\\Downloads\\Набор_данных_БашНИПИ_+СФУ+4столбца.xlsx", skiprows=2, sheet_name=0)
    for j in range(df.shape[1]):
        if j < 16:
            df.iloc[:, j] = df.iloc[:, j].fillna(0)
        else:
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].median())


