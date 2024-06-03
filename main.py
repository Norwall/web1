import flet as ft
import os
import time, datetime
import pandas as pd
import numpy as np
import zipfile
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flet.matplotlib_chart import MatplotlibChart
import gc
import json
import pickle
import optuna
import shutil
from sklearn2pmml import sklearn2pmml
from doepy3 import build
import webbrowser

optuna.logging.set_verbosity(optuna.logging.WARNING)  # TODO CHANGE to WARNING  /  DEBUG
matplotlib.use("svg")

# DEFAULT_FLET_PATH = ''  # or 'ui/path'
# DEFAULT_FLET_PORT = 8502

def get_cost_br(volume_arr: np.array, cost_arr: np.array) -> float:
    return np.sum(volume_arr * cost_arr)


def obj_func(trial, model_meta, model_list, config, opt_directions, is_step: bool, is_opt_weight: bool, opt_weight, opt_weight_sign):
    is_deadly_penalty = False

    for idx, feat_name in enumerate(config['inputs']['names']):
        if is_step:
            x = trial.suggest_float(feat_name,
                                    config['inputs']['min'][idx],
                                    config['inputs']['max'][idx],
                                    step=config['inputs']['step'][idx])
        else:
            x = trial.suggest_float(feat_name, config['inputs']['min'][idx], config['inputs']['max'][idx])

    res_obj_func = []
    for idx, target_name in enumerate(config['criteria']['names']):
        if target_name != 'Стоимость БР':
            for model_dict in model_list:
                if model_dict['target'] == target_name:
                    input_arr = []
                    for feat_name in config['inputs']['names']:
                        norm_val = trial.params[feat_name] - model_meta[feat_name]['min']
                        if model_meta[feat_name]['min'] != model_meta[feat_name]['max']:
                            norm_val /= (model_meta[feat_name]['max'] - model_meta[feat_name]['min'])
                        input_arr.append(norm_val)
                    res = model_dict['model'].predict([input_arr])
                    res_obj_func.append(res[0])
                    break
        else:
            br_cost = config['base_costs']
            add_volumes = []
            for add_name in config['add_cost']['names']:
                add_volumes.append(trial.params[add_name])
            br_cost += get_cost_br(np.array(add_volumes), config['add_cost']['values'])
            res_obj_func.append(br_cost)

    for idx, bound in enumerate(config['criteria']['<=']):
        if bound is not None:
            if res_obj_func[idx] > bound:
                is_deadly_penalty = True

    if is_deadly_penalty is False:
        for idx, bound in enumerate(config['criteria']['>=']):
            if bound is not None:
                if res_obj_func[idx] < bound:
                    is_deadly_penalty = True

    res_obj_func = [res_obj_func[idx] for idx, opt_val in enumerate(config['criteria']['opt']) if opt_val is not None]

    if is_deadly_penalty:
        for idx, opt_direct in enumerate(opt_directions):
            if opt_direct == 'maximize':
                res_obj_func[idx] = -1000000
            else:
                res_obj_func[idx] = 1000000

    if is_opt_weight:
        res_obj_func = [np.sum(res_obj_func * opt_weight * opt_weight_sign)]

    return res_obj_func


def training_linear_model(X, y, model_name: str, model_options: list):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RidgeRegression':
        model = Ridge(alpha=float(model_options[0]))
    elif model_name == 'ARDRegression':
        model = ARDRegression(compute_score=False)
    elif model_name == 'MLPRegression':
        model = MLPRegressor((int(float(model_options[0])), int(float(model_options[0]))), solver='lbfgs', random_state=19, max_iter=10000)
    elif model_name == 'KernelRegression':
        model = KernelRidge(alpha=float(model_options[0]), kernel='polynomial', coef0=1)
    elif model_name == 'DecisionTreeRegression':
        model = DecisionTreeRegressor(max_depth=int(float(model_options[0])), random_state=19)
    else:
        raise Exception(f'Unexpected model: {model_name}')

    model.fit(X, y)
    res_y = model.predict(X)
    metric_r2 = model.score(X, y)
    metric_r = np.sqrt(metric_r2)
    print(f'R2: {metric_r2}')
    print(f'R: {metric_r}')
    return model, metric_r, res_y


def fix_names(names):
    """ Change unacceptable symbols in columns name with _ """
    new_names = []
    for i, x in enumerate(names):
        x = x.strip().replace('/', '_').replace(',', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace(
            '"', '_').replace('>', '_').replace('<', '_').replace('|', '_')
        if x in new_names:
            if x[-1] != '_':
                x += '_' + str(i)
            else:
                x += str(i)
        new_names.append(x)
    return new_names


def get_res_df(data_names: str) -> pd.DataFrame:
    data_names_list = data_names.split('\n')
    for idx_name in range(len(data_names_list)):
        data_names_list[idx_name] = os.path.join('uploads', data_names_list[idx_name])
    df_list = []
    for path_name in data_names_list:
        df = pd.read_excel(path_name, skiprows=None, sheet_name=0)
        df_list.append(df)
    del df
    res_df = pd.concat(df_list, ignore_index=True)
    costs_df = pd.read_excel(data_names_list[-1], skiprows=None, sheet_name=1)
    units_df = pd.read_excel(data_names_list[-1], skiprows=None, sheet_name=2)
    del df_list
    res_df.columns = fix_names(res_df.columns)
    costs_df.columns = fix_names(costs_df.columns)
    units_df.columns = fix_names(units_df.columns)
    if np.any(res_df.isnull() == True):
        res_df = res_df.fillna(0)
    if np.any(costs_df.isnull() == True):
        costs_df = costs_df.fillna(0)
    if np.any(units_df.isnull() == True):
        units_df = units_df.fillna('')
    print(res_df.info())
    cost_dict = {'names': costs_df.columns.to_list(), 'values': costs_df.values[0]}
    print(cost_dict)
    del costs_df
    units_dict = dict(zip(units_df.columns.to_list(), units_df.values[0]))
    units_dict['Стоимость БР'] = 'руб./м3'
    print(units_dict)
    del units_df
    return res_df, cost_dict, units_dict


def main(page: ft.Page):
    page.title = "СмартБур"
    page.theme_mode = ft.ThemeMode.DARK
    # page.vertical_alignment = ft.MainAxisAlignment.CENTER
    # page.window_width = 3000
    page.df_to_view = None  # pd.read_excel("C:\\Users\\Aleksei\\Downloads\\Набор данных_БашНИПИ.xlsx", skiprows=2, sheet_name=1) # for desktop pres
    page.cost_dict = None
    page.units_dict = None
    page.window_height = 800
    page.bgcolor = ft.colors.GREY_400
    page.scroll = 'adaptive'
    my_app_bar_h = 40

    # --- main menu ---

    def experiment_button_click(e):
        icon_mode.name = ft.icons.LIST
        text_mode.value = 'Планирование эксперимента'
        main_buttons.visible = False
        exp_field.visible = True
        back_button.disabled = False
        page.update()

    def load_file_click(e):
        icon_mode.name = ft.icons.FILE_UPLOAD
        text_mode.value = 'Загрузка данных'
        main_buttons.visible = False
        upload_obj.visible = True
        back_button.disabled = False
        page.update()

    def model_training_click(e):
        icon_mode.name = ft.icons.MODEL_TRAINING
        text_mode.value = 'Моделирование'
        main_buttons.visible = False
        ml.visible = True
        back_button.disabled = False
        page.update()

    def optimization_main_click(e):
        icon_mode.name = ft.icons.AUTO_MODE
        text_mode.value = 'Оптимизация'
        main_buttons.visible = False
        opt_field.visible = True
        back_button.disabled = False
        page.update()

    def about_api_click(e):
        icon_mode.name = ft.icons.DESCRIPTION
        text_mode.value = 'О приложении'
        main_buttons.visible = False
        exp_field.visible = False
        upload_obj.visible = False
        ml.visible = False
        opt_field.visible = False
        about_api.visible = True
        back_button.disabled = False
        page.update()

    def back_button_click(e):
        exp_field.visible = False
        upload_obj.visible = False
        ml.visible = False
        opt_field.visible = False
        about_api.visible = False
        main_buttons.visible = True
        back_button.disabled = True
        icon_mode.name = ft.icons.MENU
        text_mode.value = 'Главное меню'
        page.update()

    def pdf_instruction_click(e):
        pdf_abs_path = os.path.abspath('pdf_instruction/Руководство пользователя v1.pdf')
        webbrowser.open_new(pdf_abs_path)

    def run_exe_calc(e):
        exe_abs_path = os.path.abspath('muTransformer.exe')
        os.system(exe_abs_path)

    main_buttons = ft.Column(
        [
            ft.Container(
            content=ft.Row(
                [
                    ft.Image(src="icons/logo_drill_main_page.png", width=page.window_height/6, height=page.window_height/6),
                    ft.Text('СмартБур', size=20, color=ft.colors.BLUE_GREY_800, weight=ft.FontWeight.BOLD),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
                height=page.window_height/4 + my_app_bar_h,
                # bgcolor=ft.colors.RED_200
            ),
            ft.Container(
            content=ft.Row(
                [
                    ft.Text('Оптимизация рецептуры бурового раствора на основании заданных критериев', size=16, color=ft.colors.BLUE_GREY_800, weight=ft.FontWeight.BOLD)
                ],
                alignment=ft.MainAxisAlignment.CENTER
            ),
                height=page.window_height / 15,
                # bgcolor=ft.colors.GREEN_200,
            ),

            ft.Row(
        [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.IconButton(ft.icons.LIST, icon_size=55, icon_color=ft.colors.BLACK87,
                                              bgcolor=ft.colors.ORANGE_700, on_click=experiment_button_click,
                                              tooltip='Параметры эксперимента;\nМетоды планирования;\n'
                                                      'Отображение плана;\nСохранение результатов.'),
                            ],
                        ),
                        ft.Row(
                            [
                                ft.Text('Планирование эксперимента', color=ft.colors.BLUE_GREY_800,
                                        weight=ft.FontWeight.BOLD),
                            ],
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.window_height/2 - my_app_bar_h*2,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.IconButton(ft.icons.FILE_UPLOAD, icon_size=55, icon_color=ft.colors.BLACK87,
                                              bgcolor=ft.colors.ORANGE_700, on_click=load_file_click,
                                              tooltip='Загрузка данных .xlsx;\nВизуализация в виде таблицы;\n'
                                                      'Расчет стоимости БР;\nМатрица корреляций.'),
                            ],
                        ),
                        ft.Row(
                            [
                                ft.Text('Загрузка данных', color=ft.colors.BLUE_GREY_800, weight=ft.FontWeight.BOLD),
                            ],
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.window_height/2 - my_app_bar_h*2,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.IconButton(ft.icons.MODEL_TRAINING, icon_size=55, icon_color=ft.colors.BLACK87,
                                              bgcolor=ft.colors.ORANGE_700, on_click=model_training_click,
                                              tooltip='Гистограммы распределения;\nОбучение моделей;\n'
                                                      'Сохранение моделей;\nПредсказание моделей;\nАнализ моделей.'),
                            ],
                        ),
                        ft.Row(
                            [
                                ft.Text('Моделирование', color=ft.colors.BLUE_GREY_800, weight=ft.FontWeight.BOLD),
                            ],
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.window_height/2 - my_app_bar_h*2,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.IconButton(ft.icons.AUTO_MODE, icon_color=ft.colors.BLACK87, icon_size=55,
                                              bgcolor=ft.colors.ORANGE_700, on_click=optimization_main_click,
                                              tooltip='Настройка критериев и ограничений;\nВзвешенные критерии;\n'
                                                      'Настройка алгоритма оптимизации;\n'
                                                      'Отображение и сохранение результатов;\nВизуализация Парето множества.'),
                            ],
                        ),
                        ft.Row(
                            [
                                ft.Text('Оптимизация', color=ft.colors.BLUE_GREY_800, weight=ft.FontWeight.BOLD),
                            ],
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.window_height/2 - my_app_bar_h*2,
                alignment=ft.alignment.center,
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
    ),
            ],
    spacing=0
    )

    icon_mode = ft.Icon(ft.icons.MENU, color=ft.colors.BLACK87, size=17, tooltip='Совет: для корректной работы функционала используйте "точку"(.)\n'
                                                                                 '       в качестве разделителя целой и дробной части чисел.')
    text_mode = ft.Text('Главное меню', color=ft.colors.BLACK, weight=ft.FontWeight.BOLD)

    back_button = ft.IconButton(ft.icons.ARROW_BACK, icon_color=ft.colors.BLACK87, icon_size=17,
                                bgcolor=ft.colors.ORANGE_700, on_click=back_button_click, disabled=True,
                                tooltip='Вернуться в главное меню')

    container_mode = ft.Container(content=ft.Row([icon_mode, text_mode]))

    clear_db_button = ft.Image(src="icons/colba.png", width=35, height=35)
    about_bar_button = ft.IconButton(ft.icons.DESCRIPTION, icon_color=ft.colors.BLACK87, icon_size=17,
                                bgcolor=ft.colors.ORANGE_700, on_click=about_api_click, disabled=False,
                                tooltip='О приложении')
    instruction_button = ft.IconButton(ft.icons.INTEGRATION_INSTRUCTIONS, icon_color=ft.colors.BLACK87, icon_size=17,
                                bgcolor=ft.colors.ORANGE_700, on_click=pdf_instruction_click, disabled=False,
                                tooltip='Руководство пользователя')
    calculator_exe_button = ft.IconButton(ft.icons.CALCULATE, icon_color=ft.colors.BLACK87, icon_size=17,
                                bgcolor=ft.colors.ORANGE_700, on_click=run_exe_calc, disabled=False,
                                tooltip='Калькулятор перевода единиц измерения')
    right_side = ft.Container(content=ft.Row([calculator_exe_button, about_bar_button, instruction_button, clear_db_button]))

    custom_appbar = ft.Container(
        content=ft.Row(
            [
                back_button,
                container_mode,
                right_side,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        height=my_app_bar_h,
        padding=5,
        gradient=ft.LinearGradient(begin=ft.alignment.top_center, end=ft.alignment.bottom_center,
                                   colors=[ft.colors.ORANGE_300, ft.colors.GREY_700])
    )
    page.overlay.append(custom_appbar)

    # --- ABOUT API ---

    about_txt = ft.Row(
        [
            ft.Text('"СмартБур" версия 1.0.1\n\nWeb-приложение для быстрой оценки технологических и экономических показателей бурового раствора с заданной рецептурой, возможностью запуска оптимизации, позволяющей в полевых условиях определять наиболее эффективное количество добавок, ХР и систем растворов, для создания промывочной жидкости, устойчивой к ухудшению реологических свойств при поступлении пластовых флюидов в процессе проявления',
                    color=ft.colors.BLACK, weight=ft.FontWeight.BOLD, width=500),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    about_api = ft.Column(
        [
          ft.Container(content=about_txt,
                       height=page.window_height - my_app_bar_h,
                       alignment=ft.alignment.center,
                       padding=ft.padding.only(top=my_app_bar_h),)
        ],
        visible=False
    )

    # --- DESIGN EXPERIMENT ---
    page.input_factors_params = None

    def generator_factors_field(num_factor: int, is_two_lvl: bool = True) -> list:
        container_factors_list = []
        if is_two_lvl:
            for j in range(num_factor):
                container_factors_list.append(ft.Container(content=ft.Column(
                    [
                        ft.TextField(label="Наименование фактора",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     height=55,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75,),
                        ft.TextField(label="Минимум",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     height=55,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75, ),
                        ft.TextField(label="Максимум",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     height=55,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75, ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                ),
                                                           alignment=ft.alignment.top_center,
                                                           padding=0,))
        else:
            for j in range(num_factor):
                container_factors_list.append(ft.Container(content=ft.Column(
                    [
                        ft.TextField(label="Наименование фактора",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     height=55,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75, ),
                        ft.TextField(label="Уровни фактора через запятую без пробелов",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     height=110,
                                     multiline=True,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75, ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                ),
                                                           alignment=ft.alignment.top_center,
                                                           padding=0,
                                                           ))

        return container_factors_list

    def get_users_num_factors_and_lvl(e):
        doe_algorithm_field.visible = False
        if choice_factor_lvl.value is not None and choice_factor_num.value is not None:
            warning_field.visible = False
            try:
                num_factor = int(float(choice_factor_num.value))
                if num_factor < 1:
                    raise ValueError
                else:
                    num_factor = min(num_factor, 10)
                    is_two_lvl = True
                    if choice_factor_lvl.value == 'любое (только ПФЭ)':
                        is_two_lvl = False
                    input_factors_field.controls = generator_factors_field(num_factor, is_two_lvl)
                    submit_input_factors_field.visible = True

            except ValueError:
                warning_txt.value = 'Введите корректное число факторов'
                warning_field.visible = True
                input_factors_field.controls = []
                submit_input_factors_field.visible = False

        else:
            warning_txt.value = 'Выберите количество уровней фактора'
            warning_field.visible = True
            input_factors_field.controls = []
            submit_input_factors_field.visible = False

        warning_txt.update()
        warning_field.update()
        input_factors_field.update()
        submit_input_factors_field.update()
        doe_algorithm_field.update()

    choice_factor_lvl = ft.Dropdown(
        label='Количество уровней фактора',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать уровни фактора',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=[ft.dropdown.Option('2 уровня (все алгоритмы)'),
                 ft.dropdown.Option('любое (только ПФЭ)'),
                 ],
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=ft.colors.GREY_400,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.75,
        width=250,
        on_change=get_users_num_factors_and_lvl,
    )

    choice_factor_num = ft.TextField(label="Количество факторов",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     width=175,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     on_submit=get_users_num_factors_and_lvl,
                                     on_change=get_users_num_factors_and_lvl,
                                     scale=0.75,)

    input_factors_field = ft.Row(
        [

        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=0,
    )

    def check_and_save_doe(e):
        doe_algorithm_field.visible = False
        warning_field.visible = False
        txt_obj_key = None
        try:
            input_factors_params = {}
            if choice_factor_lvl.value == '2 уровня (все алгоритмы)':
                alg_doe_select = [ft.dropdown.Option('Полный факторный'),
                                  ft.dropdown.Option('Дробный факторный'),
                                  ft.dropdown.Option('Центральная точка'),
                                  ft.dropdown.Option('Схема заполнения Сухарева'),
                                  ft.dropdown.Option('Схема Плакета-Бурмана'),
                                  ft.dropdown.Option('Равномерное случайное заполнение'),
                                  ft.dropdown.Option('Случайное кластерное заполнение'),
                                  ft.dropdown.Option('Латинский гиперкуб')]
                for control_obj in input_factors_field.controls:
                    for i_txt_obj, txt_obj in enumerate(control_obj.content.controls):
                        if i_txt_obj == 0:
                            if txt_obj.value == '' or txt_obj.value == ' ':
                                raise ValueError
                            txt_obj_key = txt_obj.value
                            input_factors_params[txt_obj_key] = []
                        else:
                            input_factors_params[txt_obj_key].append(float(txt_obj.value))
            else:
                alg_doe_select = [ft.dropdown.Option('Полный факторный')]
                for control_obj in input_factors_field.controls:
                    for i_txt_obj, txt_obj in enumerate(control_obj.content.controls):
                        if i_txt_obj == 0:
                            if txt_obj.value == '' or txt_obj.value == ' ':
                                raise ValueError
                            txt_obj_key = txt_obj.value
                            input_factors_params[txt_obj_key] = []
                        else:
                            str_val_list = txt_obj.value.split(',')
                            float_val_list = [float(val) for val in str_val_list]
                            input_factors_params[txt_obj_key] = float_val_list
            page.input_factors_params = input_factors_params
            doe_algorithm_dd.options = alg_doe_select
            print(page.input_factors_params)
            doe_algorithm_field.visible = True
            doe_algorithm_option_container.content.controls = []
        except (ValueError, ):  # KeyError
            warning_txt.value = 'Некорректные значения факторов'
            warning_field.visible = True

        doe_algorithm_field.update()
        warning_txt.update()
        warning_field.update()
        doe_algorithm_option_container.update()

    submit_input_factors_field = ft.Row(
        [
            ft.ElevatedButton(text='Submit',
                              icon=ft.icons.DESIGN_SERVICES,
                              icon_color=ft.colors.ORANGE_500,
                              scale=0.75,
                              on_click=check_and_save_doe,
                              tooltip='Подтвердить параметры эксперимента',),
        ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # DOE OPTIONS
    page.doe_df = None

    def generation_doe_options(e):
        doe_options_obj = []
        if doe_algorithm_dd.value == 'Дробный факторный':
            doe_options_obj.append(ft.TextField(label="Размерность",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     value=len(page.input_factors_params.keys())-1,
                                     width=150,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75,))
        elif doe_algorithm_dd.value == 'Схема заполнения Сухарева' or doe_algorithm_dd.value == 'Равномерное случайное заполнение' or doe_algorithm_dd.value == 'Случайное кластерное заполнение':
            doe_options_obj.append(ft.TextField(label="Кол-во примеров",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     value=2**len(page.input_factors_params.keys()),
                                     width=150,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75,))
        elif doe_algorithm_dd.value == 'Латинский гиперкуб':
            doe_options_obj.append(ft.TextField(label="Кол-во примеров",
                                     label_style=ft.TextStyle(color=ft.colors.BLACK, size=14),
                                     value=2**len(page.input_factors_params.keys()),
                                     width=110,
                                     text_align=ft.TextAlign.CENTER,
                                     color=ft.colors.BLUE_700,
                                     focused_border_color=ft.colors.BLUE_700,
                                     cursor_color=ft.colors.BLUE_700,
                                     scale=0.75,))
            doe_options_obj.append(ft.Dropdown(
        label='Тип распределения',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать тип распределения',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=[ft.dropdown.Option('Нормальное'),
                 ft.dropdown.Option('Пуассон'),
                 ft.dropdown.Option('Экспоненциальное'),
                 ft.dropdown.Option('Бета'),
                 ft.dropdown.Option('Гамма'),
                 ],
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=ft.colors.GREY_400,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.75,
        width=190,
    ))
        doe_algorithm_option_container.content.controls = doe_options_obj
        doe_algorithm_option_container.update()

    def get_res_doe(e):
        doe_datatable.columns = None
        doe_datatable.rows = None
        doe_datatable.visible = False
        doe_table_field.visible = False
        doe_table_field.update()
        doe_datatable.update()
        save_doe_field.visible = False
        save_doe_field.update()
        try:
            doe_options = []
            if len(doe_algorithm_option_container.content.controls) > 0:
                for obj in doe_algorithm_option_container.content.controls:
                    doe_options.append(obj.value)
            if doe_algorithm_dd.value == 'Полный факторный':
                doe_df = build.full_fact(d=page.input_factors_params)
            elif doe_algorithm_dd.value == 'Дробный факторный':
                doe_df = build.frac_fact_res(d=page.input_factors_params, res=int(float(doe_options[0])))
            elif doe_algorithm_dd.value == 'Центральная точка':
                doe_df = build.central_composite(d=page.input_factors_params, face='ccf')
            elif doe_algorithm_dd.value == 'Схема заполнения Сухарева':
                doe_df = build.sukharev(d=page.input_factors_params, num_samples=int(float(doe_options[0])))
            elif doe_algorithm_dd.value == 'Схема Плакета-Бурмана':
                doe_df = build.plackett_burman(d=page.input_factors_params)
            elif doe_algorithm_dd.value == 'Равномерное случайное заполнение':
                doe_df = build.uniform_random(d=page.input_factors_params, num_samples=int(float(doe_options[0])))
            elif doe_algorithm_dd.value == 'Случайное кластерное заполнение':
                doe_df = build.random_k_means(d=page.input_factors_params, num_samples=int(float(doe_options[0])))
            elif doe_algorithm_dd.value == 'Латинский гиперкуб':
                if doe_options[1] == 'Пуассон':
                    prob_dist = 'Poisson'
                elif doe_options[1] == 'Нормальное':
                    prob_dist = 'Normal'
                elif doe_options[1] == 'Экспоненциальное':
                    prob_dist = 'Exponential'
                elif doe_options[1] == 'Бета':
                    prob_dist = 'Beta'
                elif doe_options[1] == 'Гамма':
                    prob_dist = 'Gamma'
                else:
                    prob_dist = None
                doe_df = build.lhs(d=page.input_factors_params, num_samples=int(float(doe_options[0])), prob_distribution=prob_dist)
            else:
                raise ValueError
            if 'index' not in doe_df.columns:
                doe_df.reset_index(inplace=True)
            doe_datatable.columns = [ft.DataColumn(
                ft.Text(str(col_name)[:16], color=ft.colors.BLACK, size=10, width=75, weight=ft.FontWeight.BOLD, text_align=ft.alignment.center))
                for col_name in doe_df]
            rows = []
            for index, row in doe_df.iterrows():
                rows.append(ft.DataRow(cells=[ft.DataCell(
                    ft.Text(str(np.round(row[col_name], 4))[:9], color=ft.colors.BLACK, size=9, width=75,
                            text_align=ft.alignment.center)) for col_name in doe_df]))
            page.doe_df = doe_df
            doe_datatable.rows = rows
            doe_datatable.visible = True
            doe_table_field.visible = True
            doe_table_field.update()
            doe_datatable.update()
            save_doe_field.visible = True
            save_doe_field.update()
        except ValueError as ex:
            print(ex)

    doe_algorithm_dd = ft.Dropdown(
        label='Метод планирования эксперимента',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать метод ПЭ',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=[],
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=ft.colors.GREY_400,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.75,
        width=300,
        on_change=generation_doe_options,
        focused_color=None,
    )

    doe_algorithm_option_container = ft.Container(content=ft.Row([], spacing=0, alignment=ft.MainAxisAlignment.CENTER),
                                                  width=300,
                                                  padding=0,
                                                  alignment=ft.alignment.center,)

    doe_calc_button = ft.ElevatedButton(text='Планирование эксперимента',
                              icon=ft.icons.NEXT_PLAN,
                              icon_color=ft.colors.ORANGE_500,
                              scale=0.75,
                              on_click=get_res_doe,)

    doe_algorithm_field = ft.Row(
        [
            doe_algorithm_dd,
            doe_algorithm_option_container,
            doe_calc_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
        spacing=0,
    )

    doe_datatable = ft.DataTable(visible=False, column_spacing=2)

    doe_table_field = ft.Row([
        ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            doe_datatable,
                        ],
                        scroll=ft.ScrollMode.ALWAYS,
                        alignment=ft.MainAxisAlignment.CENTER,
                    )
                ],
                scroll=ft.ScrollMode.ALWAYS,
            ),
            height=300,
            alignment=ft.alignment.top_center,
        )
    ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # SAVE DOE

    def submit_save_doe_file(e):
        if doe_saving_text_field.value != '':
            if doe_saving_text_field.value.endswith('.xlsx'):
                doe_saving_text_field.value = doe_saving_text_field.value[:-5]
            # temp_df = pd.DataFrame(data=page.dump_opt_data)
            page.doe_df.to_excel(os.path.join('design_of_experiment',
                                          doe_saving_text_field.value + doe_saving_text_field.suffix_text),
                             encoding='utf-8', index=False)
            dialog_save_doe.open = False
            dialog_save_doe.update()

    def open_dialog_save_doe(e):
        page.dialog = dialog_save_doe
        dialog_save_doe.open = True
        page.update()

    doe_saving_text_field = ft.TextField(label='Имя_файла.xlsx', width=300, height=70,
                                         text_align=ft.alignment.center,
                                         multiline=False,
                                         suffix_text='.xlsx'
                                         )

    dialog_save_doe = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=doe_saving_text_field,
        actions=[
            ft.TextButton('Submit', on_click=submit_save_doe_file)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    save_doe_button = ft.ElevatedButton(text='Сохранить план эксперимента',
                              icon=ft.icons.SAVE_AS,
                              icon_color=ft.colors.ORANGE_500,
                              scale=0.75,
                              on_click=open_dialog_save_doe,)

    save_doe_field = ft.Row(
        [
            save_doe_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
    )

    warning_txt = ft.Text('', color=ft.colors.RED, weight=ft.FontWeight.BOLD, scale=0.85)

    warning_field = ft.Row([
        warning_txt,
    ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
    )

    exp_field = ft.Column(
        [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                choice_factor_lvl,
                                choice_factor_num,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        input_factors_field,
                        submit_input_factors_field,
                        doe_algorithm_field,
                        doe_table_field,
                        save_doe_field,
                        warning_field,
                    ]
                ),
                height=page.window_height - my_app_bar_h,
                alignment=ft.alignment.top_center,
                padding=ft.padding.only(top=my_app_bar_h),
            ),
        ],
        visible=False,
    )

    # --- LOADING FILES --- #

    page.counter_new_line_table = 0

    def add_new_row_data(e):
        new_row_dict = {}
        for text_container in header_field.controls:
            try:
                new_row_dict[text_container.content.label] = float(text_container.content.value)
            except ValueError:
                new_row_dict[text_container.content.label] = text_container.content.value
        is_empty_row = True
        for v in new_row_dict.values():
            if v != '':
                is_empty_row = False
                break
        if is_empty_row is False:
            for k in new_row_dict:
                if new_row_dict[k] == '' or new_row_dict[k] == ' ':
                    new_row_dict[k] = 0

            page.df_to_view.loc[len(page.df_to_view.index)] = list(new_row_dict.values())

            new_data_row = ft.DataRow(cells=[ft.DataCell(
                ft.Text(str(new_row_dict[k])[:25], color=ft.colors.BLACK, size=9, width=50,
                        text_align=ft.alignment.center)) for k in new_row_dict.keys()])

            if datatable.columns[-1].label.value == 'Стоимость БР':
                new_cost = get_cost_br(
                    volume_arr=page.df_to_view.loc[len(page.df_to_view.index) - 1, page.cost_dict['names']],
                    cost_arr=page.cost_dict['values'])
                new_data_row.cells.append(ft.DataCell(
                    ft.Text(str(new_cost), color=ft.colors.BLACK, size=9, width=50,
                            text_align=ft.alignment.center)))
            datatable.rows.append(new_data_row)

            datatable.update()
            new_line_field.visible = False
            page.counter_new_line_table += 1
            new_line_field.update()

            # page.df_to_view.loc[len(page.df_to_view.index)] = list(new_row_dict.values())

    def add_new_line_table(e):
        if page.counter_new_line_table % 2 == 0:
            new_line_field.visible = True
        else:
            new_line_field.visible = False

        page.counter_new_line_table += 1
        new_line_field.update()

    def pick_files_result(e):
        selected_files.value = None
        selected_files.update()
        upload_list = []
        if pick_files_dialog.result is not None and pick_files_dialog.result.files is not None:
            for f in pick_files_dialog.result.files:
                upload_list.append(
                    ft.FilePickerUploadFile(
                        f.name,
                        upload_url=page.get_upload_url(f.name, 600),
                    )
                )
            pr.visible = True
            pr.update()
            pick_files_dialog.upload(upload_list)
            time.sleep(2)

        print(upload_list)
        if len(upload_list) > 0:
            for i in range(len(upload_list)):
                if i == 0:
                    selected_files.value = upload_list[i].name
                else:
                    selected_files.value += f'\n{upload_list[i].name}'
        else:
            selected_files.value = "Файл не выбран!"
            datatable.visible = False
            datatable.columns = None
            datatable.rows = None
            lv.visible = False
            lv.update()
            datatable.update()
            add_new_line_row.visible = False
            add_new_line_row.update()
            new_line_field.visible = False
            new_line_field.update()
            page.counter_new_line_table = 0
            save_and_download_field.visible = False
            save_and_download_field.update()
            calculation_br_button.disabled = True
            calculation_br_button.icon = ft.icons.CALCULATE
            calculation_br_button.icon_color = ft.colors.ORANGE_500
            calculation_br_button.update()
            corr_matrix_button.disabled = True
            corr_matrix_button.update()

        if selected_files.value != "Файл не выбран!":
            page.counter_feat_selector = 0
            page.counter_target_selector = 0
            page.df_to_view, page.cost_dict, page.units_dict = get_res_df(selected_files.value)
            datatable.columns = [ft.DataColumn(
                ft.Text(str(col_name)[:16], color=ft.colors.BLACK, size=9, width=50, text_align=ft.alignment.center))
                for col_name in page.df_to_view]
            rows = []
            for index, row in page.df_to_view.iterrows():
                rows.append(ft.DataRow(cells=[ft.DataCell(
                    ft.Text(str(row[col_name])[:25], color=ft.colors.BLACK, size=9, width=50,
                            text_align=ft.alignment.center)) for col_name in page.df_to_view]))
            datatable.rows = rows
            datatable.visible = True
            lv.visible = True
            lv.update()
            datatable.update()
            header_field.controls = header_text_field_generator(page.df_to_view.columns)
            header_field.update()
            add_new_line_row.visible = True
            add_new_line_row.update()
            save_and_download_field.visible = True
            save_and_download_field.update()
            calculation_br_button.disabled = False
            calculation_br_button.icon = ft.icons.CALCULATE
            calculation_br_button.icon_color = ft.colors.ORANGE_500
            calculation_br_button.update()
            corr_matrix_button.disabled = False
            corr_matrix_button.update()

        add_feats_hist_dict_dd(e)
        pr.visible = False
        pr.update()
        selected_files.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(pick_files_dialog)

    selected_files = ft.Text(color=ft.colors.BLACK, text_align=ft.MainAxisAlignment.CENTER, scale=0.75)

    pr = ft.Row(
        [
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=ft.colors.ORANGE_900),
            ft.Text(color=ft.colors.BLACK, text_align=ft.MainAxisAlignment.CENTER, value="Загрузка файлов...",
                    scale=0.75)
        ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    datatable = ft.DataTable(visible=False, column_spacing=0)

    lv = ft.Row([
        ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            datatable,
                        ],
                        scroll=ft.ScrollMode.ALWAYS,
                    )
                ],
                scroll=ft.ScrollMode.ALWAYS,
            ),
            height=page.window_height / 2,
            alignment=ft.alignment.center
        )
    ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # lv = ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=False, visible=False, height=page.window_height / 2)
    # lv.controls.append(datatable)

    add_new_line_button = ft.IconButton(ft.icons.ADD, icon_color=ft.colors.ORANGE_500, scale=0.75,  # icon_size=10,
                                        bgcolor=ft.colors.BLUE_GREY_900, on_click=add_new_line_table,
                                        tooltip='Добавить строку',
                                        disabled=False)

    add_new_line_row = ft.Row(
        [
            add_new_line_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False
    )

    def header_text_field_generator(header) -> list:
        items = []
        for header_name in header:
            items.append(
                ft.Container(
                    content=ft.TextField(label=header_name[:16], width=60, height=50, text_align=ft.alignment.center,
                                         label_style=ft.TextStyle(color=ft.colors.BLACK, size=9),
                                         text_style=ft.TextStyle(color=ft.colors.BLUE_900, size=9),
                                         focused_border_color=ft.colors.BLUE_900,
                                         cursor_color=ft.colors.BLACK,
                                         multiline=True,
                                         max_lines=5,
                                         ),
                    height=60,
                    alignment=ft.alignment.center,
                )
            )
        return items

    header_field = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=0,
    )

    new_line_field = ft.Column([
        header_field,
        ft.Row(
            [
                ft.ElevatedButton(text='Submit',
                                  icon=ft.icons.TABLE_ROWS,
                                  icon_color=ft.colors.ORANGE_500,
                                  scale=0.75,
                                  on_click=add_new_row_data,
                                  tooltip='Подтвердить'),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    ],
        visible=False,
    )

    # BUTTON SELECT FILES

    def submit_selected_files(e):
        file_name_list = []
        for checkbox in dialog_files_db.content.controls:
            if checkbox.value is True:
                file_name_list.append(checkbox.label)
        if len(file_name_list) == 0:
            selected_files.value = "Файл не выбран!"
            datatable.visible = False
            datatable.columns = None
            datatable.rows = None
            lv.visible = False
            lv.update()
            datatable.update()
            add_new_line_row.visible = False
            add_new_line_row.update()
            new_line_field.visible = False
            new_line_field.update()
            page.counter_new_line_table = 0
            save_and_download_field.visible = False
            save_and_download_field.update()
            calculation_br_button.disabled = True
            calculation_br_button.icon = ft.icons.CALCULATE
            calculation_br_button.icon_color = ft.colors.ORANGE_500
            calculation_br_button.update()
            corr_matrix_button.disabled = True
            corr_matrix_button.update()
        else:
            page.counter_feat_selector = 0
            page.counter_target_selector = 0
            for i in range(len(file_name_list)):
                if i == 0:
                    selected_files.value = file_name_list[i]
                else:
                    selected_files.value += f'\n{file_name_list[i]}'

            page.df_to_view, page.cost_dict, page.units_dict = get_res_df(selected_files.value)
            datatable.columns = [ft.DataColumn(
                ft.Text(str(col_name)[:16], color=ft.colors.BLACK, size=9, width=50, text_align=ft.alignment.center))
                for col_name in page.df_to_view]
            rows = []
            for index, row in page.df_to_view.iterrows():
                rows.append(ft.DataRow(cells=[ft.DataCell(
                    ft.Text(str(row[col_name])[:25], color=ft.colors.BLACK, size=9, width=50,
                            text_align=ft.alignment.center)) for col_name in page.df_to_view]))
            datatable.rows = rows
            datatable.visible = True
            lv.visible = True
            lv.update()
            datatable.update()
            header_field.controls = header_text_field_generator(page.df_to_view.columns)
            header_field.update()
            add_new_line_row.visible = True
            add_new_line_row.update()
            save_and_download_field.visible = True
            save_and_download_field.update()
            calculation_br_button.disabled = False
            calculation_br_button.icon = ft.icons.CALCULATE
            calculation_br_button.icon_color = ft.colors.ORANGE_500
            calculation_br_button.update()
            corr_matrix_button.disabled = False
            corr_matrix_button.update()

        add_feats_hist_dict_dd(e)
        dialog_files_db.open = False
        dialog_files_db.update()
        selected_files.update()

    def open_dialog_files(e):
        dialog_files_db.content.controls = checkbox_generator('uploads')
        page.dialog = dialog_files_db
        dialog_files_db.open = True
        page.update()

    def checkbox_generator(path_view: str) -> list:
        checkbox_list = []
        for file_name in os.listdir(path_view):
            checkbox_list.append(ft.Checkbox(label=file_name, value=False))
        return checkbox_list

    dialog_files_db = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_selected_files)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    # BUTTON DELETE FILE
    def submit_delete_file(e):
        file_name_list = []
        for checkbox in dialog_delete_files_ad.content.controls:
            if checkbox.value is True:
                file_name_list.append(os.path.join('uploads', checkbox.label))
        if len(file_name_list) > 0:
            for path in file_name_list:
                os.remove(path)

        dialog_delete_files_ad.open = False
        dialog_delete_files_ad.update()

    def open_dialog_delete_files(e):
        dialog_delete_files_ad.content.controls = checkbox_generator('uploads')
        page.dialog = dialog_delete_files_ad
        dialog_delete_files_ad.open = True
        page.update()

    dialog_delete_files_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_delete_file)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    # BUTTON CALCULATION BR

    def calculation_br_button_func(e):
        if datatable.columns[-1].label.value != 'Стоимость БР':
            datatable.columns.append(ft.DataColumn(
                ft.Text('Стоимость БР', color=ft.colors.BLACK, size=9, width=50, text_align=ft.alignment.center)))

            for idx_row in page.df_to_view.index:
                cost_br = get_cost_br(volume_arr=page.df_to_view.loc[idx_row, page.cost_dict['names']],
                                      cost_arr=page.cost_dict['values'])

                datatable.rows[idx_row].cells.append(ft.DataCell(
                    ft.Text(str(cost_br), color=ft.colors.BLACK, size=9, width=50,
                            text_align=ft.alignment.center)))

            datatable.update()
            calculation_br_button.disabled = True
            calculation_br_button.icon = ft.icons.CHECK
            calculation_br_button.icon_color = ft.colors.GREEN_500
            calculation_br_button.update()

    calculation_br_button = ft.ElevatedButton('Рассчитать стоимость БР',
                                              icon=ft.icons.CALCULATE,
                                              icon_color=ft.colors.ORANGE_500,
                                              scale=0.75,
                                              on_click=calculation_br_button_func,
                                              style=ft.ButtonStyle(bgcolor={
                                                  ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                              }),
                                              disabled=True, )

    calculation_br_row = ft.Row(
        [
            calculation_br_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # BUTTON SAVE FILE

    def submit_save_file(e):
        if file_saving_field.value != '':
            if file_saving_field.value.endswith('.xlsx'):
                file_saving_field.value = file_saving_field.value[:-5]
            page.df_to_view.to_excel(os.path.join('uploads', file_saving_field.value + file_saving_field.suffix_text),
                                     index=False)
            dialog_save_file.open = False
            dialog_save_file.update()

    def open_dialog_save_files(e):
        page.dialog = dialog_save_file
        dialog_save_file.open = True
        page.update()

    file_saving_field = ft.TextField(label='Имя_файла.xlsx', width=300, height=70, text_align=ft.alignment.center,
                                     multiline=False,
                                     suffix_text='.xlsx'
                                     )

    dialog_save_file = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=file_saving_field,
        actions=[
            ft.TextButton('Submit', on_click=submit_save_file)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    # BUTTON DOWNLOAD FILE

    def submit_downloads_files(e):
        file_path_list = []
        for checkbox in dialog_files_db.content.controls:
            if checkbox.value is True:
                file_path_list.append(os.path.join('uploads', checkbox.label))
        if len(file_path_list) != 0:
            tmp_path = f'downloads/download_{time.time_ns()}.zip'
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_path_list:
                    zipf.write(file_path)
            page.splash = ft.ProgressBar()
            page.update()
            time.sleep(5)
            page.splash = None
            page.launch_url(tmp_path, web_window_name='_self')
            dialog_files_downloads.open = False
            page.update()

    def open_dialog_downloads_files(e):
        dialog_files_downloads.content.controls = checkbox_generator('uploads')
        page.dialog = dialog_files_downloads
        dialog_files_downloads.open = True
        page.update()

    dialog_files_downloads = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_downloads_files)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    save_and_download_field = ft.Column(
        [
            ft.Row(
                [  # BUTTON SAVE FILE
                    ft.ElevatedButton(
                        'Сохранить файл',
                        icon=ft.icons.SAVE_AS,
                        icon_color=ft.colors.ORANGE_500,
                        scale=0.75,
                        on_click=open_dialog_save_files,
                    ),  # BUTTON DOWNLOAD FILE
                    ft.ElevatedButton(
                        'Скачать файл',
                        icon=ft.icons.FILE_DOWNLOAD,
                        icon_color=ft.colors.ORANGE_500,
                        scale=0.75,
                        on_click=open_dialog_downloads_files,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        ],
        visible=False
    )

    # CORR MATRIX
    def corr_matrix_plt(e):
        df_for_corr = page.df_to_view.select_dtypes(include='number')
        drop_uniq_col_list = []
        for col in df_for_corr.columns:
            if df_for_corr[col].nunique() == 1:
                drop_uniq_col_list.append(col)
        df_for_corr.drop(drop_uniq_col_list, axis=1, inplace=True)
        new_header_names = []
        for col_name in df_for_corr.columns:
            new_header_names.append(col_name[:19])
        map_cols = dict(zip(df_for_corr.columns.to_list(), new_header_names))
        df_for_corr.rename(columns=map_cols, inplace=True)
        corr = np.round(df_for_corr.corr(), 2)
        del df_for_corr

        fig, ax = plt.subplots(figsize=(14, 13))
        plt.subplots_adjust(left=0.15, right=1, top=1, bottom=0.1)
        sns.heatmap(abs(corr),
                    annot=True,
                    cmap='copper',
                    cbar=True,
                    square=True,
                    ax=ax,
                    cbar_kws={"shrink": 0.75})
        ax.set_title(f'Матрица корреляций')
        corr_mx_chart = MatplotlibChart(fig, transparent=False, original_size=False)
        plt.close()

        return [corr_mx_chart]

    def open_ad_corr_matrix(e):
        ad_corr_matrix.content.controls = corr_matrix_plt(e)
        page.dialog = ad_corr_matrix
        ad_corr_matrix.open = True
        page.update()

    ad_corr_matrix = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
        actions=None,
        on_dismiss=None
    )

    corr_matrix_button = ft.ElevatedButton('Матрица корреляций',
                                           icon=ft.icons.ANALYTICS,
                                           icon_color=ft.colors.ORANGE_500,
                                           scale=0.75,
                                           on_click=open_ad_corr_matrix,
                                           style=ft.ButtonStyle(bgcolor={
                                               ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                           },),
                                           disabled=True,)

    upload_obj = ft.Column(
        [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.ElevatedButton(
                                    'Загрузить файл',
                                    icon=ft.icons.UPLOAD_FILE,
                                    icon_color=ft.colors.ORANGE_500,
                                    scale=0.75,
                                    on_click=lambda _: pick_files_dialog.pick_files(
                                        allow_multiple=True,
                                        allowed_extensions=['xlsx'],
                                    ),
                                ),
                                # BUTTON SELECT FILES
                                ft.ElevatedButton(
                                    'Выбрать файл',
                                    icon=ft.icons.FILE_PRESENT,
                                    icon_color=ft.colors.ORANGE_500,
                                    scale=0.75,
                                    on_click=open_dialog_files,
                                ),
                                # BUTTON DELETE FILES
                                ft.ElevatedButton(
                                    'Удалить файл',
                                    icon=ft.icons.AUTO_DELETE,
                                    icon_color=ft.colors.ORANGE_500,
                                    scale=0.75,
                                    on_click=open_dialog_delete_files,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),

                        selected_files,
                        pr,
                        calculation_br_row,
                        lv,
                        add_new_line_row,
                        new_line_field,
                        ft.Row(
                            [
                                corr_matrix_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        save_and_download_field,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.window_height - my_app_bar_h,
                alignment=ft.alignment.center,
                padding=ft.padding.only(top=my_app_bar_h),
            ),
        ],
        visible=False,
    )

    # --- MACHINE LEARNING --- #

    # HISTOGRAM DISTRIBUTION
    def hist_chart_generator(e):
        min_val = np.round(page.df_to_view[hist_dict_dd.value].min(), 4)
        max_val = np.round(page.df_to_view[hist_dict_dd.value].max(), 4)
        mean_val = np.round(page.df_to_view[hist_dict_dd.value].mean(), 4)
        std_val = np.round(page.df_to_view[hist_dict_dd.value].std(), 4)
        median_val = np.round(page.df_to_view[hist_dict_dd.value].median(), 4)
        plus_2_s = np.round(mean_val + 2 * std_val, 4)
        minus_2_s = np.round(mean_val - 2 * std_val, 4)
        count_up_2_s = len(page.df_to_view[hist_dict_dd.value][page.df_to_view[hist_dict_dd.value] > plus_2_s])
        count_low_2_s = len(page.df_to_view[hist_dict_dd.value][page.df_to_view[hist_dict_dd.value] < minus_2_s])
        plus_3_s = np.round(mean_val + 3 * std_val, 4)
        minus_3_s = np.round(mean_val - 3 * std_val, 4)
        count_up_3_s = len(page.df_to_view[hist_dict_dd.value][page.df_to_view[hist_dict_dd.value] > plus_3_s])
        count_low_3_s = len(page.df_to_view[hist_dict_dd.value][page.df_to_view[hist_dict_dd.value] < minus_3_s])
        stat_str = f'Минимум: {min_val}\nМаксимум: {max_val}\nСреднее: {mean_val}\nСтанд.откл. (\u03c3): {std_val}' \
                   f'\nМедиана: {median_val}\nСр. +2 \u03c3 : {plus_2_s}\nСр. -2 \u03c3 : {minus_2_s}\nКол-во значений' \
                   f' > +2 \u03c3 : {count_up_2_s}\nКол-во значений < -2 \u03c3 : {count_low_2_s}\nСр. +3 \u03c3 :' \
                   f' {plus_3_s}\nСр. -3 \u03c3 : {minus_3_s}\nКол-во значений > +3 \u03c3 : {count_up_3_s}\nКол-во' \
                   f' значений < -3 \u03c3 : {count_low_3_s}'
        fig, ax = plt.subplots(figsize=(9, 5))

        ax = sns.histplot(page.df_to_view[hist_dict_dd.value], bins=30, kde=True, color='mediumslateblue', alpha=0.7,
                          line_kws={'lw': 3})
        if len(ax.lines) > 0:
            ax.lines[0].set_color('darkorange')
        ax.grid(visible=True)
        ax.set_ylabel('Количество')
        plt.tight_layout()
        hist_chart_container.content = MatplotlibChart(fig, transparent=True, original_size=False)
        plt.close()
        hist_chart_container.border = None
        hist_chart_container.update()
        stat_text_container.content.value = stat_str
        stat_text_container.border = None
        stat_text_container.update()

    def add_feats_hist_dict_dd(e):
        options_cols_hist_list = []
        if page.df_to_view is not None:
            for col_name in page.df_to_view.select_dtypes(include='number').columns:
                options_cols_hist_list.append(ft.dropdown.Option(col_name))
        hist_dict_dd.options = options_cols_hist_list
        hist_dict_dd.update()

    hist_dict_dd = ft.Dropdown(
        label='Гистограмма распределения и статистика',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать данные',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=None,
        autofocus=True,
        border_color=ft.colors.ORANGE_100,
        focused_border_color=ft.colors.ORANGE_500,
        bgcolor=ft.colors.GREY_500,
        scale=0.75,
        width=350,
        on_change=hist_chart_generator,
    )

    hist_chart_container = ft.Container(content=None,
                                        expand=True,
                                        height=350,
                                        alignment=ft.alignment.center_right,
                                        padding=0,
                                        border=ft.Border(top=ft.BorderSide(width=2), bottom=ft.BorderSide(width=2), left=ft.BorderSide(width=2)),
                                        )

    stat_text_container = ft.Container(content=ft.Text('', color=ft.colors.DEEP_PURPLE_400, weight=ft.FontWeight.BOLD, scale=0.85),
                                       height=350,
                                       width=200,
                                       alignment=ft.alignment.center_left,
                                       padding=0,
                                       border=ft.Border(ft.BorderSide(width=2), ft.BorderSide(width=2), ft.BorderSide(width=2)),
                                       )

    hist_dist_field = ft.Column(
        [
          ft.Row(
              [
                  hist_dict_dd,
              ],
              alignment=ft.MainAxisAlignment.CENTER,
          ),
          ft.Row(
              [
                  hist_chart_container,
                  stat_text_container,
              ],
              # alignment=ft.MainAxisAlignment.CENTER,
              spacing=0,
          ),
        ],
        spacing=0,
    )

    # MODEL SELECTOR

    def change_model_options(e):
        while len(model_options_row.controls) > 0:
            model_options_row.controls.pop()
        if model_type_dd.value == 'RidgeRegression':
            model_options_row.controls.append(ft.TextField(label='L1 регуляризатор',
                                                           label_style=ft.TextStyle(
                                                           color=ft.colors.BLACK),
                                                           color=ft.colors.BLUE_700,
                                                           text_size=15,
                                                           focused_border_color=ft.colors.BLUE_700,
                                                           cursor_color=ft.colors.BLUE_700,
                                                           value='1.0',
                                                           scale=0.75,
                                                           width=150, ))
            model_options_field.visible = True
        elif model_type_dd.value == 'MLPRegression':
            model_options_row.controls.append(ft.TextField(label='Количество нейронов',
                                                           label_style=ft.TextStyle(
                                                               color=ft.colors.BLACK),
                                                           color=ft.colors.BLUE_700,
                                                           text_size=15,
                                                           focused_border_color=ft.colors.BLUE_700,
                                                           cursor_color=ft.colors.BLUE_700,
                                                           value='25',
                                                           scale=0.75,
                                                           width=150, ))
            model_options_field.visible = True
        elif model_type_dd.value == 'KernelRegression':
            model_options_row.controls.append(ft.TextField(label='L1 регуляризатор',
                                                           label_style=ft.TextStyle(
                                                               color=ft.colors.BLACK),
                                                           color=ft.colors.BLUE_700,
                                                           text_size=15,
                                                           focused_border_color=ft.colors.BLUE_700,
                                                           cursor_color=ft.colors.BLUE_700,
                                                           value='0.01',
                                                           scale=0.75,
                                                           width=150, ))
            model_options_field.visible = True
        elif model_type_dd.value == 'DecisionTreeRegression':
            model_options_row.controls.append(ft.TextField(label='Глубина дерева',
                                                           label_style=ft.TextStyle(
                                                               color=ft.colors.BLACK),
                                                           color=ft.colors.BLUE_700,
                                                           text_size=15,
                                                           focused_border_color=ft.colors.BLUE_700,
                                                           cursor_color=ft.colors.BLUE_700,
                                                           value='3',
                                                           scale=0.75,
                                                           width=150, ))
            model_options_field.visible = True
        else:
            model_options_field.visible = False

        model_options_field.update()

    model_type_dd = ft.Dropdown(
        label='Тип модели',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать тип модели',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=[ft.dropdown.Option('LinearRegression'),
                 ft.dropdown.Option('RidgeRegression'),
                 ft.dropdown.Option('ARDRegression'),
                 ft.dropdown.Option('MLPRegression'),
                 ft.dropdown.Option('KernelRegression'),
                 ft.dropdown.Option('DecisionTreeRegression')],
        autofocus=True,
        border_color=ft.colors.ORANGE_100,
        focused_border_color=ft.colors.ORANGE_500,
        bgcolor=ft.colors.GREY_500,
        scale=0.75,
        on_change=change_model_options
    )

    # MODEL OPTIONS

    model_options_row = ft.Row(
        [],
        alignment=ft.MainAxisAlignment.CENTER
    )
    model_options_field = ft.Container(content=model_options_row,
                                       alignment=ft.alignment.center,
                                       visible=False)

    # FEATURE SELECTOR
    page.counter_feat_selector = 0

    def feature_generator(all_values: bool) -> list:
        features_list = []
        if page.df_to_view is None:
            return None
        else:
            for col_name in page.df_to_view.select_dtypes(include='number').columns:
                print(page.df_to_view[col_name].dropna().std())
                if page.df_to_view[col_name].dropna().std() < 0.0001:
                    features_list.append(ft.Checkbox(label=col_name, value=False, check_color=ft.colors.RED_700,
                                                     active_color=ft.colors.RED_200,
                                                     border_side=ft.BorderSide(width=None, color=ft.colors.RED),
                                                     label_style=ft.TextStyle(color=ft.colors.RED),
                                                     tooltip='Нулевое стадратное отклонение'))
                else:
                    features_list.append(ft.Checkbox(label=col_name, value=all_values))
            return features_list

    def open_feature_selection_ad(e):
        if page.counter_feat_selector == 0:
            feature_selection_ad.content.controls = feature_generator(all_values=True)
            page.counter_feat_selector += 1
        page.dialog = feature_selection_ad
        feature_selection_ad.open = True
        page.update()

    feature_selection_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
    )

    feature_select_button = ft.ElevatedButton('Выбрать входы',
                                              icon=ft.icons.INPUT,
                                              scale=0.85,
                                              on_click=open_feature_selection_ad,
                                              icon_color=ft.colors.ORANGE_500,
                                              )

    # TARGET SELECTOR
    page.counter_target_selector = 0

    def open_target_selection_ad(e):
        if page.counter_target_selector == 0:
            target_selection_ad.content.controls = feature_generator(all_values=False)
            page.counter_target_selector += 1
        page.dialog = target_selection_ad
        target_selection_ad.open = True
        page.update()

    target_selection_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
    )

    target_select_button = ft.ElevatedButton('Выбрать выходы',
                                             icon=ft.icons.OUTPUT,
                                             scale=0.85,
                                             on_click=open_target_selection_ad,
                                             icon_color=ft.colors.ORANGE_500,
                                             )

    # MODEL TRAINING
    page.plt_charts = None
    page.model_list = None
    page.model_metadata = None
    page.pred_data = None  # for save modeling result in file

    def run_training(e):
        try:
            save_pred_data_button.icon = ft.icons.SAVE_ALT_SHARP
            save_pred_data_button.icon_color = ft.colors.ORANGE_500
            save_pred_data_button.tooltip = None
            save_pred_data_button.disabled = True
            save_pred_data_button.update()
            save_model_button.icon = ft.icons.DATA_SAVER_ON
            save_model_button.icon_color = ft.colors.ORANGE_500
            save_model_button.tooltip = None
            save_model_button.disabled = True
            save_model_button.update()
            export_to_pmml_button.icon = ft.icons.IMPORT_EXPORT
            export_to_pmml_button.icon_color = ft.colors.ORANGE_500
            export_to_pmml_button.disabled = True
            export_to_pmml_button.update()
            show_charts_button.disabled = True
            show_charts_button.update()
            features_name_list = []
            target_name_list = []
            for checkbox_obj in feature_selection_ad.content.controls:
                if checkbox_obj.value is True:
                    features_name_list.append(checkbox_obj.label)
            for checkbox_obj in target_selection_ad.content.controls:
                if checkbox_obj.value is True:
                    target_name_list.append(checkbox_obj.label)

            if (model_type_dd.value is not None) and (len(features_name_list) > 0) and (len(target_name_list) > 0):
                model_options_list = []
                for element in model_options_row.controls:
                    model_options_list.append(element.value)
                model_training_button.disabled = True
                model_training_button.update()
                pr_training.visible = True
                pr_training.update()

                models_list = []
                gc.collect()
                model_metadata = {'modelType': model_type_dd.value, 'targets': target_name_list}
                for feat in features_name_list:
                    model_metadata[feat] = {'min': float(page.df_to_view[feat].min()),
                                            'max': float(page.df_to_view[feat].max())}
                # print(model_metadata)
                page.pred_data = page.df_to_view[features_name_list].copy()
                for target_name in target_name_list:
                    model_data = {'target': target_name}
                    model_data['model'], model_data['metric'], model_data['predict'] = \
                        training_linear_model(page.df_to_view[features_name_list], page.df_to_view[target_name],
                                              model_type_dd.value, model_options_list)
                    models_list.append(model_data)
                    page.pred_data[target_name] = model_data['predict']

                # print(page.pred_data)
                # print(models_list)
                page.model_list = models_list
                page.model_metadata = model_metadata
                # Generation and visible fields on optimization window
                input_param_table_generator(e)
                criteria_bound_table_generator(e)
                algorithm_opt_selector.visible = True
                algorithm_opt_selector.update()

                plt_charts = []
                page.plt_charts = None
                plt.close('all')
                gc.collect()
                for idx, model in enumerate(models_list):
                    # print(f"{target_name_list[idx]}:\n{model['predict']}")
                    fig, ax = plt.subplots()
                    ax.scatter(range(len(model['predict'])), page.df_to_view[target_name_list[idx]].values, color='orange',
                               label='Исходные данные')
                    ax.plot(range(len(model['predict'])), model['predict'], color='blue', label='Предсказанные значения')
                    ax.set_ylabel(target_name_list[idx])
                    ax.set_title(f'R: {model["metric"]}')
                    ax.legend()
                    plt_charts.append(MatplotlibChart(fig, transparent=False, original_size=False))
                page.plt_charts = plt_charts
                save_pred_data_button.disabled = False
                save_pred_data_button.update()
                save_model_button.disabled = False
                save_model_button.update()
                if model_type_dd.value != 'KernelRegression':
                    export_to_pmml_button.disabled = False
                export_to_pmml_button.update()
                show_charts_button.disabled = False
                show_charts_button.update()
                if len(page.model_list) > 0:
                    txt_select_predict_model.value = f'Модель загружена'
                    txt_select_predict_model.color = ft.colors.GREEN_700
                    txt_select_predict_model.update()
                    inputs_predict_content_generator(e)
                    outputs_predict_content_generator(e)
                    inference_field.visible = True
                    inference_field.update()
                    analyze_targets_dd_generator(e)
                    model_analyze_field.visible = True
                    model_analyze_field.update()
                    txt_select_model.value = 'Модель загружена'
                    txt_select_model.color = ft.colors.GREEN_700
                    txt_select_model.update()
        except:
            pass

        pr_training.visible = False
        pr_training.update()
        model_training_button.disabled = False
        model_training_button.update()

    model_training_button = ft.ElevatedButton('Построить модель',
                                              icon=ft.icons.MODEL_TRAINING,
                                              scale=0.85,
                                              on_click=run_training,
                                              icon_color=ft.colors.ORANGE_500,
                                              style=ft.ButtonStyle(bgcolor={
                                                  ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                              }),
                                              )

    pr_training = ft.Row(
        [
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=ft.colors.ORANGE_900),
            ft.Text(color=ft.colors.BLACK, text_align=ft.MainAxisAlignment.CENTER, value="Обучение моделей...",
                    scale=0.75)
        ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    def open_charts_ml_predict_ad(e):
        charts_ml_predict_ad.content.controls = page.plt_charts
        page.dialog = charts_ml_predict_ad
        charts_ml_predict_ad.open = True
        page.update()

    charts_ml_predict_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
    )

    show_charts_button = ft.ElevatedButton('Посмотреть результаты',
                                           icon=ft.icons.VISIBILITY,
                                           scale=0.85,
                                           on_click=open_charts_ml_predict_ad,
                                           icon_color=ft.colors.ORANGE_500,
                                           style=ft.ButtonStyle(bgcolor={
                                               ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                           }),
                                           disabled=True
                                           )

    # SAVE PRED DATA

    def get_generated_pred_data_name(e):
        current_date = str(datetime.date.today().isoformat())
        current_time = str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')
        generated_model_dir_name = page.model_metadata['modelType'] + '_' + current_date + '_' + current_time
        return generated_model_dir_name

    def save_pred_data_processing(e):
        if not os.path.exists(f'models_result'):
            os.mkdir(f'models_result')

        if save_pred_data_text_field.value.endswith('.xlsx'):
            save_pred_data_text_field.value = save_pred_data_text_field.value[:-5]
        page.pred_data.to_excel(os.path.join('models_result', save_pred_data_text_field.value + save_pred_data_text_field.suffix_text),
                                 index=False)

        dialog_save_pred_data.open = False
        dialog_save_pred_data.update()
        save_pred_data_button.icon = ft.icons.CHECK
        save_pred_data_button.icon_color = ft.colors.GREEN_500
        save_pred_data_button.tooltip = save_pred_data_text_field.value
        save_pred_data_button.disabled = True
        save_pred_data_button.update()

    def open_dialog_save_pred_data(e):
        save_pred_data_text_field.value = get_generated_pred_data_name(e)
        page.dialog = dialog_save_pred_data
        dialog_save_pred_data.open = True
        page.update()

    save_pred_data_text_field = ft.TextField(label='Имя файла', width=1000, height=70,
                                             text_align=ft.alignment.center,
                                             multiline=False,
                                             suffix_text='.xlsx',
                                             )

    dialog_save_pred_data = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=save_pred_data_text_field,
        actions=[
            ft.TextButton('Submit', on_click=save_pred_data_processing)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    save_pred_data_button = ft.ElevatedButton('Сохранить результаты',
                                           icon=ft.icons.SAVE_ALT_SHARP,
                                           scale=0.85,
                                           on_click=open_dialog_save_pred_data,
                                           icon_color=ft.colors.ORANGE_500,
                                           style=ft.ButtonStyle(bgcolor={
                                               ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                           }),
                                           disabled=True
                                           )

    # MODEL SAVING
    page.model_dir_name = None

    def get_generated_model_dir_name(e):
        current_date = str(datetime.date.today().isoformat())
        current_time = str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')
        generated_model_dir_name = page.model_metadata['modelType'] + '_' + current_date + '_' + current_time
        return generated_model_dir_name

    def save_model_processing(e):
        os.mkdir(f'models/{model_saving_text_field.value}')

        with open(f'models/{model_saving_text_field.value}/model_metadata.json', 'w') as f:
            json.dump(page.model_metadata, f)

        for model_data in page.model_list:
            pickle.dump(model_data['model'], open(f'models/{model_saving_text_field.value}/{model_data["target"]}.p',
                                                  'wb'))  # pickle.load(open(filename, 'rb'))

        page.counter_saved_model_selector = 0
        page.counter_predict_model_selector = 0
        dialog_model_saving.open = False
        dialog_model_saving.update()
        save_model_button.icon = ft.icons.CHECK
        save_model_button.icon_color = ft.colors.GREEN_500
        save_model_button.tooltip = model_saving_text_field.value
        save_model_button.disabled = True
        save_model_button.update()

    def open_dialog_model_saving(e):
        model_saving_text_field.value = get_generated_model_dir_name(e)
        page.dialog = dialog_model_saving
        dialog_model_saving.open = True
        page.update()

    model_saving_text_field = ft.TextField(label='Имя модели', width=1000, height=70,
                                           text_align=ft.alignment.center,
                                           multiline=False,
                                           suffix_text=None
                                           )

    dialog_model_saving = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=model_saving_text_field,
        actions=[
            ft.TextButton('Submit', on_click=save_model_processing)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    save_model_button = ft.ElevatedButton('Сохранить модели',
                                          icon=ft.icons.DATA_SAVER_ON,
                                          scale=0.85,
                                          on_click=open_dialog_model_saving,
                                          icon_color=ft.colors.ORANGE_500,
                                          style=ft.ButtonStyle(bgcolor={
                                              ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                          }),
                                          disabled=True
                                          )
    # PMML MODEL EXPORT

    def export_model_processing(e):
        os.mkdir(f'pmml_models/{model_saving_text_field.value}')

        for model_data in page.model_list:
            sklearn2pmml(model_data['model'], f'pmml_models/{model_saving_text_field.value}/{model_data["target"]}.pmml', with_repr=False)

        dialog_model_export.open = False
        dialog_model_export.update()
        export_to_pmml_button.icon = ft.icons.CHECK
        export_to_pmml_button.icon_color = ft.colors.GREEN_500
        export_to_pmml_button.disabled = True
        export_to_pmml_button.update()

    def open_dialog_model_export(e):
        model_saving_text_field.value = get_generated_model_dir_name(e)
        page.dialog = dialog_model_export
        dialog_model_export.open = True
        page.update()

    dialog_model_export = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=model_saving_text_field,
        actions=[
            ft.TextButton('Submit', on_click=export_model_processing)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    export_to_pmml_button = ft.ElevatedButton('Экспорт в PMML',
                                          icon=ft.icons.IMPORT_EXPORT,
                                          scale=0.85,
                                          on_click=open_dialog_model_export,
                                          icon_color=ft.colors.ORANGE_500,
                                          style=ft.ButtonStyle(bgcolor={
                                              ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                          }),
                                          disabled=True
                                          )

    # MODEL DELETE

    def submit_delete_model(e):
        model_dirs = []
        for checkbox in select_delete_model_ad.content.controls:
            if checkbox.value is True:
                model_dirs.append(os.path.join('models', checkbox.label))
        if len(model_dirs) > 0:
            for path in model_dirs:
                shutil.rmtree(path)

        page.counter_predict_model_selector = 0
        page.counter_saved_model_selector = 0
        select_delete_model_ad.open = False
        select_delete_model_ad.update()

    def saved_model_checkbox_generator():
        checkbox_list = []
        for saved_model_dir in os.listdir('models'):
            checkbox_list.append(ft.Checkbox(label=saved_model_dir, value=False))
        return checkbox_list

    def open_delete_saved_model_ad(e):
        select_delete_model_ad.content.controls = saved_model_checkbox_generator()
        page.dialog = select_delete_model_ad
        select_delete_model_ad.open = True
        page.update()

    select_delete_model_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column([
        ],
            scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_delete_model)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    delete_model_button = ft.ElevatedButton('Удалить модели',
                                            icon=ft.icons.FOLDER_DELETE,
                                            scale=0.85,
                                            on_click=open_delete_saved_model_ad,
                                            icon_color=ft.colors.ORANGE_500,
                                            disabled=False
                                            )

    # LEFT RIGHT TITLE
    left_modeling_title = ft.Row(
        [
            ft.Text('ВИЗУАЛИЗАЦИЯ ДАННЫХ И ОБУЧЕНИЕ МОДЕЛЕЙ', style=ft.TextStyle(weight=ft.FontWeight.BOLD, color=ft.colors.ORANGE_800))
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    right_modeling_title = ft.Row(
        [
            ft.Text('РАСЧЕТ И АНАЛИЗ МОДЕЛИ', style=ft.TextStyle(weight=ft.FontWeight.BOLD, color=ft.colors.ORANGE_800))
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # PREDICT ELEMENTS
    # MODEL SELECTING (PREDICT)

    page.counter_predict_model_selector = 0

    def submit_predict_saved_model(e):
        if radio_select_predict_model.value is not None:
            with open(f'models/{radio_select_predict_model.value}/model_metadata.json', 'r') as file:
                page.model_metadata = json.load(file)

            page.model_list = []
            for model_saved in os.listdir(f'models/{radio_select_predict_model.value}'):
                if model_saved != 'model_metadata.json':
                    model_data = {'target': model_saved.split('.')[0]}
                    with open(f'models/{radio_select_predict_model.value}/{model_saved}', 'rb') as file:
                        model_data['model'] = pickle.load(file)
                    page.model_list.append(model_data)

            txt_select_predict_model.value = f'Модель загружена: {radio_select_predict_model.value}'
            txt_select_predict_model.color = ft.colors.GREEN_700

        select_predict_model_ad.open = False
        select_predict_model_ad.update()
        txt_select_predict_model.update()
        inputs_predict_content_generator(e)
        outputs_predict_content_generator(e)
        inference_field.visible = True
        inference_field.update()
        analyze_targets_dd_generator(e)
        model_analyze_field.visible = True
        model_analyze_field.update()

        print(page.model_metadata)
        print(page.model_list)

    def open_select_predict_model_ad(e):
        if page.counter_predict_model_selector == 0:
            radio_select_predict_model.content.controls = saved_model_generator()  # From opt field
            page.counter_predict_model_selector += 1
        page.dialog = select_predict_model_ad
        select_predict_model_ad.open = True
        page.update()

    radio_select_predict_model = ft.RadioGroup(content=ft.Column(controls=None))

    select_predict_model_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column([
            radio_select_predict_model,
        ],
            scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_predict_saved_model)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    select_predict_model_button = ft.ElevatedButton(text='Модели',
                                                    icon=ft.icons.FILE_PRESENT,
                                                    icon_color=ft.colors.ORANGE_500,
                                                    scale=0.85,
                                                    on_click=open_select_predict_model_ad)

    txt_select_predict_model = ft.Text(value='Выберите модель', color=ft.colors.RED_700, weight=ft.FontWeight.BOLD,
                                       scale=0.85)

    model_selecting_predict = ft.Row(
        [
            select_predict_model_button,
            txt_select_predict_model,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # INFERENCE INPUT/OUTPUT/BUTTON_PREDICT

    def text_color_gen(e):
        for txt_field in inputs_predict_content.controls[1:]:
            try:
                if (float(txt_field.value) > page.model_metadata[txt_field.label]['max']) or (float(txt_field.value) < page.model_metadata[txt_field.label]['min']):
                    txt_field.color = ft.colors.RED_700
                else:
                    txt_field.color = ft.colors.BLUE_700
            except (ValueError, TypeError):
                txt_field.color = ft.colors.RED_700
        inputs_predict_content.update()

    def inputs_predict_content_generator(e):
        while len(inputs_predict_content.controls) > 1:
            inputs_predict_content.controls.pop()
        for input_name in page.model_metadata.keys():
            if input_name != 'modelType' and input_name != 'targets':
                try:
                    suffix_unit_text = ' ' + page.units_dict[input_name]
                except (TypeError, KeyError):
                    suffix_unit_text = ' '
                inputs_predict_content.controls.append(ft.TextField(label=input_name,
                                                                    label_style=ft.TextStyle(
                                                                      color=ft.colors.BLACK),
                                                                    color=ft.colors.BLUE_700,
                                                                    text_size=15,
                                                                    focused_border_color=ft.colors.BLUE_700,
                                                                    cursor_color=ft.colors.BLUE_700,
                                                                    value=None,
                                                                    suffix_text=suffix_unit_text,
                                                                    suffix_style=ft.TextStyle(color=ft.colors.BLUE_900),
                                                                    tooltip=f"min: {page.model_metadata[input_name]['min']}\n"
                                                                            f"max: {page.model_metadata[input_name]['max']}",
                                                                    on_change=text_color_gen,
                                                                    scale=0.75,
                                                                    height=40,
                                                                    width=250, ),)

        inputs_predict_content.update()

    inputs_predict_content = ft.Column([
        ft.Text('Входные данные модели', color=ft.colors.BLACK, weight=ft.FontWeight.BOLD, scale=0.85),
    ],
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    input_data_predict_container = ft.Container(content=inputs_predict_content,
                                 alignment=ft.alignment.top_center,
                                 expand=True)

    def outputs_predict_content_generator(e):
        while len(outputs_predict_content.controls) > 1:
            outputs_predict_content.controls.pop()
        for target_name in page.model_metadata['targets']:
            outputs_predict_content.controls.append(ft.Text(value=target_name + ': ',
                                                            color=ft.colors.BLACK,
                                                            scale=0.85,
                                                            ),)

    outputs_predict_content = ft.Column(
        [
            ft.Text('Выходные данные модели', color=ft.colors.BLACK, weight=ft.FontWeight.BOLD, scale=0.85),
        ],
        spacing=2,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    output_data_predict_container = ft.Container(content=outputs_predict_content,
                                 alignment=ft.alignment.top_center,
                                 expand=True)

    def start_predict_func(e):
        is_correct_inputs = True
        inputs_dict = {}
        for input_txt_field in inputs_predict_content.controls[1:]:  # skip title
            try:
                inputs_dict[input_txt_field.label] = float(input_txt_field.value)
            except ValueError:
                is_correct_inputs = False

        targets_values_dict = dict.fromkeys(page.model_metadata['targets'])

        if is_correct_inputs:
            for model_dict in page.model_list:
                input_arr = []
                for feat_name in inputs_dict.keys():
                    norm_val = inputs_dict[feat_name] - page.model_metadata[feat_name]['min']
                    if page.model_metadata[feat_name]['min'] != page.model_metadata[feat_name]['max']:
                        norm_val /= (page.model_metadata[feat_name]['max'] -
                                     page.model_metadata[feat_name]['min'])
                    input_arr.append(norm_val)
                res = model_dict['model'].predict([input_arr])
                targets_values_dict[model_dict['target']] = res[0]

            for output_txt_field in outputs_predict_content.controls[1:]:  # skip title
                output_txt_field.value = output_txt_field.value.split(': ')[0]
                try:
                    output_txt_field.value += ': ' + str(round(targets_values_dict[output_txt_field.value], 2)) + ' ' + page.units_dict[output_txt_field.value]
                except (TypeError, KeyError):
                    output_txt_field.value += ': ' + str(round(targets_values_dict[output_txt_field.value], 2))

            outputs_predict_content.update()

    start_predict_button = ft.ElevatedButton(text='Рассчитать',
                                             icon=ft.icons.PLAY_ARROW,
                                             icon_color=ft.colors.ORANGE_500,
                                             scale=0.85,
                                             on_click=start_predict_func)

    inference_field = ft.Column(
        [
            ft.Row(
                [
                    input_data_predict_container,
                    output_data_predict_container,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    start_predict_button,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        ],
        visible=False,
        alignment=ft.MainAxisAlignment.START
    )

    # MODEL ANALYZE

    def analyze_model_visualizer(e):
        step = 0.05
        current_model = None
        model_inputs_num = len(page.model_metadata.keys()) - 2
        model_fim = {}
        for k in page.model_metadata.keys():
            if k != 'modelType' and k != 'targets':
                model_fim[k] = None
        inputs_order = list(model_fim.keys())
        for model_dict in page.model_list:
            if model_dict['target'] == analyze_targets_dd.value:
                current_model = model_dict['model']
                break

        gen_one_diag = np.eye(model_inputs_num)
        gen_zero_diag = np.ones([model_inputs_num, model_inputs_num])
        gen_zero_diag[np.diag_indices(4)] = 0
        input_arr = np.vstack([np.zeros(model_inputs_num), gen_one_diag, gen_zero_diag, np.ones(model_inputs_num)])

        pred_arr = current_model.predict(input_arr)

        zero_diff = abs(pred_arr[1:model_inputs_num+1] - pred_arr[0])
        one_diff = abs(pred_arr[-model_inputs_num-1:-1] - pred_arr[-1])
        fim_val = (zero_diff + one_diff) / np.sum(zero_diff + one_diff)

        for idx_k, k in enumerate(model_fim.keys()):
            model_fim[k] = fim_val[idx_k]
        model_fim = dict(sorted(model_fim.items(), key=lambda item: item[1], reverse=True))

        correct_names = [val_k[:10] + '\n' + val_k[10:20] for val_k in model_fim.keys()]

        fig, ax = plt.subplots(figsize=(9, 15))

        ax.barh(correct_names[:10], list(model_fim.values())[:10], color='sandybrown')  # list(model_fim.keys())[:10]

        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.invert_yaxis()
        plt.xlim(0, 1)
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     str(round((i.get_width()), 2)),
                     fontsize=20, fontweight='bold',
                     color='blue')
        ax.set_title('Диаграмма чувствительности\nТоп-10',
                     loc='center', fontsize=24)
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        ax.grid(visible=True, color='blue',
                linestyle='-.', linewidth=0.5,
                alpha=0.4)

        fim_chart_container.content = MatplotlibChart(fig, transparent=True, original_size=False)
        fim_chart_container.update()
        plt.close()

        x1_input_idx = inputs_order.index(list(model_fim.keys())[0])
        x2_input_idx = inputs_order.index(list(model_fim.keys())[1])

        x1_range = np.arange(0, 1 + step, step)
        x2_range = np.arange(0, 1 + step, step)
        x1, x2 = np.meshgrid(x1_range, x2_range)
        x1_inputs = x1.flatten()
        x2_inputs = x2.flatten()

        model_inputs_arr = []
        for i in range(model_inputs_num):
            if i == x1_input_idx:
                model_inputs_arr.append(x1_inputs)
            elif i == x2_input_idx:
                model_inputs_arr.append(x2_inputs)
            else:
                model_inputs_arr.append(np.full(len(x1_inputs), 0.5))

        model_inputs_arr = np.vstack(model_inputs_arr)
        model_inputs_arr = model_inputs_arr.transpose()

        model_output_arr = current_model.predict(model_inputs_arr)
        model_output_arr = model_output_arr.reshape(len(x1_range), len(x2_range))

        X = np.linspace(page.model_metadata[list(model_fim.keys())[0]]['min'],
                        page.model_metadata[list(model_fim.keys())[0]]['max'], int(1/step +1))
        Y = np.linspace(page.model_metadata[list(model_fim.keys())[1]]['min'],
                        page.model_metadata[list(model_fim.keys())[1]]['max'], int(1 / step + 1))
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9, 9))
        surf = ax.plot_surface(X, Y, model_output_arr, cmap='copper',
                               linewidth=0, antialiased=False)
        if page.df_to_view is not None:
            ax.scatter(page.df_to_view[list(model_fim.keys())[0]], page.df_to_view[list(model_fim.keys())[1]], page.df_to_view[analyze_targets_dd.value], s=70, color='saddlebrown')
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=14)
        ax.set_title('3D график модели',
                     loc='center', fontsize=24)
        plt.tick_params(labelsize=14)

        ax.set_ylabel(list(model_fim.keys())[1][:20], fontsize=16)
        ax.set_xlabel(list(model_fim.keys())[0][:20], fontsize=16)
        plt.tight_layout()

        d3_chart_surface.content = MatplotlibChart(fig, transparent=True, original_size=False)
        d3_chart_surface.update()
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 9))
        sns.heatmap(model_output_arr,
                    annot=False,
                    cmap='copper',
                    cbar=True,
                    square=True,
                    ax=ax,
                    cbar_kws={"shrink": 0.75})
        ax.set_title(f'Тепловая карта', fontsize=24)
        ax.set_ylabel(list(model_fim.keys())[1][:20], fontsize=16)
        ax.set_xlabel(list(model_fim.keys())[0][:20], fontsize=16)
        ax.invert_yaxis()
        plt.tight_layout()

        heatmap_chart_model.content = MatplotlibChart(fig, transparent=True, original_size=False)
        heatmap_chart_model.update()
        plt.close()

    def analyze_targets_dd_generator(e):
        analyze_targets_dd_options = []
        for target in page.model_metadata['targets']:
            analyze_targets_dd_options.append(ft.dropdown.Option(target))
        analyze_targets_dd.options = analyze_targets_dd_options
        analyze_targets_dd.update()

    analyze_targets_dd = ft.Dropdown(
        label='Выход модели',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать выход модели',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=None,
        autofocus=True,
        border_color=ft.colors.ORANGE_100,
        focused_border_color=ft.colors.ORANGE_500,
        bgcolor=ft.colors.GREY_400,
        scale=0.75,
        width=350,
        on_change=analyze_model_visualizer,
    )

    fim_chart_container = ft.Container(content=None,
                                       height=650,
                                       width=350,
                                       alignment=ft.alignment.top_center,
                                       )
    d3_chart_surface = ft.Container(content=None,
                                    height=350,
                                    width=350,
                                    alignment=ft.alignment.top_center,
                                    )
    heatmap_chart_model = ft.Container(content=None,
                                 height=350,
                                 width=350,
                                 alignment=ft.alignment.top_center,)

    model_analyze_field = ft.Row(
        [
            ft.Column(
                [
                    ft.Container(content=analyze_targets_dd,
                                 height=50,
                                 width=350,
                                 alignment=ft.alignment.top_center,
                                 ),
                    fim_chart_container,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Column(
                [
                    d3_chart_surface,
                    heatmap_chart_model,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
        visible=False,
    )

    # ML WINDOW

    ml = ft.Container(content=ft.Row(  # Column
        [
            ft.Container(
                content=ft.Column(
                    [
                        left_modeling_title,
                        hist_dist_field,
                        ft.Row(
                            [
                                model_type_dd,
                                model_options_field,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER
                        ),
                        ft.Row(
                            [
                                feature_select_button,
                                target_select_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER
                        ),
                        ft.Row(
                            [
                                model_training_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        pr_training,
                        ft.Row(
                            [
                                save_model_button,
                                export_to_pmml_button,
                                delete_model_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        ft.Row(
                            [
                                show_charts_button,
                                save_pred_data_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    scroll=ft.ScrollMode.ALWAYS,
                ),
                height=page.window_height - my_app_bar_h,
                alignment=ft.alignment.top_center,
                expand=True,
            ),
            ft.VerticalDivider(width=10, thickness=2, color=ft.colors.BLACK),
            ft.Container(content=ft.Column(
                [
                    right_modeling_title,
                    model_selecting_predict,
                    inference_field,
                    model_analyze_field,
                ],
                scroll=ft.ScrollMode.ALWAYS
            ),
                height=page.window_height - my_app_bar_h,
                alignment=ft.alignment.top_center,
                expand=True,
            ),
        ],

        height=page.window_height - my_app_bar_h,
    ),
        margin=ft.margin.only(top=my_app_bar_h - 5),
        visible=False,
    )

    # --- OPTIMIZATION --- #

    # MODEL SELECTING
    page.counter_saved_model_selector = 0

    def submit_selected_saved_model(e):
        if radio_select_saved_model.value is not None:
            with open(f'models/{radio_select_saved_model.value}/model_metadata.json', 'r') as file:
                page.model_metadata = json.load(file)

            page.model_list = []
            for model_saved in os.listdir(f'models/{radio_select_saved_model.value}'):
                if model_saved != 'model_metadata.json':
                    model_data = {'target': model_saved.split('.')[0]}
                    with open(f'models/{radio_select_saved_model.value}/{model_saved}', 'rb') as file:
                        model_data['model'] = pickle.load(file)
                    page.model_list.append(model_data)

            txt_select_model.value = f'Модель загружена: {radio_select_saved_model.value}'
            txt_select_model.color = ft.colors.GREEN_700

        select_saved_model_ad.open = False
        select_saved_model_ad.update()
        txt_select_model.update()
        input_param_table_generator(e)
        criteria_bound_table_generator(e)
        algorithm_opt_selector.visible = True
        algorithm_opt_selector.update()
        print(page.model_metadata)
        print(page.model_list)

    def saved_model_generator():
        radio_dir_model_names = []
        for saved_model_dir in os.listdir('models'):
            radio_dir_model_names.append(ft.Radio(value=saved_model_dir, label=saved_model_dir))
        return radio_dir_model_names

    def open_select_saved_model_ad(e):
        if page.counter_saved_model_selector == 0:
            radio_select_saved_model.content.controls = saved_model_generator()
            page.counter_saved_model_selector += 1
        page.dialog = select_saved_model_ad
        select_saved_model_ad.open = True
        page.update()

    radio_select_saved_model = ft.RadioGroup(content=ft.Column(controls=None))

    select_saved_model_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column([
            radio_select_saved_model,
        ],
            scroll='adaptive'),
        actions=[
            ft.TextButton('Submit', on_click=submit_selected_saved_model)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        actions_padding=None,
    )

    select_saved_model_button = ft.ElevatedButton(text='Модели',
                                                  icon=ft.icons.FILE_PRESENT,
                                                  icon_color=ft.colors.ORANGE_500,
                                                  scale=0.85,
                                                  on_click=open_select_saved_model_ad)

    txt_select_model = ft.Text(value='Выберите модель', color=ft.colors.RED_700, weight=ft.FontWeight.BOLD, scale=0.85)

    # COMPONENTS BR

    def component_br_button_func(e):
        if page.model_metadata is None and page.cost_dict is None:
            component_br_txt_field.controls[0].content = ft.Text(
                value='Выберите модель и загрузите данные с ценами на странице "Загрузка данных"',
                color=ft.colors.RED_700,
                weight=ft.FontWeight.BOLD,
                scale=0.85)
        elif page.model_metadata is not None and page.cost_dict is None:
            component_br_txt_field.controls[0].content = ft.Text(
                value='Загрузите данные с ценами на странице "Загрузка данных"',
                color=ft.colors.RED_700,
                weight=ft.FontWeight.BOLD,
                scale=0.85)
        elif page.model_metadata is None and page.cost_dict is not None:
            component_br_txt_field.controls[0].content = ft.Text(
                value='Выберите или обучите модель',
                color=ft.colors.RED_700,
                weight=ft.FontWeight.BOLD,
                scale=0.85)
        else:
            component_br_txt_field.controls[0].content = ft.Row(
                [

                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=0,
            )
            for component_name in page.cost_dict['names']:
                if component_name not in page.model_metadata['targets'] and component_name not in page.model_metadata.keys():
                    try:
                        suffix_unit_name = page.units_dict[component_name]
                    except (TypeError, KeyError):
                        suffix_unit_name = ''
                    component_br_txt_field.controls[0].content.controls.append(ft.TextField(label=component_name,
                                                                                            label_style=ft.TextStyle(
                                                                                              color=ft.colors.BLACK),
                                                                                            color=ft.colors.BLUE_700,
                                                                                            text_size=15,
                                                                                            focused_border_color=ft.colors.BLUE_700,
                                                                                            cursor_color=ft.colors.BLUE_700,
                                                                                            value=0,
                                                                                            suffix_text=suffix_unit_name,
                                                                                            suffix_style=ft.TextStyle(color=ft.colors.BLUE_900),
                                                                                            scale=0.7,
                                                                                            width=150, ),)
            if len(component_br_txt_field.controls[0].content.controls) == 0:
                component_br_txt_field.controls[0].content = ft.Text(
                value='Все компоненты БР используются в качестве входных параметров модели',
                color=ft.colors.BLUE_700,
                weight=ft.FontWeight.BOLD,
                scale=0.85)

        component_br_button.icon = ft.icons.AUTORENEW
        component_br_button.text = 'Обновить компонентный состав БР'
        component_br_button.update()
        component_br_txt_field.visible = True
        component_br_txt_field.update()

    component_br_button = ft.ElevatedButton(text='Компонентный состав БР',
                                            icon=ft.icons.SETTINGS_INPUT_COMPONENT,
                                            icon_color=ft.colors.ORANGE_500,
                                            scale=0.85,
                                            on_click=component_br_button_func,
                                            tooltip='Компонентный состав бурового раствора по параметрам, не участвующим в моделях')

    component_br_button_field = ft.Row(
        [
            component_br_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    component_br_txt_field = ft.Row(
        [
            ft.Container(content=None)
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
        spacing=0,
    )

    # SHOW AND HIDE NEXT FIELD
    page.counter_show_hide_opt_inputs = 0

    def show_hide_opt_inputs_func(e):
        if page.counter_show_hide_opt_inputs % 2 == 0:
            params_criteria_field.visible = True
            show_hide_opt_inputs_button.text = 'Скрыть параметры оптимизации'
            show_hide_opt_inputs_button.icon = ft.icons.ARROW_DROP_UP

        else:
            params_criteria_field.visible = False
            show_hide_opt_inputs_button.text = 'Показать параметры оптимизации'
            show_hide_opt_inputs_button.icon = ft.icons.ARROW_DROP_DOWN

        params_criteria_field.update()
        show_hide_opt_inputs_button.update()
        page.counter_show_hide_opt_inputs += 1
        pareto_chart_field.update()
        algorithm_opt_selector.update()

    show_hide_opt_inputs_button = ft.ElevatedButton(text='Показать параметры оптимизации',
                                                    icon=ft.icons.ARROW_DROP_DOWN,
                                                    icon_color=ft.colors.ORANGE_500,
                                                    scale=0.85,
                                                    on_click=show_hide_opt_inputs_func,
                                                    autofocus=True)

    show_hide_opt_inputs_field = ft.Row(
        [
            show_hide_opt_inputs_button,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # OPTIMIZATION PARAMETERS, CRITERIA, BOUNDS

    def input_param_table_generator(e):
        for idx_col, col in enumerate(input_params_table.controls):
            col.content = ft.Column([], spacing=0, )
            if page.model_metadata is not None:
                if idx_col == 0:
                    for feat_name in page.model_metadata.keys():
                        if feat_name != 'modelType' and feat_name != 'targets':
                            col.content.controls.append(ft.Container(content=ft.Text(feat_name,
                                                                                     color=ft.colors.BLACK,
                                                                                     scale=0.7,
                                                                                     weight=ft.FontWeight.BOLD),
                                                                     height=38,
                                                                     alignment=ft.alignment.center))
                elif idx_col == 1:
                    for feat_name in page.model_metadata.keys():
                        if feat_name != 'modelType' and feat_name != 'targets':
                            try:
                                suffix_unit_name = page.units_dict[feat_name]
                            except (TypeError, KeyError):
                                suffix_unit_name = ''
                            col.content.controls.append(ft.Container(content=ft.TextField(label='минимум',
                                                                                          label_style=ft.TextStyle(
                                                                                              color=ft.colors.BLACK),
                                                                                          color=ft.colors.BLUE_700,
                                                                                          text_size=15,
                                                                                          counter_text=None,
                                                                                          focused_border_color=ft.colors.BLUE_700,
                                                                                          cursor_color=ft.colors.BLUE_700,
                                                                                          value=str(np.round(page.model_metadata[
                                                                                                        feat_name][
                                                                                                        'min'], 2))[:13],
                                                                                          suffix_text=suffix_unit_name,
                                                                                          suffix_style=ft.TextStyle(color=ft.colors.BLUE_900),
                                                                                          scale=0.7,
                                                                                          width=150, ),
                                                                     height=38,
                                                                     alignment=ft.alignment.center))
                elif idx_col == 2:
                    for feat_name in page.model_metadata.keys():
                        if feat_name != 'modelType' and feat_name != 'targets':
                            try:
                                suffix_unit_name = page.units_dict[feat_name]
                            except (TypeError, KeyError):
                                suffix_unit_name = ''
                            col.content.controls.append(ft.Container(content=ft.TextField(label='максимум',
                                                                                          label_style=ft.TextStyle(
                                                                                              color=ft.colors.BLACK),
                                                                                          color=ft.colors.BLUE_700,
                                                                                          text_size=15,
                                                                                          focused_border_color=ft.colors.BLUE_700,
                                                                                          cursor_color=ft.colors.BLUE_700,
                                                                                          value=str(np.round(page.model_metadata[
                                                                                                        feat_name][
                                                                                                        'max'], 2))[:13],
                                                                                          suffix_text=suffix_unit_name,
                                                                                          suffix_style=ft.TextStyle(
                                                                                              color=ft.colors.BLUE_900),
                                                                                          scale=0.7,
                                                                                          width=150, ),
                                                                     height=38,
                                                                     alignment=ft.alignment.center))
                else:
                    for feat_name in page.model_metadata.keys():
                        if feat_name != 'modelType' and feat_name != 'targets':
                            try:
                                suffix_unit_name = page.units_dict[feat_name]
                            except (TypeError, KeyError):
                                suffix_unit_name = ''
                            col.content.controls.append(ft.Container(content=ft.TextField(label='шаг',
                                                                                          label_style=ft.TextStyle(
                                                                                              color=ft.colors.BLACK),
                                                                                          color=ft.colors.BLUE_700,
                                                                                          text_size=15,
                                                                                          focused_border_color=ft.colors.BLUE_700,
                                                                                          cursor_color=ft.colors.BLUE_700,
                                                                                          suffix_text=suffix_unit_name,
                                                                                          suffix_style=ft.TextStyle(
                                                                                              color=ft.colors.BLUE_900),
                                                                                          value=0.01,
                                                                                          scale=0.7,
                                                                                          width=150, ),
                                                                     height=38,
                                                                     alignment=ft.alignment.center))
        if page.model_metadata is not None:
            count_feats = 0
            for k in page.model_metadata.keys():
                if k != 'modelType' and k != 'targets':
                    count_feats += 1
            max_row_counter = max(len(page.model_metadata['targets']), count_feats)
            params_criteria_field.height = 20 + (max_row_counter+1) * 38

        params_criteria_field.update()

    input_params_table = ft.Row(
        [
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=0,
    )

    content_input_param_opt = ft.Column(
        [
            ft.Row(
                [
                    ft.Text('Входные параметры', weight=ft.FontWeight.BOLD, color=ft.colors.BLACK,
                            scale=0.85),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                height=20,
            ),
            input_params_table,

        ],
        spacing=0,
    )

    input_params_opt_field = ft.Container(
        content=content_input_param_opt,
        expand=True,
        alignment=ft.alignment.top_center,
    )

    def criteria_bound_table_generator(e):
        for idx_col, col in enumerate(criteria_bound_table.controls):
            col.content = ft.Column([], spacing=0, )
            if page.model_metadata is not None:
                if idx_col == 0:
                    for target_name in ['Стоимость БР'] + page.model_metadata['targets']:
                        col.content.controls.append(ft.Container(content=ft.Text(target_name,
                                                                                 color=ft.colors.BLACK,
                                                                                 scale=0.7,
                                                                                 weight=ft.FontWeight.BOLD),
                                                                 height=38,
                                                                 alignment=ft.alignment.center))
                elif idx_col == 1:
                    for _ in ['Стоимость БР'] + page.model_metadata['targets']:
                        col.content.controls.append(ft.Container(content=ft.RadioGroup(content=ft.Row([
                            ft.Radio(label='мин', value='minimize', scale=0.7,
                                     label_style=ft.TextStyle(color=ft.colors.BLACK),
                                     overlay_color=ft.colors.GREY_400,
                                     fill_color=ft.colors.ORANGE_500,
                                     toggleable=True,
                                     tooltip='минимизация'),
                            ft.Radio(label='макс', value='maximize', scale=0.7,
                                     label_style=ft.TextStyle(color=ft.colors.BLACK),
                                     overlay_color=ft.colors.GREY_400,
                                     fill_color=ft.colors.ORANGE_900,
                                     toggleable=True,
                                     tooltip='максимизация'),
                        ],
                            spacing=0,
                            alignment=ft.MainAxisAlignment.CENTER
                        ), ),
                            height=38,
                            alignment=ft.alignment.center))
                elif idx_col == 2:
                    for _ in ['Стоимость БР'] + page.model_metadata['targets']:
                        try:
                            suffix_unit_name = page.units_dict[_]
                        except (TypeError, KeyError):
                            suffix_unit_name = ''
                        col.content.controls.append(ft.Container(content=ft.TextField(label='>=',
                                                                                      label_style=ft.TextStyle(
                                                                                          color=ft.colors.BLACK),
                                                                                      color=ft.colors.BLUE_700,
                                                                                      text_size=15,
                                                                                      focused_border_color=ft.colors.BLUE_700,
                                                                                      cursor_color=ft.colors.BLUE_700,
                                                                                      suffix_text=suffix_unit_name,
                                                                                      suffix_style=ft.TextStyle(
                                                                                          color=ft.colors.BLUE_900),
                                                                                      scale=0.7,
                                                                                      width=150, ),
                                                                 height=38,
                                                                 alignment=ft.alignment.center))

                elif idx_col == 3:
                    for _ in ['Стоимость БР'] + page.model_metadata['targets']:
                        try:
                            suffix_unit_name = page.units_dict[_]
                        except (TypeError, KeyError):
                            suffix_unit_name = ''
                        col.content.controls.append(ft.Container(content=ft.TextField(label='<=',
                                                                                      label_style=ft.TextStyle(
                                                                                          color=ft.colors.BLACK),
                                                                                      color=ft.colors.BLUE_700,
                                                                                      text_size=15,
                                                                                      focused_border_color=ft.colors.BLUE_700,
                                                                                      cursor_color=ft.colors.BLUE_700,
                                                                                      suffix_text=suffix_unit_name,
                                                                                      suffix_style=ft.TextStyle(
                                                                                          color=ft.colors.BLUE_900),
                                                                                      scale=0.7,
                                                                                      width=150, ),
                                                                 height=38,
                                                                 alignment=ft.alignment.center))
                        if _ == 'Стоимость БР':
                            col.content.controls[-1].content.value = '1000000'
                else:
                    for _ in ['Стоимость БР'] + page.model_metadata['targets']:
                        col.content.controls.append(ft.Container(content=ft.TextField(label='Вес критерия',
                                                                                      label_style=ft.TextStyle(
                                                                                          color=ft.colors.BLACK),
                                                                                      color=ft.colors.BLUE_700,
                                                                                      text_size=15,
                                                                                      focused_border_color=ft.colors.BLUE_700,
                                                                                      cursor_color=ft.colors.BLUE_700,
                                                                                      scale=0.7,
                                                                                      width=150, ),
                                                                 height=38,
                                                                 alignment=ft.alignment.center))
        params_criteria_field.update()

    criteria_bound_table = ft.Row(
        [
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
            ft.Container(expand=True, alignment=ft.alignment.top_center, content=None),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=0,
    )

    content_criteria_bound_opt = ft.Column(
        [
            ft.Row(
                [
                    ft.Text('Критерии и ограничения', weight=ft.FontWeight.BOLD, color=ft.colors.BLACK,
                            scale=0.85),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                height=20,
            ),
            criteria_bound_table,
        ],
        spacing=0,
    )

    input_criteria_opt_field = ft.Container(
        content=content_criteria_bound_opt,
        expand=True,
        alignment=ft.alignment.top_center,
    )

    params_criteria_field = ft.Row(
        [
            input_params_opt_field,
            ft.VerticalDivider(width=15, thickness=2, color=ft.colors.BLACK),
            input_criteria_opt_field,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=0,
        height=20,
        visible=False,
    )

    # SELECTING ALGORITHM OPTIMIZATION AND OPTIONS

    page.res_opt_str = None
    page.dump_opt_data = None
    page.is_multi_obj = False
    page.criteria_names = None
    page.criteria_dict = None

    def auto_change_options(e):
        if opt_algorithm_dd.value == 'TPE':
            opt_options_container.content = opt_options_TPE_row
        else:
            opt_options_container.content = opt_options_GA_row
        opt_options_container.update()

    opt_algorithm_dd = ft.Dropdown(
        label='Алгоритм оптимизации',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выбрать алгоритм оптимизации',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=[ft.dropdown.Option('NSGA-II'),
                 ft.dropdown.Option('TPE'),
                 ],
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=None,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.85,
        width=250,
        on_change=auto_change_options,
        focused_color=None,
    )
    opt_algorithm_container = ft.Container(content=opt_algorithm_dd,
                                           height=45,
                                           bgcolor=ft.colors.GREY_400, )

    population_size_text_field = ft.TextField(label='Количество точек поиска',
                                              label_style=ft.TextStyle(
                                                  color=ft.colors.BLACK),
                                              value=25,
                                              color=ft.colors.BLUE_700,
                                              text_size=15,
                                              focused_border_color=ft.colors.BLUE_700,
                                              cursor_color=ft.colors.BLUE_700,
                                              scale=0.85,
                                              width=150, )

    num_generation_text_field = ft.TextField(label='Количество шагов поиска',
                                             label_style=ft.TextStyle(
                                                 color=ft.colors.BLACK),
                                             value=100,
                                             color=ft.colors.BLUE_700,
                                             text_size=15,
                                             focused_border_color=ft.colors.BLUE_700,
                                             cursor_color=ft.colors.BLUE_700,
                                             scale=0.85,
                                             width=150, )

    use_step_checkbox = ft.Checkbox(label='Использовать заданный шаг', value=False, scale=0.85,
                                    check_color=ft.colors.ORANGE_700,
                                    active_color=ft.colors.ORANGE_300,
                                    border_side=ft.BorderSide(width=2, color=ft.colors.ORANGE),
                                    label_style=ft.TextStyle(color=ft.colors.BLACK))

    opt_options_GA_row = ft.Row(
        [
            population_size_text_field,
            num_generation_text_field,
            use_step_checkbox,
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )

    num_trials_text_field = ft.TextField(label='Количество шагов поиска',
                                         label_style=ft.TextStyle(
                                             color=ft.colors.BLACK),
                                         value=2500,
                                         color=ft.colors.BLUE_700,
                                         text_size=15,
                                         focused_border_color=ft.colors.BLUE_700,
                                         cursor_color=ft.colors.BLUE_700,
                                         scale=0.85,
                                         width=150,
                                         tooltip='Количество итераций алгоритма')

    opt_options_TPE_row = ft.Row(
        [
            num_trials_text_field,
            use_step_checkbox,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    opt_options_container = ft.Container(content=None,
                                         height=45,
                                         visible=True)

    def start_opt_func(e):
        pr_optimization.visible = True
        pr_optimization.controls[1].value = 'Получение данных...'
        pr_optimization.update()
        opt_start_button.disabled = True
        opt_start_button.update()
        show_text_result_opt_button.disabled = True
        show_text_result_opt_button.update()
        save_text_result_opt_button.disabled = True
        save_text_result_opt_button.update()
        pareto_chart_field.visible = False
        pareto_chart_field.update()
        config_opt = {'inputs': {}, 'criteria': {}, 'base_costs': None, 'add_cost': {'names': [], 'values': []}}
        calc_costs_dict = {}
        try:
            if component_br_txt_field.controls[0].content is None:
                component_br_button.autofocus = True
                raise ValueError
            elif type(component_br_txt_field.controls[0].content) is ft.Text:
                config_opt['base_costs'] = 0
            else:
                names_br = []
                volumes = []
                costs = []
                for txt_field in component_br_txt_field.controls[0].content.controls:
                    volumes.append(float(txt_field.value))
                    costs.append(page.cost_dict['values'][page.cost_dict['names'].index(txt_field.label)])
                    names_br.append(txt_field.label)

                calc_costs = np.array(volumes)*np.array(costs)
                calc_costs_dict = dict(zip(names_br, calc_costs))
                config_opt['base_costs'] = np.sum(calc_costs)  # get_cost_br(np.array(volumes), np.array(costs))

            if page.cost_dict is not None:
                for input_feat in page.model_metadata.keys():
                    if input_feat in page.cost_dict['names']:
                        config_opt['add_cost']['names'].append(input_feat)
                        config_opt['add_cost']['values'].append(page.cost_dict['values'][page.cost_dict['names'].index(input_feat)])

            for idx_container, col_container in enumerate(input_params_table.controls):
                col_table = []
                for value_container in col_container.content.controls:
                    if idx_container == 0:
                        col_table.append(value_container.content.value)
                    else:
                        col_table.append(float(value_container.content.value))

                if idx_container == 0:
                    config_opt['inputs']['names'] = col_table
                elif idx_container == 1:
                    config_opt['inputs']['min'] = col_table
                elif idx_container == 2:
                    config_opt['inputs']['max'] = col_table
                else:
                    config_opt['inputs']['step'] = col_table

            for idx_container, col_container in enumerate(criteria_bound_table.controls):
                col_table = []
                for value_container in col_container.content.controls:
                    if idx_container == 0:
                            col_table.append(value_container.content.value)
                    elif idx_container == 1:
                        if value_container.content.value == '':
                            col_table.append(None)
                        else:
                            col_table.append(value_container.content.value)
                    else:
                        try:
                            col_table.append(float(value_container.content.value))
                        except ValueError:
                            col_table.append(None)

                if idx_container == 0:
                    config_opt['criteria']['names'] = col_table
                elif idx_container == 1:
                    config_opt['criteria']['opt'] = col_table
                elif idx_container == 2:
                    config_opt['criteria']['>='] = col_table
                elif idx_container == 3:
                    config_opt['criteria']['<='] = col_table
                else:
                    config_opt['criteria']['weight'] = col_table

            opt_directions = [val for val in config_opt['criteria']['opt'] if val is not None]
            opt_weight = [val for i_val, val in enumerate(config_opt['criteria']['weight']) if config_opt['criteria']['opt'][i_val] is not None]

            opt_weight = np.array([val if val is not None else 0 for val in opt_weight])
            is_opt_weight = False
            opt_weight_sign = []
            study_opt_direction = opt_directions.copy()
            if np.sum(opt_weight) != 0:
                opt_weight = opt_weight / np.sum(opt_weight)
                is_opt_weight = True
                study_opt_direction = ['minimize']
                for opt_direct in opt_directions:
                    if opt_direct == 'minimize':
                        opt_weight_sign.append(1)
                    else:
                        opt_weight_sign.append(-1)

            opt_counter = len(study_opt_direction)
            if opt_counter != 0:
                pr_optimization.controls[1].value = 'Оптимизация...'
                pr_optimization.update()
                if opt_counter == 1:
                    page.is_multi_obj = False
                else:
                    page.is_multi_obj = True
                if opt_algorithm_dd.value == 'TPE':
                    study = optuna.create_study(sampler=optuna.samplers.MOTPESampler(), directions=study_opt_direction)
                    num_trials = int(float(num_trials_text_field.value))
                elif opt_algorithm_dd.value == 'NSGA-II':
                    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(
                        population_size=int(float(population_size_text_field.value))), directions=study_opt_direction)
                    num_trials = int(float(population_size_text_field.value) * float(num_generation_text_field.value))
                else:
                    raise ValueError
                study.optimize(
                    lambda trial: obj_func(trial, page.model_metadata, page.model_list, config_opt, opt_directions,
                                           is_step=use_step_checkbox.value, is_opt_weight=is_opt_weight,
                                           opt_weight=opt_weight, opt_weight_sign=opt_weight_sign), n_trials=num_trials, n_jobs=-1, show_progress_bar=True)

                pr_optimization.controls[1].value = 'Формирование результатов...'
                pr_optimization.update()
                unique_pareto_df = pd.DataFrame()
                if opt_counter > 1:
                    start_time_list_creating = time.time()
                    unique_pareto_list_dict = []
                    for idx_trial, fz_trial in enumerate(study.best_trials):
                        unique_pareto_list_dict.append(fz_trial.params)
                        if idx_trial % 10 == 0:
                            est_percent = min(int((idx_trial / (num_trials/2)) * 100), 100)
                            pr_optimization.controls[1].value = f'Формирование результатов - {est_percent}%'
                            pr_optimization.update()
                    unique_pareto_df = pd.DataFrame(data=unique_pareto_list_dict)
                    del unique_pareto_list_dict
                    unique_pareto_df = unique_pareto_df.drop_duplicates()
                    unique_pareto_df.reset_index(drop=True, inplace=True)

                    stop_time_list_creating = time.time()
                    print(f'Time unique_pareto_df creating: {stop_time_list_creating - start_time_list_creating} sec')

                pr_optimization.controls[1].value = 'Формирование результатов - 100%'
                pr_optimization.update()
                pr_optimization.controls[1].value = 'Обработка результатов...'
                pr_optimization.update()
                opt_criteria_names = [config_opt['criteria']['names'][idx] for idx, val in
                                      enumerate(config_opt['criteria']['opt']) if val is not None]
                if opt_counter > 1:
                    page.criteria_dict = dict(zip(opt_criteria_names, opt_directions))
                    page.criteria_names = opt_criteria_names

                opt_constraint_names = []
                for i_cfg in range(len(config_opt['criteria']['names'])):
                    if config_opt['criteria']['<='][i_cfg] is not None or config_opt['criteria']['>='][i_cfg] is not None:
                        if config_opt['criteria']['names'][i_cfg] not in opt_criteria_names:
                            opt_constraint_names.append(config_opt['criteria']['names'][i_cfg])
                dump_opt_data = dict.fromkeys(config_opt['inputs']['names'] + opt_criteria_names + opt_constraint_names)
                for k_name in dump_opt_data.keys():
                    dump_opt_data[k_name] = []
                add_calc_costs_dict = {}
                res_opt_str = ''
                start_time_total = time.time()
                if opt_counter == 1:
                    res_opt_str += 'Лучшее решение:\n'
                    res_opt_str += '=' * 36 + '\n'
                    res_opt_str += 'Параметры:\n'
                    for k in study.best_trial.params:
                        added_val = np.round(study.best_trial.params[k], 2)
                        try:
                            res_opt_str += f'{k}: {added_val} {page.units_dict[k]}\n'
                        except (TypeError, KeyError):
                            res_opt_str += f'{k}: {added_val}\n'
                        dump_opt_data[k].append(added_val)
                    res_opt_str += '- ' * 36 + '\n'
                    res_opt_str += 'Критерии:\n'
                    criteria_values = []
                    for criteria_name in opt_criteria_names:
                        if criteria_name != 'Стоимость БР':
                            for model_dict in page.model_list:
                                if model_dict['target'] == criteria_name:
                                    input_arr = []
                                    for feat_name in config_opt['inputs']['names']:
                                        norm_val = study.best_trial.params[feat_name] - page.model_metadata[feat_name][
                                            'min']
                                        if page.model_metadata[feat_name]['min'] != page.model_metadata[feat_name][
                                            'max']:
                                            norm_val /= (page.model_metadata[feat_name]['max'] -
                                                         page.model_metadata[feat_name]['min'])
                                        input_arr.append(norm_val)
                                    res = model_dict['model'].predict([input_arr])
                                    criteria_values.append(np.round(res[0], 2))
                                    break
                        else:
                            br_cost = config_opt['base_costs']
                            add_volumes = []
                            for add_name in config_opt['add_cost']['names']:
                                add_volumes.append(study.best_trial.params[add_name])
                            br_cost += get_cost_br(np.array(add_volumes), config_opt['add_cost']['values'])
                            criteria_values.append(np.round(br_cost, 2))
                            add_calc_costs = np.array(add_volumes) * config_opt['add_cost']['values']
                            add_calc_costs_dict = dict(zip(config_opt['add_cost']['names'], add_calc_costs))

                    criteria_temp = dict(zip(opt_criteria_names, criteria_values))
                    for k in criteria_temp:
                        try:
                            res_opt_str += f'{k}: {criteria_temp[k]} {page.units_dict[k]}\n'
                        except (TypeError, KeyError):
                            res_opt_str += f'{k}: {criteria_temp[k]}\n'
                        if k == 'Стоимость БР':
                            res_opt_str += '   Стоимость компонентов:\n'
                            for k_component in calc_costs_dict:
                                res_opt_str += f'   - {k_component}: {np.round(calc_costs_dict[k_component], 2)} руб.\n'
                            for k_component in add_calc_costs_dict:
                                res_opt_str += f'   - {k_component}: {np.round(add_calc_costs_dict[k_component], 2)} руб.\n'

                        dump_opt_data[k].append(criteria_temp[k])
                    res_opt_str += '- ' * 36 + '\n'
                    res_opt_str += 'Ограничения:\n'
                    constraint_values = []
                    for constraint_name in opt_constraint_names:
                        if constraint_name != 'Стоимость БР':
                            for model_dict in page.model_list:
                                if model_dict['target'] == constraint_name:
                                    input_arr = []
                                    for feat_name in config_opt['inputs']['names']:
                                        norm_val = study.best_trial.params[feat_name] - page.model_metadata[feat_name][
                                            'min']
                                        if page.model_metadata[feat_name]['min'] != page.model_metadata[feat_name][
                                            'max']:
                                            norm_val /= (page.model_metadata[feat_name]['max'] -
                                                         page.model_metadata[feat_name]['min'])
                                        input_arr.append(norm_val)
                                    res = model_dict['model'].predict([input_arr])
                                    constraint_values.append(np.round(res[0], 2))
                                    break
                        else:
                            br_cost = config_opt['base_costs']
                            add_volumes = []
                            for add_name in config_opt['add_cost']['names']:
                                add_volumes.append(study.best_trial.params[add_name])
                            br_cost += get_cost_br(np.array(add_volumes), config_opt['add_cost']['values'])
                            constraint_values.append(np.round(br_cost, 2))
                            add_calc_costs = np.array(add_volumes) * config_opt['add_cost']['values']
                            add_calc_costs_dict = dict(zip(config_opt['add_cost']['names'], add_calc_costs))

                    constraint_temp = dict(zip(opt_constraint_names, constraint_values))
                    for k in constraint_temp:
                        try:
                            res_opt_str += f'{k}: {constraint_temp[k]} {page.units_dict[k]}\n'
                        except (TypeError, KeyError):
                            res_opt_str += f'{k}: {constraint_temp[k]}\n'
                        if k == 'Стоимость БР':
                            res_opt_str += '   Стоимость компонентов:\n'
                            for k_component in calc_costs_dict:
                                res_opt_str += f'   - {k_component}: {np.round(calc_costs_dict[k_component], 2)} руб.\n'
                            for k_component in add_calc_costs_dict:
                                res_opt_str += f'   - {k_component}: {np.round(add_calc_costs_dict[k_component], 2)} руб.\n'

                        dump_opt_data[k].append(constraint_temp[k])
                    res_opt_str += '=' * 36 + '\n'
                else:  # MULTI OBJECTIVE OPTIMIZATION RESULT
                    ind_params_names = unique_pareto_df.columns.to_list()
                    res_opt_str += 'Лучшие решения:\n'
                    for idx_row in range(len(unique_pareto_df)):

                        res_opt_str += '=' * 36 + '\n'
                        res_opt_str += 'Параметры:\n'
                        for k in ind_params_names:
                            added_val = np.round(unique_pareto_df.loc[idx_row, k], 2)
                            try:
                                res_opt_str += f'{k}: {added_val} {page.units_dict[k]}\n'
                            except (TypeError, KeyError):
                                res_opt_str += f'{k}: {added_val}\n'
                            dump_opt_data[k].append(added_val)
                        res_opt_str += '- ' * 36 + '\n'
                        res_opt_str += 'Критерии:\n'

                        criteria_values = []
                        for criteria_name in opt_criteria_names:
                            if criteria_name != 'Стоимость БР':
                                for model_dict in page.model_list:
                                    if model_dict['target'] == criteria_name:
                                        input_arr = []
                                        for feat_name in config_opt['inputs']['names']:
                                            norm_val = unique_pareto_df.loc[idx_row, feat_name] - page.model_metadata[feat_name][
                                                'min']
                                            if page.model_metadata[feat_name]['min'] != page.model_metadata[feat_name][
                                                'max']:
                                                norm_val /= (page.model_metadata[feat_name]['max'] -
                                                             page.model_metadata[feat_name]['min'])
                                            input_arr.append(norm_val)
                                        res = model_dict['model'].predict([input_arr])
                                        criteria_values.append(np.round(res[0], 2))
                                        break
                            else:
                                br_cost = config_opt['base_costs']
                                add_volumes = []
                                for add_name in config_opt['add_cost']['names']:
                                    add_volumes.append(unique_pareto_df.loc[idx_row, add_name])
                                br_cost += get_cost_br(np.array(add_volumes), config_opt['add_cost']['values'])
                                criteria_values.append(np.round(br_cost, 2))
                                add_calc_costs = np.array(add_volumes) * config_opt['add_cost']['values']
                                add_calc_costs_dict = dict(zip(config_opt['add_cost']['names'], add_calc_costs))

                        criteria_temp = dict(zip(opt_criteria_names, criteria_values))
                        for k in criteria_temp:
                            try:
                                res_opt_str += f'{k}: {criteria_temp[k]} {page.units_dict[k]}\n'
                            except (TypeError, KeyError):
                                res_opt_str += f'{k}: {criteria_temp[k]}\n'
                            if k == 'Стоимость БР':
                                res_opt_str += '   Стоимость компонентов:\n'
                                for k_component in calc_costs_dict:
                                    res_opt_str += f'   - {k_component}: {np.round(calc_costs_dict[k_component], 2)} руб.\n'
                                for k_component in add_calc_costs_dict:
                                    res_opt_str += f'   - {k_component}: {np.round(add_calc_costs_dict[k_component], 2)} руб.\n'

                            dump_opt_data[k].append(criteria_temp[k])

                        res_opt_str += '- ' * 36 + '\n'
                        res_opt_str += 'Ограничения:\n'
                        constraint_values = []
                        for constraint_name in opt_constraint_names:
                            if constraint_name != 'Стоимость БР':
                                for model_dict in page.model_list:
                                    if model_dict['target'] == constraint_name:
                                        input_arr = []
                                        for feat_name in config_opt['inputs']['names']:
                                            norm_val = unique_pareto_df.loc[idx_row, feat_name] - page.model_metadata[feat_name][
                                                'min']
                                            if page.model_metadata[feat_name]['min'] != page.model_metadata[feat_name][
                                                'max']:
                                                norm_val /= (page.model_metadata[feat_name]['max'] -
                                                             page.model_metadata[feat_name]['min'])
                                            input_arr.append(norm_val)
                                        res = model_dict['model'].predict([input_arr])
                                        constraint_values.append(np.round(res[0], 2))
                                        break
                            else:
                                br_cost = config_opt['base_costs']
                                add_volumes = []
                                for add_name in config_opt['add_cost']['names']:
                                    add_volumes.append(unique_pareto_df.loc[idx_row, add_name])
                                br_cost += get_cost_br(np.array(add_volumes), config_opt['add_cost']['values'])
                                constraint_values.append(np.round(br_cost, 2))
                                add_calc_costs = np.array(add_volumes) * config_opt['add_cost']['values']
                                add_calc_costs_dict = dict(zip(config_opt['add_cost']['names'], add_calc_costs))

                        constraint_temp = dict(zip(opt_constraint_names, constraint_values))
                        for k in constraint_temp:
                            try:
                                res_opt_str += f'{k}: {constraint_temp[k]} {page.units_dict[k]}\n'
                            except (TypeError, KeyError):
                                res_opt_str += f'{k}: {constraint_temp[k]}\n'
                            if k == 'Стоимость БР':
                                res_opt_str += '   Стоимость компонентов:\n'
                                for k_component in calc_costs_dict:
                                    res_opt_str += f'   - {k_component}: {np.round(calc_costs_dict[k_component], 2)} руб.\n'
                                for k_component in add_calc_costs_dict:
                                    res_opt_str += f'   - {k_component}: {np.round(add_calc_costs_dict[k_component], 2)} руб.\n'

                            dump_opt_data[k].append(constraint_temp[k])

                        res_opt_str += '=' * 36 + '\n'
                        res_opt_str += '\n'
                    stop_time_total = time.time()
                    print(f"Total time processing Pareto front's: {stop_time_total - start_time_total} sec")

                page.dump_opt_data = dump_opt_data
                page.res_opt_str = res_opt_str
                show_text_result_opt_button.disabled = False
                show_text_result_opt_button.update()
                save_text_result_opt_button.disabled = False
                save_text_result_opt_button.update()
                if page.is_multi_obj:
                    pareto_x_axis_dd.options = criteria_axis_generator(e)
                    pareto_x_axis_dd.update()
                    pareto_y_axis_dd.options = criteria_axis_generator(e)
                    pareto_y_axis_dd.update()
                    pareto_chart_field.visible = True
                    pareto_chart_field.update()
        except ValueError:
            pass

        pr_optimization.visible = False
        pr_optimization.update()
        opt_start_button.disabled = False
        opt_start_button.update()
        gc.collect()

    opt_start_button = ft.ElevatedButton(text='Оптимизировать',
                                         icon=ft.icons.AUTO_MODE,
                                         icon_color=ft.colors.ORANGE_600,
                                         scale=0.85,
                                         on_click=start_opt_func,
                                         style=ft.ButtonStyle(bgcolor={
                                             ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                         }),
                                         )

    opt_start_container = ft.Container(content=opt_start_button,
                                       height=30)

    algorithm_opt_selector = ft.Row(
        [
            opt_algorithm_container,
            opt_options_container,
            opt_start_container,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        visible=False,
    )

    pr_optimization = ft.Row(
        [
            ft.ProgressRing(width=16, height=16, stroke_width=2, color=ft.colors.ORANGE_900),
            ft.Text(color=ft.colors.BLACK, text_align=ft.MainAxisAlignment.CENTER, value="Оптимизация...",
                    scale=0.85)
        ],
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    # SHOW OPT RESULT
    def open_text_result_opt_ad(e):
        text_result_opt_ad.content = ft.Text(page.res_opt_str)
        page.dialog = text_result_opt_ad
        text_result_opt_ad.open = True
        page.update()

    text_result_opt_ad = ft.AlertDialog(modal=False,
                                        adaptive=True,
                                        content=None,
                                        scrollable=True)

    show_text_result_opt_button = ft.ElevatedButton('Посмотреть результаты оптимизации',
                                                    icon=ft.icons.VISIBILITY,
                                                    scale=0.85,
                                                    on_click=open_text_result_opt_ad,
                                                    icon_color=ft.colors.ORANGE_500,
                                                    style=ft.ButtonStyle(bgcolor={
                                                        ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                                    }),
                                                    disabled=True,
                                                    )

    # SAVE OPT RESULT

    def submit_save_txt_file(e):
        if text_result_opt_saving_field.value != '':
            if text_result_opt_saving_field.value.endswith('.xlsx'):
                text_result_opt_saving_field.value = text_result_opt_saving_field.value[:-5]
            temp_df = pd.DataFrame(data=page.dump_opt_data)
            temp_df.to_excel(os.path.join('opt_results',
                                          text_result_opt_saving_field.value + text_result_opt_saving_field.suffix_text),
                             encoding='utf-8')
            dialog_save_text_result_opt.open = False
            dialog_save_text_result_opt.update()

    def open_dialog_save_text_result_opt(e):
        page.dialog = dialog_save_text_result_opt
        dialog_save_text_result_opt.open = True
        page.update()

    text_result_opt_saving_field = ft.TextField(label='Имя_файла.xlsx', width=300, height=70,
                                                text_align=ft.alignment.center,
                                                multiline=False,
                                                suffix_text='.xlsx'
                                                )

    dialog_save_text_result_opt = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=text_result_opt_saving_field,
        actions=[
            ft.TextButton('Submit', on_click=submit_save_txt_file)
        ],
        actions_alignment=ft.MainAxisAlignment.CENTER,
    )

    save_text_result_opt_button = ft.ElevatedButton('Сохранить результаты оптимизации',
                                                    icon=ft.icons.SAVE_ALT,
                                                    scale=0.85,
                                                    on_click=open_dialog_save_text_result_opt,
                                                    icon_color=ft.colors.ORANGE_500,
                                                    style=ft.ButtonStyle(bgcolor={
                                                        ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                                    }),
                                                    disabled=True,
                                                    )

    # PARETO CHART

    def pareto_chart_generator(e):
        df_chart = pd.DataFrame(data=page.dump_opt_data)
        df_chart = df_chart[[pareto_x_axis_dd.value, pareto_y_axis_dd.value]]
        fig, ax = plt.subplots(figsize=(9, 9))
        if len(page.criteria_names) > 2:
            df_chart.sort_values(axis=0, by=[pareto_x_axis_dd.value], inplace=True)
            df_chart.reset_index(drop=True, inplace=True)
            stat_df_chart = df_chart.copy()
            idx_list = []
            if page.criteria_dict[pareto_x_axis_dd.value] == 'maximize' and page.criteria_dict[pareto_y_axis_dd.value] == 'maximize':
                crit_idx = df_chart.shape[0]-1
                while df_chart.shape[0] > 0:
                    curr_idx = df_chart[pareto_y_axis_dd.value].idxmax()
                    idx_list.append(curr_idx)
                    if curr_idx+1 <= crit_idx:
                        df_chart = df_chart.loc[curr_idx+1:, :]
                    else:
                        break
            elif page.criteria_dict[pareto_x_axis_dd.value] == 'maximize' and page.criteria_dict[pareto_y_axis_dd.value] == 'minimize':
                crit_idx = df_chart.shape[0] - 1
                while df_chart.shape[0] > 0:
                    curr_idx = df_chart[pareto_y_axis_dd.value].idxmin()
                    idx_list.append(curr_idx)
                    if curr_idx + 1 <= crit_idx:
                        df_chart = df_chart.loc[curr_idx + 1:, :]
                    else:
                        break
            elif page.criteria_dict[pareto_x_axis_dd.value] == 'minimize' and page.criteria_dict[pareto_y_axis_dd.value] == 'maximize':
                crit_idx = 0
                while df_chart.shape[0] > 0:
                    curr_idx = df_chart[pareto_y_axis_dd.value].idxmax()
                    idx_list.append(curr_idx)
                    if curr_idx - 1 >= crit_idx:
                        df_chart = df_chart.loc[:curr_idx - 1, :]
                    else:
                        break
            elif page.criteria_dict[pareto_x_axis_dd.value] == 'minimize' and page.criteria_dict[pareto_y_axis_dd.value] == 'minimize':
                crit_idx = 0
                while df_chart.shape[0] > 0:
                    curr_idx = df_chart[pareto_y_axis_dd.value].idxmin()
                    idx_list.append(curr_idx)
                    if curr_idx - 1 >= crit_idx:
                        df_chart = df_chart.loc[:curr_idx - 1, :]
                    else:
                        break
            else:
                raise Exception(f'Unexpected direction of optimization')
            df_chart = stat_df_chart.loc[idx_list, :]

        ax.scatter(df_chart[pareto_x_axis_dd.value], df_chart[pareto_y_axis_dd.value], color='xkcd:dusty orange', s=55, edgecolor='xkcd:chocolate brown', linewidth=2)
        ax.set_xlabel(pareto_x_axis_dd.value)
        ax.set_ylabel(pareto_y_axis_dd.value)
        ax.set_title(f'Парето фронт')
        ax.grid(True)
        plt.tight_layout()

        pareto_charts = [MatplotlibChart(fig, original_size=False, transparent=False)]
        plt.close()

        return pareto_charts

    def open_charts_ml_predict_ad(e):
        try:
            if pareto_x_axis_dd.value is not None and pareto_y_axis_dd.value is not None:
                if pareto_x_axis_dd.value != pareto_y_axis_dd.value:
                    show_pareto_chart_ad.content.controls = pareto_chart_generator(e)
                    page.dialog = show_pareto_chart_ad
                    show_pareto_chart_ad.open = True
                    page.update()
        except Exception as ex:
            print(f'{ex}')
            pass

    show_pareto_chart_ad = ft.AlertDialog(
        modal=False,
        adaptive=True,
        content=ft.Column(scroll='adaptive'),
    )

    show_pareto_chart_button = ft.ElevatedButton('Построить график',
                                                 icon=ft.icons.BRUSH,
                                                 scale=0.85,
                                                 on_click=open_charts_ml_predict_ad,
                                                 icon_color=ft.colors.ORANGE_500,
                                                 style=ft.ButtonStyle(bgcolor={
                                                     ft.MaterialState.DISABLED: ft.colors.GREY_900,
                                                 }),
                                                 disabled=False,
                                                 )

    def criteria_axis_generator(e):
        dd_options = []
        for criteria in page.criteria_names:
            dd_options.append(ft.dropdown.Option(criteria))
        return dd_options

    pareto_y_axis_dd = ft.Dropdown(
        label='Критерий по оси Y',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выберите критерий',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=None,
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=None,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.75,
        width=350,
        focused_color=None,
    )

    pareto_x_axis_dd = ft.Dropdown(
        label='Критерий по оси X',
        label_style=ft.TextStyle(color=ft.colors.BLACK),
        hint_style=ft.TextStyle(color=ft.colors.BLUE_900),
        hint_text='Выберите критерий',
        text_style=ft.TextStyle(color=ft.colors.BLUE_900),
        options=None,
        autofocus=False,
        border_color=ft.colors.ORANGE_600,
        focused_border_color=ft.colors.ORANGE_800,
        border_width=2,
        focused_border_width=2,
        bgcolor=ft.colors.GREY_400,
        focused_bgcolor=None,
        fill_color=ft.colors.GREY_400,
        filled=True,
        scale=0.75,
        width=350,
        focused_color=None,
    )

    pareto_chart_field = ft.Column(
        [
            ft.Row([
                ft.Text('Визуализация Парето-оптимального множества', weight=ft.FontWeight.BOLD,
                        color=ft.colors.BLUE_GREY_800, scale=0.9)
            ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row([
                pareto_x_axis_dd,
                pareto_y_axis_dd,
                show_pareto_chart_button,
            ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        ],
        visible=False,
    )

    # OPTIMIZATION WINDOW
    opt_field = ft.Column(
        [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                select_saved_model_button,
                                txt_select_model,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=0,
                        ),
                        component_br_button_field,
                        component_br_txt_field,
                        show_hide_opt_inputs_field,
                        params_criteria_field,
                        algorithm_opt_selector,
                        pr_optimization,
                        ft.Row(
                            [
                                show_text_result_opt_button,
                                save_text_result_opt_button,
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        pareto_chart_field,
                    ],
                    scroll=ft.ScrollMode.ALWAYS
                ),
                height=page.window_height - my_app_bar_h,
                alignment=ft.alignment.top_center,
                padding=ft.padding.only(top=my_app_bar_h),

            ),
        ],
        visible=False,
    )

    # --- page structure ---

    page.add(
        ft.Column([
            main_buttons,
            exp_field,
            upload_obj,
            ml,
            opt_field,
            about_api
        ],
            alignment=ft.MainAxisAlignment.CENTER
        )
    )


# ft.app(main)
# ft.app(port=8550, target=main)


# if __name__ == '__main__':
os.environ['FLET_SECRET_KEY'] = '19'
# flet_path = os.getenv("FLET_PATH", DEFAULT_FLET_PATH)
# flet_port = int(os.getenv("FLET_PORT", DEFAULT_FLET_PORT))
ft.app(target=main, view=ft.WEB_BROWSER, upload_dir="uploads",
       assets_dir='assets')  # view=ft.WEB_BROWSER name=flet_path , port=flet_port
