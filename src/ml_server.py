import os
import pickle
import pandas as pd
import numpy as np
import sklearn
import wtforms.validators
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from ensembles import MyRandomForest, MyGradBoost

import datetime
from pathlib import Path
import numpy as np

import plotly
import plotly.subplots
import plotly.graph_objects as go
from shapely.geometry.polygon import Point
from shapely.geometry.polygon import Polygon

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, ValidationError, NumberRange
from wtforms import StringField, SubmitField, FileField, BooleanField, URLField, FloatField, DecimalField

from utils import polygon_random_point

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
messages = []
lines = []
method = 'DEFAULT'
data_csv = None
n_estimators, k_frac, max_depth, learning_rate = 0, 0, 0, 0
data_address = None


class Message:
    header = ''
    text = ''


class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    submit = SubmitField('Get Result')


class Response(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    sentiment = StringField('Sentiment', validators=[DataRequired()])
    submit = SubmitField('Try Again')


class FileForm(FlaskForm):
    file_path = FileField('Загрузить файл', validators=[
        DataRequired('Выберите файл'),
        FileAllowed(['csv'], 'формат CSV!')
    ])
    submit = SubmitField('Получить предсказания')


class RFConstructionForm(FlaskForm):
    n_estimators = DecimalField('n_estimators, количество базовых моделей', validators=[NumberRange(1, 9999)])
    k_frac = FloatField('k_frac, доля используемых признаков в каждой базовой модели', validators=[NumberRange(0.01, 1)])
    max_depth = DecimalField('max_depth, максимальная глубина каждой из базовых моделей (0 означает отсутствие ограничений)', validators=[NumberRange(0)])
    submit = SubmitField('Обучить модель')
    file_path = FileField('Вы можете дополнительно загрузить датасет для обучения модели, иначе будет использоваться стандартный', validators=[
        FileAllowed(['csv'], 'формат CSV!')
    ])


class GBConstructionForm(RFConstructionForm):
    learning_rate = FloatField('learning_rate, показатель скорости обучения градиентного бустинга', validators=[NumberRange(0.01, 1)])
    submit = SubmitField('Обучить модель')
    file_path = FileField(
        'Вы можете дополнительно загрузить датасет для обучения модели, иначе будет использоваться стандартный',
        validators=[
            FileAllowed(['csv'], 'формат CSV!')
        ])


def rmse(X, y):
    return mse(X, y) ** 0.5


def score_model(name):
    try:
        path = Path("").absolute()
        path = str(path.parent) + "/data/" + name
        model = pickle.load(open(path, "rb"))
        global data_csv
        data_csv['date'] = data_csv['date'].values.astype(str)
        data_csv['year'] = data_csv['date'].str.slice(stop=4).astype(int)
        data_csv['month'] = data_csv['date'].str.slice(4, 6).astype(int)
        data_csv['day'] = data_csv['date'].str.slice(6, 8).astype(int)
        if 'price' in data_csv:
            y = np.array(data_csv['price'])
            X = data_csv.drop(['price', 'date'], axis=1)
        else:
            y = -1
            X = data_csv.drop(['date'], axis=1)
        X.index = np.arange(X.shape[0])
        X.columns = np.arange(X.shape[1])
        pred = model.predict(X)
        filename = 'results.sav'
        pickle.dump(pred, open(filename, 'wb'))
        if type(y) != int:
            score = 'RMSE = ' + str(rmse(y, pred))
        else:
            score = 'Predicted successfully'
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        score = 'Exception: {0}'.format(exc)

    return score


def train_custom_score():
    try:
        global data_csv
        print(data_csv.head())
        data_csv['date'] = data_csv['date'].values.astype(str)
        data_csv['year'] = data_csv['date'].str.slice(stop=4).astype(int)
        data_csv['month'] = data_csv['date'].str.slice(4, 6).astype(int)
        data_csv['day'] = data_csv['date'].str.slice(6, 8).astype(int)
        y = np.array(data_csv['price'])
        X = data_csv.drop(['price', 'date'], axis=1)
        X.index = np.arange(X.shape[0])
        X.columns = np.arange(X.shape[1])
        global method, n_estimators, k_frac, max_depth, learning_rate
        if method == 'Градиентный бустинг':
            if max_depth != 0:
                model = MyGradBoost(n_estimators, k_frac, max_depth, learning_rate)
            else:
                model = MyGradBoost(n_estimators, k_frac, None, learning_rate)
        else:
            if max_depth != 0:
                model = MyRandomForest(n_estimators, k_frac, max_depth)
            else:
                model = MyRandomForest(n_estimators, k_frac, None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = 'RMSE = ' + str(rmse(y_test, pred))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        score = 'Exception: {0}'.format(exc)
    return score


@app.route('/random_forest', methods=['GET', 'POST'])
def random_forest():
    global method
    method = 'Случайный лес'
    file_form = FileForm()

    if request.method == 'POST' and file_form.validate_on_submit():
        global data_csv
        data_csv = pd.read_csv(file_form.file_path.data.stream)
        return redirect(url_for('random_forest_res'))

    return render_template('method_usage.html', form=file_form)


@app.route('/random_forest_res')
def random_forest_res():
    score = score_model('random_forest_best.sav')
    return render_template('rf_results.html', score=score)


@app.route('/grad_boost', methods=['GET', 'POST'])
def grad_boost():
    global method
    method = 'Градиентный бустинг'
    file_form = FileForm()

    if request.method == 'POST' and file_form.validate_on_submit():
        global data_csv
        data_csv = pd.read_csv(file_form.file_path.data.stream)
        return redirect(url_for('grad_boost_res'))

    return render_template('method_usage.html', form=file_form)


@app.route('/grad_boost_res')
def grad_boost_res():
    score = score_model('grad_boost_best.sav')
    return render_template('rf_results.html', score=score)


@app.route('/my_model_rf')
@app.route('/my_model_rf', methods=['GET', 'POST'])
def my_model_rf():
    const_form = RFConstructionForm()

    if request.method == 'POST' and const_form.validate_on_submit():
        global method, n_estimators, k_frac, max_depth, data_csv
        method = 'Случайный лес'
        n_estimators = int(const_form.n_estimators.data)
        k_frac = float(const_form.k_frac.data)
        max_depth = int(const_form.max_depth.data)
        fileName = const_form.file_path.data
        print(fileName)
        if const_form.file_path.data == 'application/octet-stream':  # use basic dataset
            data_csv = pd.read_csv('https://raw.githubusercontent.com/GooseIt/WebForestBoost/data/kc_house_data.csv',
                               on_bad_lines='skip')
        else:
            data_csv = pd.read_csv(const_form.file_path.data.stream, on_bad_lines='skip')
        return redirect(url_for('my_model_learning'))

    return render_template('my_model_interactive.html', form=const_form)


@app.route('/my_model_gb')
@app.route('/my_model_gb', methods=['GET', 'POST'])
def my_model_gb():
    const_form = GBConstructionForm()

    if request.method == 'POST' and const_form.validate_on_submit():
        global method, n_estimators, k_frac, max_depth, learning_rate, data_csv
        method = 'Градиентный бустинг'
        n_estimators = int(const_form.n_estimators.data)
        k_frac = float(const_form.k_frac.data)
        max_depth = int(const_form.max_depth.data)
        learning_rate = float(const_form.learning_rate.data)
        if const_form.file_path.data is None:  # use basic dataset
            data_csv = pd.read_csv('https://raw.githubusercontent.com/GooseIt/WebForestBoost/data/kc_house_data.csv',
                               on_bad_lines='skip')
        else:
            data_csv = pd.read_csv(const_form.file_path.data.stream, on_bad_lines='skip')
        return redirect(url_for('my_model_learning'))

    return render_template('my_model_interactive.html', form=const_form)


@app.route('/my_model_learning')
def my_model_learning():
    score = train_custom_score()
    print(score)
    return render_template('my_model_learning.html', form=FlaskForm(), score=score)


@app.route('/pick_your_fighter')
def pick_your_fighter():
    return render_template('pick_your_fighter.html')


@app.route('/')
@app.route('/index')
def index():
    return render_template('starting_page.html')


# plotly visualization of training process
@app.route('/dashboard', methods=['GET', 'POST'])
def get_dashboard():
    np.random.seed(42)

    current_parameter = request.values.get('parameter', 'oxygen')
    start_datetime = request.values.get('start_time', '2018-09-22T08:54')
    end_datetime = request.values.get('end_time', '2021-11-22T09:02')
    default_start_time, default_end_time = start_datetime, end_datetime

    start_date, start_time = start_datetime.split('T')
    end_date, end_time = end_datetime.split('T')
    start_ts = int(
        datetime.datetime(
            *[int(_) for _ in start_date.split('-')], *[int(_) for _ in start_time.split(':')], 00
        ).timestamp()
    )
    end_ts = int(
        datetime.datetime(
            *[int(_) for _ in end_date.split('-')],
            *[int(_) for _ in end_time.split(':')], 00
        ).timestamp()
    )

    parameter_names = {
        'oxygen': ('кислород', 'кислорода'),
        'humidity': ('влажность', 'влажности'),
        'methane': ('метан', 'метана'),
    }[current_parameter]

    fig = plotly.subplots.make_subplots(
        rows=2, cols=2, specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        subplot_titles=(
            f"Показания {parameter_names[1]} в различных зонах шахты", f"Средний уровень {parameter_names[1]}",
            None, "Движение породы на шахте"
        )
    )

    fences = ['Зона работ', 'Зона движения пластов', 'Зона трещин']
    for fence_idx, fence_name in enumerate(fences):
        zone_dots = np.random.normal(loc=[-1, 0, 1][fence_idx], scale=0.7, size=(3, 2))
        zone_polygon = Polygon(zone_dots)

        # Generate random dots in polygon
        x, y, z, time = [], [], [], []
        for idx in range(100):
            new_point = polygon_random_point(zone_dots)
            if zone_polygon.contains(Point(new_point)):
                if start_ts <= 1537318787 + 1000000 * idx <= end_ts:
                    x.append(new_point[0])
                    y.append(new_point[1])
                    z.append(new_point[0] / 100 + new_point[1] / 100)
                    time.append(datetime.date.fromtimestamp(1537318787 + 1000000 * idx).strftime('%Y-%m-%d'))

        colorbar = plotly.graph_objs.scatter.marker.ColorBar(x=0.55, thickness=20, title='%')

        fig.add_trace(
            go.Scatter(
                x=zone_dots[:, 0], y=zone_dots[:, 1], fill="toself",
                fillcolor='#%02x%02x%02x' % (30, 30, 30), showlegend=False, opacity=0.2,
                name=fence_name, hovertemplate=f'{fence_name}', hoverinfo='skip'
            ), row=1, col=1
        )
        fig.add_scatter(
            x=x, y=y, marker={
                'color': z,
                'colorbar': colorbar if fence_idx == 0 else None,
            }, showlegend=False, mode='markers',
            customdata=z, hovertemplate=' '.join([f'{parameter_names[0]}:', '%{customdata:.3f}%'])
        )

        fig.add_trace(
            go.Scatter(
                x=time, y=z, name=fence_name
            ), row=1, col=2,
        )

    mine_time, move_1, move_2, move_3, move_4 = [], [], [], [], []
    for idx in range(100):
        if start_ts <= 1537318787 + 1000000 * idx <= end_ts:
            move_1.append(np.random.randint(0, 2) > 0)
            move_2.append(np.random.randint(0, 2) > 0)
            move_3.append(np.random.randint(0, 2) > 0)
            move_4.append(np.random.randint(0, 2) > 0)
            mine_time.append(datetime.date.fromtimestamp(1537318787 + 1000000 * idx).strftime('%Y-%m-%d'))

    for idx, move_data in enumerate([move_1, move_2, move_3, move_4]):
        fig.add_trace(
            go.Scatter(
                x=mine_time, y=move_data, showlegend=True, name=f'Movement {idx + 1}'
            ), row=2, col=2
        )
    fig.add_annotation(
        go.layout.Annotation(
            text='Движение породы на шахте',
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.93,
            y=0.40,
            bordercolor='black',
            borderwidth=0
        ), font={'size': 16}
    )

    fig.update_yaxes(row=1, col=1, autorange="reversed")
    fig.update_layout(
        hovermode='closest',
        title_text='',
        title_x=0.5, width=1500, height=700
    )

    return render_template(
        'dashboard.html',
        dashboard_div=fig.to_html(full_html=False)
    )
