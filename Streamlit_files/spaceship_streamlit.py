import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from math import ceil, floor
from wordcloud import WordCloud
from pycaret.classification import *

from sklearn import preprocessing, impute
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.metrics import confusion_matrix
# from sklearn import tree, svm
# from sklearn.model_selection import cross_val_score, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Building functions:


@st.cache_data
def show_hist(df, column):
    figg = px.histogram(df, column)
    figg.update_layout(xaxis_title=column, yaxis_title="Frequency", font={'size': 14, 'color': 'black'},
                       template='plotly_dark', plot_bgcolor='white', bargap=0.1)
    figg.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    figg.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    st.plotly_chart(figg, theme=None, use_container_width=True)


@st.cache_data
def show_heatmap(df):
    df_heatmap = df.copy()
    df_heatmap[y_name] = df_heatmap[y_name].astype(int)
    df_heatmap2 = df_heatmap.drop(columns=list(df_heatmap.select_dtypes('object')), axis=1)
    dum = pd.get_dummies(df_heatmap2, dtype=int)
    fiig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(dum.corr(numeric_only=True), fmt=".3f", annot=True, cmap='coolwarm', linewidth=.5, cbar=False)
    ax.tick_params(axis='both', labelsize=12, grid_color='b')
    st.pyplot(fiig)


def make_filter(df):

    options = st.multiselect('Select which variables to filter by:', list(df.columns))
    st.caption('*If no variable is selected, the method will return all rows of the dataset')
    df_filter = df.copy()

    if options:
        for opt in options:
            ft = df_filter[opt]

            if ft.dtype == 'category':
                filt = st.radio(f'Select {opt}:', list(ft.unique()))
                df_filter = df_filter.loc[(ft == filt)]

            elif pd.api.types.is_float_dtype(ft):
                step = (ceil(ft.max()) - floor(ft.min()))/100
                filt = st.slider(f'Select {opt}:', ft.min(), ft.max(), (ft.min(), ft.mean()+1), step)
                df_filter = df_filter[ft.between(filt[0], filt[1])]

            elif pd.api.types.is_integer_dtype(ft):
                filt = st.slider(f'Select {opt}:', ft.min(), ft.max(), (ft.min(), ceil(ft.mean()) + 1), 1)
                df_filter = df_filter[ft.between(filt[0], filt[1])]

            else:
                filt = st.text_input(f'Input {opt}', key='texto_'+opt)
                df_filter = df_filter.loc[ft.str.contains(str(filt), case=False, na=False)]

    st.write(f"{len(df_filter)} result(s) found.")

    return df_filter


def set_dtypes(df, level=5):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= level:
            df[col] = df[col].astype('category')

        if pd.api.types.is_object_dtype(df[col]) and df[col].nunique() <= level:
            df[col] = df[col].astype('category')

    return df


@st.cache_data
def load_data():
    train = pd.read_csv(
        r'https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/Streamlit_files/spaceship_train.csv')
    test = pd.read_csv(
        r'https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/Streamlit_files/spaceship_test.csv')

    return train, test

# ----------------------------------------------------------------------------------------------------------------------
# Page title, datasets download, and defining types of variables:
# Everything inside this part is specific from the dataset itself!!!


def label_def():
    feats = ['A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the '
             'passenger is travelling with and pp is their number within the group. People in a group are often '
             'family members, but not always.',
             'The planet the passenger departed from, typically their planet of permanent residence.',
             'Indicates whether the passenger elected to be put into suspended animation for the duration of the '
             'voyage. Passengers in cryosleep are confined to their cabins.',
             'The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either '
             'P for Port or S for Starboard.',
             'The planet the passenger will be debarking to.',
             'The age of the passenger.',
             'Whether the passenger has paid for special VIP service during the voyage.',
             'Amount the passenger has billed at this Spaceship Titanic luxury amenity.',
             'Amount the passenger has billed at this Spaceship Titanic luxury amenity.',
             'Amount the passenger has billed at this Spaceship Titanic luxury amenity.',
             'Amount the passenger has billed at this Spaceship Titanic luxury amenity.',
             'Amount the passenger has billed at this Spaceship Titanic luxury amenity.',
             'The first and last names of the passenger.',
             'Whether the passenger was transported to another dimension. This is the target, the column you are '
             'trying to predict.']

    return feats


def func_feat(x, x_ref):
    dict_feat = {'AgeInterval': {'message_box': "Add 'AgeInterval' categorical feature: Age data is split into 4 bins",
                                 'method': pd.qcut(x['Age'], [0, 0.25, 0.5, 0.75, 1], duplicates='drop')},

                 'TotalBilled': {'message_box': "Add 'TotalBilled' numerical feature: sum of all local amenities",
                                 'method': x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']},

                 'Deck': {'message_box': "Add 'Deck' categorical feature: Deck of the passengers cabin",
                          'method': x_ref['Cabin'].str[0].astype('category')},
                 #
                 # 'FmSize': {'message_box': "Add 'FmSize' numerical feature: sum of 'SibSp' and 'Parch'",
                 #            'method': x['SibSp'] + x['Parch'] + 1},
                 #
                 # 'Title': {'message_box': "Add 'Title' categorical feature: title of each passenger name",
                 #           'method': x_ref['Name'].apply(lambda w: w.split('. ')[0].split(', ')[1]).astype('category')},

                 # 'IsAlone': {'message_box': "Add 'IsAlone' categorical feature: binary level to indicate whether the "
                 #                            "passenger was travelling alone or not",
                 #             'method': ((x['Parch'] == 0) & (x['SibSp'] == 0)).astype('category')},
                 #
                 'Side': {'message_box': "Add 'Side' categorical feature: side of the passengers cabin",
                              'method': x_ref['Cabin'].str[-1].astype('category')}
                 #
                 # 'MultiTicket': {'message_box': "Add 'MultiTicket' numerical feature: number of people to hold the "
                 #                                "same ticket number including the passenger himself",
                 #                 'method': x_ref['Ticket'].map(x_ref['Ticket'].value_counts()).astype('category')}
                 }

    return dict_feat


def introduction():
    st.title(':passenger_ship: Spaceship Titanic - Predict which passengers are transported to an alternate dimension '
             '(kaggle competition)', anchor='titanic_train')

    st.write('''
    Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've 
    received a transmission from four light-years away and things aren't looking good.

    The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on 
    board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly 
    habitable exoplanets orbiting nearby stars.

    While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship 
    Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its 
    namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to 
    an alternate dimension!
    
    To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were 
    transported by the anomaly using records recovered from the spaceship’s damaged computer system.
    ''')

    st.header('Training dataset visualization', help='Both datasets are available for download on [kaggle]('
                                                     'https://www.kaggle.com/competitions/spaceship-titanic).')

    st.write('''
    To get started, the datasets were loaded from my github account:
    ''')


p_title = 'Spaceship challenge'
st.set_page_config(page_title=p_title, page_icon=":space_invader:", layout='wide')

introduction()

# ----------------------------------------------------------------------------------------------------------------------

with st.echo('above'):
    data_load_state = st.text('Loading data...')
    df_train, df_test = load_data()
    y_name = 'Transported'
    y = df_train[y_name]

data_load_state.text("Datasets successfully loaded! (using st.cache_data)")

df_train = set_dtypes(df_train)
df_test = set_dtypes(df_test)

# ----------------------------------------------------------------------------------------------------------------------
# Filter data function is in the sidebar (left-hand side of the page): results are shown in the tab 'filtered data'

st.write('''
The filter function can be accessed by the sidebar on the left side of the page. The filtered data is shown below:
''')

with st.sidebar:
    st.subheader('Filter train dataset')
    df_filtered = make_filter(df_train)

# ----------------------------------------------------------------------------------------------------------------------

# Visualize the titanic dataset:
st.write('**Filtered train dataset**')
st.dataframe(df_filtered, hide_index=True, use_container_width=True)

# Expanded tab to show explanation of the variables

st.write('Here is a brief explanation about the features:')
with st.expander("See features note"):
    features = label_def()
    df_exp = pd.DataFrame({'Variable': df_train.columns, 'Definition': features})
    st.dataframe(df_exp, hide_index=True, use_container_width=True)

st.divider()
# ----------------------------------------------------------------------------------------------------------------------

# Overview of the dataset


@st.cache_data(experimental_allow_widgets=True)
def show_overview(df):

    st.subheader('Overview')

    col1, col2 = st.columns([1, 1.5], gap="small")
    with col1:
        st.write('**Train dataset statistics**')

        st.markdown(f"- Number of observations: {len(df)}")
        st.markdown(f"- Number of features: {len(df.columns)}")
        st.markdown(f"- Number of missing cells: {df.isnull().sum().sum()}")
        st.markdown("- Missing cells (%): {:.2%}".format(df.isnull().sum().sum()/df.size))
        st.markdown(f"- Number of duplicated rows: {df.duplicated().sum()}")
        st.markdown("- Total size in memory: {:.2f} KB".format(df.memory_usage(deep=True).sum()/1024))

    with col2:
        st.write('**Type of features**')

        df_type = pd.concat([pd.Series(df.select_dtypes('number').columns),
                             pd.Series(df.select_dtypes('category').columns),
                             pd.Series(df.select_dtypes('object').columns)], axis=1)
        df_type = df_type.set_axis(['Numerical features', 'Categorical features', 'Text features'], axis='columns')
        st.dataframe(df_type, hide_index=True, use_container_width=True)


show_overview(df_train)

st.divider()
# ----------------------------------------------------------------------------------------------------------------------

# Variable information


def show_var_info(df):
    st.subheader('Features information')
    name_num = st.tabs(list(df.select_dtypes('number').columns))

    for idx, tabb in enumerate(name_num):
        with tabb:
            coluna = df.select_dtypes('number').columns[idx]

            col1a, col2a, col3a = st.columns([1.3, 1, 2.5], gap="medium")

            with col1a:
                st.write('**Basic information**')

                df_n1 = pd.DataFrame(data=[df.select_dtypes('number')[coluna].nunique(),
                                           df.select_dtypes('number')[coluna].isnull().sum()],
                                     index=['# of distinct values', '# of missing cells'], columns=[' '])

                df_n2 = pd.DataFrame(list(df.select_dtypes('number')[coluna].describe()),
                                     index=['# of non-null cells', 'Mean value', 'Std deviation', 'Minimum',
                                            '1st quartile', 'Median', '3rd quartile', 'Maximum'],
                                     columns=[' '])
                df_stat = pd.concat([df_n1, df_n2], axis=0)
                st.dataframe(df_stat, use_container_width=True)

            with col2a:
                st.write('**Most frequent values**')
                df_col4 = pd.DataFrame(df.select_dtypes('number')[coluna].value_counts())
                df_col4.reset_index(inplace=True)
                st.dataframe(df_col4.head(10), hide_index=True, use_container_width=True)

            with col3a:
                show_hist(df.select_dtypes('number'), coluna)

    name_cat = st.tabs(list(df.select_dtypes('category').columns))

    for idx, tabb in enumerate(name_cat):
        with tabb:
            coluna = df.select_dtypes('category').columns[idx]
            n_unique = df.select_dtypes('category')[coluna].nunique()
            n_null = df.select_dtypes('category')[coluna].isnull().sum()

            col1b, col2b, col3b = st.columns([1.2, 1, 2], gap="medium")

            with col1b:
                st.write('**Basic information**')
                df_col3 = pd.DataFrame(data=[n_unique, n_null],
                                       index=['# of distinct values', '# of missing cells'], columns=[' '])
                st.dataframe(df_col3, use_container_width=True)

            with col2b:
                st.write('**Most frequent values**')
                df_col4 = pd.DataFrame(df.select_dtypes('category')[coluna].value_counts())
                df_col4.reset_index(inplace=True)
                st.dataframe(df_col4.head(10), hide_index=True, use_container_width=True)

            with col3b:
                show_hist(df.select_dtypes('category'), coluna)

    name_text = st.tabs(list(df.select_dtypes('object').columns))
    for idx, tabb in enumerate(name_text):
        with tabb:
            coluna = df.select_dtypes('object').columns[idx]
            n_unique = df.select_dtypes('object')[coluna].nunique()
            n_null = df.select_dtypes('object')[coluna].isnull().sum()

            col1t, col2t, col3t = st.columns([1.2, 1, 2], gap="medium")
            with col1t:
                st.write('**Basic information**')
                df_col4 = pd.DataFrame(data=[n_unique, n_null],
                                       index=['# of distinct values', '# of missing cells'], columns=[' '])
                st.dataframe(df_col4, use_container_width=True)

            with col2t:
                st.write('**Most frequent words**')
                sp = ['(', ')', '.', ',', '"', "'"]
                c = ' '.join(df[coluna].dropna())
                for i in sp:
                    c = c.replace(i, '')
                dt = pd.DataFrame(c.split())
                df_col5 = pd.DataFrame(dt.value_counts())
                df_col5.reset_index(inplace=True)
                df_col5.columns = [coluna, 'count']
                st.dataframe(df_col5, hide_index=True, use_container_width=True)

            with col3t:
                st.write('**Word cloud**')
                wc = WordCloud(max_font_size=100, width=400, height=300, max_words=100, include_numbers=True,
                               background_color="white").generate(' '.join(df[coluna].dropna()))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.pyplot()

show_var_info(df_train)

st.divider()
# ----------------------------------------------------------------------------------------------------------------------

st.subheader('Features interaction')

list_graph = list(df_train.select_dtypes('number').columns) + list(df_train.select_dtypes('category').columns)
list_graph.remove(y_name)
tab1, tab2, tab3, tab4 = st.tabs(['Heatmap', 'Scatterplot', 'Grouped by count', 'Grouped by target feature'])
with tab1:
    show_heatmap(df_train)

with tab2:
    col1d, col2d, col3d = st.columns([1, 1, 3], gap="small")

    with col1d:
        X_axis = st.radio("Choose X-axis feature", list_graph, key='graphX2')

    with col2d:
        Y_axis = st.radio("Choose Y-axis feature", list_graph, key='graphY2')

    with col3d:
        fig = px.scatter(df_train, x=X_axis, y=Y_axis, symbol=y_name, color=y_name,
                         color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(xaxis_title=X_axis, yaxis_title=Y_axis, font={'size': 14, 'color': 'black'},
                          plot_bgcolor='white', scattermode="group")
        fig.update_xaxes(mirror=True, ticks='outside', showgrid=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showgrid=True, linecolor='black', gridcolor='lightgrey')
        st.plotly_chart(fig, theme=None, use_container_width=True)

with tab3:
    col1e, col2e = st.columns([1, 2], gap="large")

    try:
        with col1e:
            option1 = st.selectbox('Select 1st feature:', list(df_train.columns))
            option2 = st.selectbox('Select 2nd feature:', list(df_train.columns) + ['-'])
            option3 = st.selectbox('Select 3rd feature:', list(df_train.columns) + ['-'])
            option4 = st.selectbox('Select 4th feature:', list(df_train.columns) + ['-'])
            option_list = [option1, option2, option3, option4]
            option_list = [opt for opt in option_list if opt != '-']

        with col2e:
            st.dataframe(df_train.groupby(option_list).size().reset_index(name='count'),
                         hide_index=True, use_container_width=True)
    except ValueError:
        st.write(':x: Error: it must be different features.')

with tab4:
    col1c, col2c = st.columns([1, 4], gap="small")

    with col1c:
        X_axis = st.radio("Choose Y-axis features", list_graph, key='graphX1')

    with col2c:
        df_group = df_train.groupby([X_axis, y_name])[y_name].value_counts().reset_index()
        fig = px.histogram(df_group, y=X_axis, x='count', color=y_name, text_auto=True, orientation='h', nbins=20)
        fig.update_layout(xaxis_title='Frequency', yaxis_title=X_axis, font={'size': 14, 'color': 'black'},
                          template='plotly_dark', plot_bgcolor='white', bargap=0.01)
        fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        st.plotly_chart(fig, theme=None, use_container_width=True)

st.divider()
st.divider()
# ======================================================================================================================

st.header('Machine learning application')

st.subheader('Data preprocessing')


class DropObj(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **Object Feature dropper**')
        return self

    @staticmethod
    def transform(self, x):
        st.write('By default, all object features will be dropped from the whole dataset.')
        dropped_list = list(x.select_dtypes('object'))
        x.drop(columns=x.select_dtypes('object'), axis=1, inplace=True)
        st.write(':heavy_check_mark: Operation successfully done. **{}** were dropped'.format(', '.join(dropped_list)))

        return x


class FillNA(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **NaN replacer**')
        return self

    @staticmethod
    def transform(self, x):
        cols_na = [col for col in x.columns if any(x[col].isnull())]

        for c in cols_na:
            try:
                if pd.api.types.is_numeric_dtype(x[c]):
                    strat = st.radio(' :black_medium_small_square: What value to replace NaN in _{}_?'.format(c),
                                     ('mean', 'median', 'most_frequent', 'constant', 'skip', 'drop rows'),
                                     key='radio_' + c)
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                else:
                    strat = st.radio(' :black_medium_small_square: What value to replace NaN in _{}_?'.format(c),
                                     ('most_frequent', 'constant', 'skip', 'drop rows'),
                                     key='radio_' + c)
                    if strat == 'constant':
                        st.write(' :black_medium_small_square: What value to replace NaN in _{}_?'.format(c))

                if strat == 'skip':
                    st.write('Nothing will be done for this feature. It still keeps missing value(s).')
                    continue

                if strat == 'drop rows':
                    st.write('This feature has {} missing cells.'.format(x[c].isnull().sum()))
                    if any(filter(lambda z: z >= df_train.shape[0], list(x[x[c].isnull()].index))):
                        st.write(
                            ':x: CANNOT DO! At least one missing cell in {} belongs to the test dataset, which needs '
                            'to be submitted in its full size. The observations cannot be dropped from the analysis!'
                            .format(c))
                    else:
                        drop_idx = list(x[x[c].isnull()].index)
                        x.dropna(subset=c, inplace=True, ignore_index=False)
                        y.drop(index=drop_idx, inplace=True)
                        st.write(':heavy_check_mark: Operation successfully done: the rows have been dropped.')

                    continue

                if strat == 'constant':
                    value = st.text_input('Input which string/value: ', 'NA', key='texto_' + c)
                    value = int(value) if pd.api.types.is_numeric_dtype(x[c]) else str(value)

                else:
                    value = None

                imputer = impute.SimpleImputer(strategy=strat, fill_value=value)
                dt = imputer.fit_transform(x[[c]])
                x[c] = pd.DataFrame(dt)

                if pd.api.types.is_numeric_dtype(x[c]):
                    st.write(':heavy_check_mark: Operation successfully done. The value replace was **{:.2f}**.'
                             .format(imputer.statistics_[0]))

                if pd.api.types.is_object_dtype(x[c]):
                    x[c] = x[c].astype('category')
                    st.write(':heavy_check_mark: Operation successfully done. The value replace was **{}**.'
                             .format(imputer.statistics_[0]))

            except ValueError:
                st.write(':x: This imputer or strategy will not work. Try a different approach!')

        return x


class FeatureEng(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **New features Addition**')
        st.write('Select the feature you would like to create and add them to the whole dataset:')
        return self

    @staticmethod
    def transform(self, x):

        dict_feat = func_feat(x=x, x_ref=X_ref)
        for key, _ in dict_feat.items():
            want_feat = st.checkbox(dict_feat[key]['message_box'])
            if want_feat:
                try:
                    x[key] = dict_feat[key]['method']
                    st.write(":heavy_check_mark: Operation successfully done. Feature {} has been added to the dataset."
                             .format(key))
                except ValueError:
                    st.write(':x: Operation not performed. Feature not included!')

        return x


class DropRow(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **Row dropper**')
        return self

    @staticmethod
    def transform(self, x):
        st.write('Searching for null cells...')
        feat_na = list(x.columns[x.isnull().any()])
        if feat_na:
            for n in feat_na:
                st.markdown('- The feature {} has {} missing cells'.format(n, x[n].isnull().sum()))

            feat_na2 = feat_na.copy()
            for n in feat_na:
                if any(filter(lambda z: z >= df_train.shape[0], list(x[x[n].isnull()].index))):
                    st.write(':x: CANNOT DO! At least one missing cell in {} belongs to the test dataset, which needs '
                             'to be submitted in its full size. This observation cannot be dropped from the analysis!'
                             .format(n))
                    feat_na2.remove(n)

            if feat_na2:
                rows_na = st.multiselect('Select the feature you want to remove NaN row from:',
                                         feat_na2)
                x.dropna(axis=0, subset=rows_na, inplace=True)
                if rows_na:
                    st.write(':heavy_check_mark: Operation successfully done. The dataset has now {} observations.'
                             .format(x.shape[0]))

        else:
            st.write(':heavy_check_mark: No action needed: there is no longer missing cells in the whole dataset.')

        return x


class DropColumn(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **Feature dropper**')
        return self

    @staticmethod
    def transform(self, x):
        st.write('By default, all object features will be dropped from the whole dataset.')
        # dropped_list = list(x.select_dtypes('object'))
        x.drop(columns=x.select_dtypes('object'), axis=1, inplace=True)
        colunas = st.multiselect('Are there any other features would you like to drop from the dataset?',
                                 list(x.select_dtypes(['category', 'number'])))
        x = x.drop(columns=x[colunas], axis=1)
        st.write(':heavy_check_mark: Operation successfully done. The features in the dataset are **{}**.'
                 .format(', '.join(x.columns)))
        return x


class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write(':arrow_forward: **Categorical encoding**')
        return self

    @staticmethod
    def transform(self, x):
        list_enc = list(x.select_dtypes('category'))
        list_enc2 = list_enc.copy()

        for c in list_enc2:
            if x[c].nunique() == 2:
                proxy = {x[c].unique()[0]: 0, x[c].unique()[1]: 1}
                st.write(f':heavy_check_mark: Ordinal Encoding has been applied for **{c}**: 0 for {x[c].unique()[0]} '
                         f'and 1 for {x[c].unique()[1]}.')
                x[c] = x[c].replace(proxy)
                list_enc.remove(c)

        st.write(f":heavy_check_mark: One Hot Encoding has been applied for features **{', '.join(list_enc)}**.")
        dumm = pd.get_dummies(x[list_enc], dtype=int)
        x = pd.concat([x, dumm], axis=1)
        x.drop(columns=list_enc, inplace=True)

        return x


class Pipelines:

    @staticmethod
    def preproc_pipeline(x):
        pipe2 = Pipeline([('drpO', DropObj()), ('imp', FillNA()),
                          ('feng', FeatureEng()),
                          ('drp', DropColumn()), ('enc', Encoder())])
        return pipe2.fit_transform(x)

    @staticmethod
    def scaling(x):
        return pd.DataFrame(preprocessing.StandardScaler().fit_transform(x))

    @staticmethod
    @st.cache_resource
    def classifiers_default(X_tgt, y_tgt, y_n):
        setup(data=pd.concat([X_tgt, y_tgt], axis=1), target=y_n, preprocess=False, fold=5, verbose=False,
              data_split_stratify=False, data_split_shuffle=False, fold_strategy='kfold')
        best = compare_models(verbose=False, fold=5, include=['gbc', 'lr', 'knn', 'dt', 'rf'])
        r = pull()
        r.drop(columns='TT (Sec)', inplace=True)
        st.dataframe(r, hide_index=True, use_container_width=True)

        return best

    @staticmethod
    @st.cache_resource
    def hypertuning_models(_best):
        tuned = tune_model(_best, choose_better=True, verbose=False, search_library='optuna')
        r2 = pull()
        st.dataframe(r2, hide_index=False, use_container_width=True)
        st.write('The accuracy for the hypertuned model is {}.'.format(r2.loc['Mean', 'Accuracy']))

        return tuned


X_train_test = pd.concat([df_train.drop(columns=y_name, axis=1), df_test], axis=0, ignore_index=True)
X_train_test = set_dtypes(X_train_test)

X_train_test.loc[0:len(df_train), 'IsTrain'] = 1
X_train_test.loc[len(df_train):, 'IsTrain'] = 0

X_ref = X_train_test.copy()

main_pipe = Pipelines()

X_pipe = main_pipe.preproc_pipeline(X_train_test)

X_train0 = X_pipe.loc[X_pipe['IsTrain'] == 1]
X_test0 = X_pipe.loc[X_pipe['IsTrain'] == 0]

X_train0.drop(columns='IsTrain', inplace=True)
X_test0.drop(columns='IsTrain', inplace=True)

X_train = main_pipe.scaling(X_train0)
X_test = main_pipe.scaling(X_test0)
X_pipe.drop(columns='IsTrain', inplace=True)

col_final = X_pipe.columns
X_train.columns = X_test.columns = list(col_final)

y.reset_index(drop=True, inplace=True)

with st.expander('Click here to see the dataframe ready for training and testing'):

    tab_X, tab_y, tab_corr = st.tabs(['X-data', 'y-data', 'Correlation matrix'])
    with tab_X:
        st.write(f'Size of train dataset: {X_train.shape}')
        st.dataframe(X_train0, hide_index=False, use_container_width=True)

    with tab_y:
        st.write(f'Size of target feature: {y.shape}')
        st.dataframe(y, hide_index=False, use_container_width=True)

    with tab_corr:
        X_corr = X_train.copy()
        X_corr.insert(0, y_name, y)
        st.dataframe(X_corr.corr())

st.divider()

st.subheader('Model training')

if st.toggle('Train ML models'):
    st.write(':arrow_forward: **Training models using default parameters...**')
    best_model = main_pipe.classifiers_default(X_tgt=X_train, y_tgt=y, y_n=y_name)
    chosen_model = best_model

    want_hyper = st.toggle('Hypertune best default model')

    if want_hyper:
        st.write(':arrow_forward: **Training the top model varying parameters using Random Search...**')
        tuned_ml = main_pipe.hypertuning_models(best_model)
        chosen_model = tuned_ml

    X_pred_train = predict_model(chosen_model, data=X_train)
    y_surv_train = X_pred_train['prediction_label'].reset_index(drop=True)
    # st.write('The confusion matrix for the prediction is shown below.')
    # df_conf = pd.DataFrame(confusion_matrix(y, y_surv_train), index=['True 0', 'True 1'],
    #                        columns=['Predicted 0', 'Predicted 1'])
    # st.dataframe(df_conf)

    st.divider()
    st.subheader('Model testing')

    with st.echo():
        X_pred = predict_model(chosen_model, data=X_test)
        y_surv = X_pred['prediction_label'].reset_index(drop=True)
        df_submit = pd.concat([df_test['PassengerId'], y_surv], axis=1, ignore_index=True)
        df_submit.columns = ['PassengerId', y_name]

        proxy_y = {df_submit[y_name].unique()[0]: True, df_submit[y_name].unique()[1]: False}
        df_submit = df_submit.replace(proxy_y)


        csv_submit = df_submit.to_csv(index=False)

    with st.expander('Click here to see the test dataset and the predicted values of the target feature'):
        tab_pred, tab_test = st.tabs(['Predicted values', 'Test dataset'])
        with tab_pred:
            st.dataframe(df_submit, hide_index=True, use_container_width=True)

        with tab_test:
            st.dataframe(df_test, hide_index=True, use_container_width=True)


    def download_file():
        st.write(':arrow_forward: **Download dataframe into csv file**')
        filename = st.text_input('Enter file name:', value=p_title + '_file.csv')
        filename = str(filename)
        st.caption('*file name must end with .csv')
        if filename.endswith('.csv'):
            st.download_button(label="Download csv file", data=csv_submit, file_name=filename, mime='text/csv')

        else:
            st.write(':x: File cannot be download! The file name must end with .csv')

    download_file()
