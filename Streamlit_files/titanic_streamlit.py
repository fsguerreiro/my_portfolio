import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import svm, preprocessing, impute, tree
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import time


# Building functions:

@st.cache_data
def show_hist(dataframe, column):
    figg = px.histogram(dataframe, column)
    figg.update_layout(xaxis_title=column, yaxis_title="Frequency", font={'size': 14, 'color': 'black'},
                       template='plotly_dark', plot_bgcolor='white', bargap=0.1)
    figg.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    figg.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    st.plotly_chart(figg, theme=None, use_container_width=True)


@st.cache_data
def show_heatmap(dataframe):
    df_heatmap = dataframe.copy()
    df_heatmap.Survived = df_heatmap.Survived.astype(int)
    a = pd.get_dummies(df_cat.drop(columns='Survived', axis=1), dtype=int)
    df_heatmap2 = pd.concat([df_heatmap, a], axis=1)
    fiig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_heatmap2.corr(numeric_only=True).round(3), annot=True, cmap='coolwarm', linewidth=.5, cbar=False)

    ax.tick_params(axis='both', labelsize=12, grid_color='b')
    st.pyplot(fiig)


def make_filter():

    options = st.multiselect(
        'Select which variables to filter by:',
        list(df_train.columns))
    st.caption('*If no variable is selected, the method will return all rows of the dataset')

    df_filter = df_train.copy()
    df_filter.loc[~df_filter.Age.isnull(), 'Age'] = df_filter.loc[~df_filter.Age.isnull(), 'Age'].astype('int')

    if options:
        for opt in options:
            if df_filter[opt].dtype == 'category':
                filt = st.radio('Select {}:'.format(opt), tuple(df_filter[opt].unique()))
                df_filter = df_filter.loc[(df_filter[opt] == filt)]

            elif df_filter[opt].dtype in ['int64', 'float64']:
                filt = st.slider('Select {}:'.format(opt), int(df_train[opt].min()), int(df_train[opt].max())+1,
                                 (int(df_train[opt].min()), int(df_train[opt].mean())+2), 1)
                df_filter = df_filter.loc[(df_filter[opt] >= filt[0]) & (df_filter[opt] <= filt[1])]

            else:
                filt = st.text_input('Input {}'.format(opt), key='texto_'+opt)
                filt = str(filt)
                df_filter = df_filter.loc[df_filter[opt].str.contains(filt, case=False, na=False)]

        st.write("{} result(s) found.".format(df_filter.shape[0]))

    else:
        st.write("{} result(s) found.".format(df_filter.shape[0]))

    return df_filter


@st.cache_data
def load_data():
    train = pd.read_csv(
        r'https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/Streamlit_files/titanic_train.csv')
    test = pd.read_csv(
        r'https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/Streamlit_files/titanic_test.csv')

    return train, test

# ----------------------------------------------------------------------------------------------------------------------
# Page title, datasets download, and defining types of variables:

st.set_page_config(page_title='Titanic challenge', page_icon=":ship:", layout='wide')

st.title(':passenger_ship: Titanic - Machine Learning from Disaster (kaggle competition)', anchor='titanic_train')

st.write('''
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, 
during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. 
Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 
passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to 
survive than others. The goal of this competition is to build a predictive model that answers the question: “what 
sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). 
''')

st.header('Training dataset visualization', help='Both datasets are available for download on [kaggle]('
                                                 'https://www.kaggle.com/competitions/titanic).')

st.write('''In this competition, it's granted access to two similar datasets that include passenger information like 
name, age, gender, socio-economic class, etc. One dataset is titled **train.csv** and the other is titled **test.csv**.

**Train.csv** will contain the details of a subset of the passengers on board (891 to be exact) and importantly, 
will reveal whether they survived or not, also known as the “ground truth”. The **test.csv** dataset contains similar 
information but does not disclose the “ground truth” for each passenger. Predicting these outcomes is the objective.

To get started, the datasets were loaded from my github account:

''')


with st.echo('above'):
    data_load_state = st.text('Loading data...')
    df_train, df_test = load_data()

time.sleep(2)
data_load_state.text("Datasets successfully loaded! (using st.cache_data)")


df_train.loc[df_train.Embarked.isnull(), 'Embarked'] = 'S'


list_category = ['Pclass', 'Survived', 'Embarked', 'Sex']
df_train[list_category] = df_train[list_category].astype('category')
df_num = df_train.select_dtypes('number')
df_cat = df_train.select_dtypes('category')
df_tex = df_train.select_dtypes('object')

# ----------------------------------------------------------------------------------------------------------------------
# Filter data function is in the sidebar (left-hand side of the page): results are shown in the tab 'filtered data'

st.write('''
The filter function can be accessed by the sidebar on the left side of the page. The filtered data is shown below:
''')

with st.sidebar:
    st.subheader('Filter train dataset')
    df_filtered = make_filter()

# ----------------------------------------------------------------------------------------------------------------------

# Visualize the titanic dataset:
st.write('**Filtered train dataset**')
st.dataframe(df_filtered, hide_index=True, use_container_width=True)

# Expanded tab to show explanation of the variables

st.write('Here is a brief explanation about the passenger features:')
with st.expander("See variable notes"):
    definition = ['Passenger identification number in the dataset',
                  'Survival status: 0 if passed away or 1 if survived',
                  'A proxy for socio-economic status: 1 for upper, 2 for middle and 3 for lower classes',
                  'Passenger name',
                  'Passenger gender',
                  'Age in years: It is fractional if less than 1 or in the form of xx.5 if estimated',
                  'Number of siblings and spouses',
                  'Number of parents and children',
                  'Ticket number',
                  'Passenger fare',
                  'Cabin number',
                  'Port of embarkation:	C = Cherbourg, Q = Queenstown, S = Southampton']
    df_exp = pd.DataFrame({'Variable': df_train.columns, 'Definition': definition})
    st.dataframe(df_exp, hide_index=True, use_container_width=True)

st.divider()
# ----------------------------------------------------------------------------------------------------------------------

# Overview of the dataset


@st.cache_data(experimental_allow_widgets=True)
def show_overview():

    st.subheader('Overview')

    col1, col2 = st.columns([1, 1.5], gap="small")
    with col1:
        st.write('**Dataset statistics**')

        st.markdown("- Number of observations: {}".format(df_train.shape[0]))
        st.markdown("- Number of variables: {}".format(df_train.shape[1]))
        st.markdown("- Number of missing cells: {}".format(df_train.isnull().sum().sum()))
        st.markdown("- Missing cells (%): {:.2%}".format(df_train.isnull().sum().sum()/df_train.size))
        st.markdown("- Number of duplicated rows: {}".format(df_train.drop(columns=['PassengerId']).duplicated().sum()))

    with col2:
        st.write('**Type of variables**')

        df_type = pd.concat([pd.Series(df_num.columns), pd.Series(df_cat.columns), pd.Series(df_tex.columns)], axis=1)
        df_type = df_type.set_axis(['Numerical variables', 'Categorical variables', 'Text variables'], axis='columns')
        st.dataframe(df_type, hide_index=True, use_container_width=True)


show_overview()

st.divider()
# ---------------------------------------------------------------------------------------------------------------------

# Variable information


@st.cache_data(experimental_allow_widgets=True)
def show_var_info():
    st.subheader('Variables information')
    name_num = st.tabs(list(df_num.columns))

    for idx, tabb in enumerate(name_num):
        with tabb:
            coluna = df_num.columns[idx]

            col1a, col2a, col3a = st.columns([1.3, 1, 2.5], gap="medium")

            with col1a:
                st.write('**Basic information**')

                df_n1 = pd.DataFrame(data=[df_num[coluna].nunique(), df_num[coluna].isnull().sum()],
                                     index=['# of distinct values', '# of missing cells'], columns=[' '])

                df_n2 = pd.DataFrame(list(df_num[coluna].describe()), index=['# of non-null cells', 'Mean value', 'Std '
                                                                             'deviation', 'Minimum', '1st quartile',
                                                                             'Median', '3rd quartile', 'Maximum'],
                                     columns=[' '])
                df_stat = pd.concat([df_n1, df_n2], axis=0)
                st.dataframe(df_stat, use_container_width=True)

            with col2a:
                st.write('**Most frequent values**')
                df_col4 = pd.DataFrame(df_num[coluna].value_counts())
                df_col4.reset_index(inplace=True)
                st.dataframe(df_col4.head(10), hide_index=True, use_container_width=True)

            with col3a:
                show_hist(df_num, coluna)

    name_cat = st.tabs(list(df_cat.columns))

    for idx, tabb in enumerate(name_cat):
        with tabb:
            coluna = df_cat.columns[idx]
            n_unique = df_cat[coluna].nunique()
            n_null = df_cat[coluna].isnull().sum()

            col1b, col2b, col3b = st.columns([1.2, 1, 2], gap="medium")

            with col1b:
                st.write('**Basic information**')
                df_col3 = pd.DataFrame(data=[n_unique, n_null],
                                       index=['Number of distinct values', 'Number of missing cells'], columns=[' '])
                st.dataframe(df_col3, use_container_width=True)

            with col2b:
                st.write('**Most frequent values**')
                df_col4 = pd.DataFrame(df_cat[coluna].value_counts())
                df_col4.reset_index(inplace=True)
                st.dataframe(df_col4.head(10), hide_index=True, use_container_width=True)

            with col3b:
                show_hist(df_cat, coluna)


show_var_info()

st.divider()
# ----------------------------------------------------------------------------------------------------------------------

st.subheader('Variables interaction')

tab1, tab2, tab3 = st.tabs(['Heatmap', 'Grouped by Survived', 'Scatterplot'])
with tab1:
    show_heatmap(df_train)

with tab2:
    col1c, col2c = st.columns([1, 4], gap="small")

    with col1c:
        list_graph = list(df_num.columns) + list(df_cat.columns)
        list_graph.remove('Survived')
        X_axis = st.radio("Choose Y-axis variable", list_graph, key='graphX1')

    with col2c:
        df_group = df_train.groupby([X_axis, 'Survived'])['Survived'].value_counts().reset_index()
        df_group.Survived = df_group.Survived.astype(str)
        df_group.Pclass = df_group.Pclass.astype(str) if X_axis == 'Pclass' else ''
        fig = px.histogram(df_group, y=X_axis, x='count', color="Survived", text_auto=True, orientation='h', nbins=20)
        fig.update_layout(xaxis_title='Frequency', yaxis_title=X_axis, font={'size': 14, 'color': 'black'},
                          template='plotly_dark', plot_bgcolor='white', bargap=0.01)
        fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        st.plotly_chart(fig, theme=None, use_container_width=True)

with tab3:
    col1d, col2d, col3d = st.columns([1, 1, 3], gap="small")

    with col1d:
        X_axis = st.radio("Choose X-axis variable", df_train[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare',
                                                              'Embarked']].columns, key='graphX2')

    with col2d:
        Y_axis = st.radio(
            "Choose Y-axis variable",
            df_train[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch',
                      'Fare', 'Embarked']].columns, key='graphY2')

    with col3d:
        fig = px.scatter(df_train, x=X_axis, y=Y_axis, symbol='Survived', color='Survived',
                         color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(xaxis_title=X_axis, yaxis_title=Y_axis, font={'size': 14, 'color': 'black'},
                          plot_bgcolor='white', scattermode="group")
        fig.update_xaxes(mirror=True, ticks='outside', showgrid=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showgrid=True, linecolor='black', gridcolor='lightgrey')
        st.plotly_chart(fig, theme=None, use_container_width=True)

st.divider()
# ======================================================================================================================

st.header('Machine learning application')

st.subheader('Data preprocessing')


class FeatureEng(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write('**Feature Engineering**')
        return self

    @staticmethod
    def transform(self, x):
        want_age_interval = st.checkbox("Add 'AgeInterval' categorical variable as ranges of 'Age'")
        if want_age_interval:
            try:
                x['AgeInterval'] = pd.qcut(x['Age'], [0, 0.2, 0.4, 0.6, 0.8, 1], duplicates='drop')
                # x.drop(columns=['Age'], inplace=True)
            except TypeError:
                st.write('Operation not performed!')

        want_fare_interval = st.checkbox("Add 'FareInterval' categorical variable as ranges of 'Fare'")
        if want_fare_interval:
            try:
                x['FareInterval'] = pd.qcut(x['Fare'], [0, 0.25, 0.5, 0.75, 1])
                # x.drop(columns=['Fare'], inplace=True)
            except:
                st.write('Operation not performed!')

        want_fare_pp = st.checkbox("Add 'FarePp' numerical variable as the ratio of 'Fare' to the family number")
        if want_fare_pp:
            try:
                x['FarePp'] = x['Fare'] / (x['SibSp'] + x['Parch'] + 1)
                # x.drop(columns=['Fare'], inplace=True)
            except:
                st.write('Operation not performed!')

        want_family_size = st.checkbox("Add 'FmSize' numerical variable as the sum of 'SibSp' and 'Parch'")
        if want_family_size:
            try:
                x['FmSize'] = x['SibSp'] + x['Parch']
                # x.drop(columns=['SibSp', 'Parch'], inplace=True)
            except:
                st.write('Operation not performed!')

        want_title = st.checkbox("Add 'Title' categorical variable as the title of each passenger")
        if want_title:
            try:
                x['Title'] = df_train['Name'].apply(lambda w: w.split('. ')[0].split(', ')[1])
                x['Title'] = x['Title'].astype('category')
            except:
                st.write('Operation not performed!')

        want_alone = st.checkbox("Add 'IsAlone' categorical variable as the number of relatives with the passenger")
        if want_alone:
            try:
                x['IsAlone'] = (x['Parch'] == 0) & (x['SibSp'] == 0)
                x['IsAlone'] = x['IsAlone'].astype(int).astype('category')
            except:
                st.write('Operation not performed!')

        return x


class DropColumn(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write('**Column drop option**')
        return self

    @staticmethod
    def transform(self, x):
        colunas = st.multiselect('Select which variables you want to drop from the dataset:', list(x.columns),
                                 list(df_tex.columns))
        return x.drop(columns=x[colunas], axis=1)


class FillNA(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write('**NaN replacement**')
        return self

    @staticmethod
    def transform(self, x):
        cols_na = [col for col in x.columns if any(x[col].isnull())]

        for c in cols_na:
            try:
                strat = st.radio('What value to replace NaN in _{}_?'.format(c), ('mean', 'median', 'most_frequent',
                                                                                  'constant', 'grouping median'),
                                 key='radio_' + c)

                if strat == 'grouping median':
                    x[c] = x[c].fillna(x.groupby(['Sex', 'Pclass', 'Embarked'])[c].transform('median'))
                    continue

                if strat == 'constant':
                    value = st.text_input('Input which string/value: ', 'NA', key='texto_' + c)
                    value = int(value) if x[c].dtype in ['int64, int32', 'float64'] else str(value)

                else:
                    value = None

                imputer = impute.SimpleImputer(strategy=strat, fill_value=value)
                dt = imputer.fit_transform(x[[c]])
                x[c] = pd.DataFrame(dt)

            except ValueError:
                st.write('This imputer or strategy will not work. Try a different approach!')

        return x


class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, x):
        st.write('**Categorical encoding**')
        return self

    @staticmethod
    def transform(self, x):
        st.write('OneHotEncoder has been applied for {}.'.format(
            list(x.select_dtypes('category'))))
        dum = pd.get_dummies(x.select_dtypes('category'), dtype=int)
        x = pd.concat([x, dum], axis=1)
        x.drop(columns=list(x.select_dtypes('category')), inplace=True)

        return x


class Pipelines:

    @staticmethod
    def preproc_pipeline(x):
        pipe = Pipeline([('drp', DropColumn()), ('imp', FillNA()), ('feng', FeatureEng()), ('enc', Encoder())])
        return pipe.fit_transform(x)

    @staticmethod
    def scaling(x):
        return pd.DataFrame(preprocessing.StandardScaler().fit_transform(x))

    @staticmethod
    def classifiers_default(xx, yy):
        ml_models = {'RandomForest': RandomForestClassifier(), 'SVM': svm.SVC(),
                     'Logistic Regression': LogisticRegression(max_iter=1000),
                     'Gradient Boosting': GradientBoostingClassifier(), 'Decision Tree': tree.DecisionTreeClassifier()}

        use_clf = st.checkbox('Run ML models to train and test by default parameters')

        if use_clf:
            ml_selected = st.multiselect('Select model to fit data:', list(ml_models.keys()), list(ml_models.keys()))
            for model in ml_selected:
                clf = ml_models[model]
                scores = cross_val_score(clf, xx, yy, cv=5)
                scr = scores.mean()
                st.markdown('- The score for {} is {:.2%}'.format(model, scr))

    @staticmethod
    def hypertuning_models(x, y):
        models = {'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'params': {"n_estimators": list(range(100, 500, 50)), "learning_rate": [0, 0.01, 0.05, 0.1, 0.5, 1],
                       "max_depth": [3, 4, 5, 6, 7], 'loss': ['log_loss', 'exponential']}
                                       }
                 }

        want_hyper = st.checkbox('Check if you want to hypertune parameters on model')
        if want_hyper:
            grid_search = RandomizedSearchCV(models['Gradient Boosting']['model'],
                                             models['Gradient Boosting']['params'], cv=5,
                                             n_iter=10, scoring='accuracy', return_train_score=False)

            grid_search.fit(x, y)
            best_scr = grid_search.best_params_
            st.write(grid_search.best_params_)

            clf2 = GradientBoostingClassifier().set_params(**best_scr)
            scors = cross_val_score(clf2, x, y, cv=5)
            st.write('The score is: {:.2%}'.format(scors.mean()))


pipes = Pipelines()

y = df_train.Survived
# df_train2 = df_train.drop(columns='Survived', axis=1)
df_full = pd.concat([df_train.drop(columns='Survived', axis=1), df_test], axis=0, ignore_index=True)
list_category = ['Pclass', 'Embarked', 'Sex']
df_full[list_category] = df_full[list_category].astype('category')

df_ml = pipes.preproc_pipeline(df_full)
X = df_ml.copy()
X_scaler = pipes.scaling(X)
X_train = X_scaler[0:df_train.shape[0]][:]


with st.expander('Click here to see the dataframe ready for training and testing'):

    tab_X, tab_y, tab_corr = st.tabs(['X-data', 'y-data', 'Correlation matrix'])
    with tab_X:
        st.dataframe(X, hide_index=False, use_container_width=True)

    with tab_y:
        st.dataframe(y, hide_index=False, use_container_width=True)

    with tab_corr:
        st.dataframe(df_ml.corr())

# col_sp, col_rnd = st.columns([2, 2], gap='medium')
# with col_sp:
#     p_split = st.slider('Select sample split for testing: ', 0.1, 0.5, 0.3, 0.01)
#
# with col_rnd:
#     n_random = st.slider('Select random state: ', 0, 50, 42, 1)

# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=p_split, random_state=n_random,
#                                                     stratify=df_ml[['Survived', 'Pclass_3', 'Sex_female']])

col_default, col_hyper = st.columns([2, 2], gap='large')
with col_default:
    pipes.classifiers_default(X_train, y)

with col_hyper:
    pipes.hypertuning_models(X_train, y)
