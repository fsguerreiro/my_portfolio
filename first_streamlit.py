import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title='The Titanic dataset', layout='wide')
st.title('The Titanic dataset', anchor='titanic_train',
         help='This dataset is available on [kaggle](https://www.kaggle.com/competitions/titanic).')
st.header('Training dataset exploration')

#st.checkbox("Use container width", value=False, key="use_container_width")

df_train = pd.read_csv('https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/titanic_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/fsguerreiro/my_portfolio/main/titanic_test.csv')

tab1, tab2, tab3 = st.tabs(["First rows", "Last rows", "Full training dataset"])
with tab1:
    st.dataframe(df_train.head(10), hide_index=True)

with tab2:
    st.dataframe(df_train.tail(10), hide_index=True)

with tab3:
    st.dataframe(df_train, hide_index=True)

n_row, n_col = df_train.shape
n_blank = df_train.isnull().sum().sum()
p_blank = n_blank/df_train.size
double_rows = df_train.duplicated().sum()

df_train.Embarked = df_train.Embarked.astype(str)
df_num = df_train[['PassengerId', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
df_tex = df_train[['Sex', 'Survived', 'Embarked']]

st.write('\n\n')
st.divider()
st.subheader('Overview')

col1, col2 = st.columns([1, 2], gap="small")

with col1:
    st.write('**Dataset statistics**')
    df_col1 = pd.DataFrame(data=[n_row, n_col, n_blank, f"{p_blank:.2%}", double_rows],
              index=['Number of observations', 'Number of variables', 'Number of missing values',
                     'Missing values (%)', 'Number of duplicate rows'], columns=[' '])
    st.dataframe(df_col1, use_container_width=True)

with col2:
    st.write('**Type of variables**')
    df_col2 = pd.DataFrame(data=[df_tex.shape[1], list(df_tex.columns),
                                 df_num.shape[1], list(df_num.columns)],
                           index=['Number of categorical variables', 'Categorical variables',
                                 'Number of numerical variables', 'Numerical variables'], columns=[' '])
    st.dataframe(df_col2, use_container_width=True)

st.divider()
st.subheader('Variables information')
name_num = st.tabs(list(df_num.columns))

for idx, tabb in enumerate(name_num):
    with tabb:
        coluna = df_num.columns[idx]
        n_unique = df_num[coluna].nunique()
        n_null = df_num[coluna].isnull().sum()
        n_mean = df_num[coluna].mean()
        n_median = df_num[coluna].median()
        n_min = df_num[coluna].min()
        n_max = df_num[coluna].max()

        col1, col2, col3 = st.columns([1.2, 1, 2], gap="medium")

        with col1:
            st.write('**Basic information**')
            df_col3 = pd.DataFrame(data=[n_unique, n_null, n_min, n_mean, n_median, n_max],
                                   index=['Number of distinct values', 'Number of missing cells',
                                          'Minimum value', 'Mean value', 'Median value',
                                          'Maximum value'], columns=[' '])
            st.dataframe(df_col3, use_container_width=True)

        with col2:
            st.write('**Most frequent values**')
            df_col4 = pd.DataFrame(df_num[coluna].value_counts())
            df_col4.reset_index(inplace=True)
            st.dataframe(df_col4.head(8), hide_index=True, use_container_width=True)

        with col3:
            st.write('**Histogram of _{}_**'.format(coluna))
            fig, ax = plt.subplots()
            ax.hist(df_num[coluna], bins=20)
            ax.set_xlabel(coluna, fontsize=13)
            ax.set_ylabel("Frequency", fontsize=13)
            ax.grid(True)
            st.pyplot(fig)


name_tex = st.tabs(list(df_tex.columns))

for idx, tabb in enumerate(name_tex):
    with tabb:
        coluna = df_tex.columns[idx]
        n_unique = df_tex[coluna].nunique()
        n_null = df_tex[coluna].isnull().sum()

        col1, col2, col3 = st.columns([1.2, 1, 2], gap="medium")

        with col1:
            st.write('**Basic information**')
            df_col3 = pd.DataFrame(data=[n_unique, n_null],
                                   index=['Number of distinct values', 'Number of missing cells'],
                                   columns=[' '])
            st.dataframe(df_col3, use_container_width=True)

        with col2:
            st.write('**Most frequent values**')
            df_col4 = pd.DataFrame(df_tex[coluna].value_counts())
            df_col4.reset_index(inplace=True)
            st.dataframe(df_col4.head(8), hide_index=True, use_container_width=True)

        with col3:
            st.write('**Histogram of _{}_**'.format(coluna))
            fig, ax = plt.subplots()
            ax.hist(df_tex[coluna], bins=20)
            ax.set_xlabel(coluna, fontsize=13)
            ax.set_ylabel("Frequency", fontsize=13)
            ax.grid(True)
            st.pyplot(fig)




st.markdown("[Back to top](#titanic_train)")




with st.sidebar:
    st.markdown("[Training data exploration](#titanic_train)")
