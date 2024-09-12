import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title='My_Streamlit')
st.header('house')
st.subheader('rent')
st.latex('y=mx*+c')
st.text('this a caption')
st.text('_this a caption_')
st.markdown('Travel Industry,Property Management and Tourism')
st.write('Travel Industry,Property Management and Tourism')
st.metric('AAPl',"$70",'4%')
st.metric('speed','90mph','10%')
st.code('''def print(a):
        print(a)''',language='python')

st.code('''#include<stdio.h>
        int main(){inta=5;
                   printf('%d',a)}''',language='c')

k=pd.read_csv(r"C:\Users\kutlc\Downloads\Global Youtube Statistics")
st.table(k)

h={'a':[1,2,3],'b':[8,9,10]}
st.json(h)


col1,col2=st.columns(2)
with col1:
    st.metric('AAPL',"$70",'4%')
with col2:    
    st.metric('speed','90mph','10%')

submit=st.button('submit')

if submit:
    st.dataframe(k)
    st.radio('choose one:',('a','b','c'))
cols=list(k.columns)
name=st.selectbox('choose a feature:',cols)
st.write(cols)

col1,col2=st.columns(2)
with col1:
    a=st.number_input('enter a number1:')
with col2:    
    b=st.number_input('enter a number2:')


submit=st.checkbox('Add')
if submit:
    st.write('the sum of a and b is ',a+b)
    option=st.multiselect('choose your varaiable',cols)

    st.write('your selection:',option)


submit=st.checkbox('showimage')
if submit:
    img=Image.open(r"C:\Users\kutlc\OneDrive\Documents\Resume3\Image20240821174114.jpg")
    st.image(img)


st.header('Plotly charts in streamlit')
data=pd.read_csv(r"C:\Users\kutlc\Downloads\Global Youtube Statistics")
st.dataframe(data.head())


columns=list(data.columns)
target=st.selectbox('choose a target:',columns)
col2=columns.copy()
col2.remove(target)
x_var=st.selectbox('choose a X variable:',col2)
y_var=st.selectbox('choose a Y variable:',col2)
fig=px.scatter(data,x=x_var,y=y_var,color=target)
st.plotly_chart(fig)

import numpy as np
data=pd.DataFrame(np.random.randn(100,3),columns=['a','b','c'])
st.line_chart(data,use_container_width=True)
st.bar_chart(data,use_container_width=True)
st.area_chart(data,use_container_width=True)
     
st.header('status elements in streamlit')
st.success('success function')
st.info('info function')
st.warning('warning function')
st.error('error function')
st.exception('exception function')

st.title('Download Button')
def data_read(data):
    return data.to_csv().encode('utf-8')
data=pd.read_csv(r"C:\Users\kutlc\Downloads\train.csv")
csv=data_read(data)
st.download_button(label='downloadbutton',
                   data=csv,
                   mime='text/csv',
                   file_name='train')

with open (r"C:\Users\kutlc\OneDrive\Documents\Resume3\Image20240821174114.jpg") as file:
    st.download_button(label='downloadimage',
                       data='file',
                       file_name='img.jpg',
                       mime='image/jpg')


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
st.title('linear Regression App')
dataset=st.sidebar.file_uploader('dataset',type='csv')
if dataset is not None:
    data=pd.read_csv(dataset)
    st.dataframe(data)

    x=data.iloc[:,0]
    y=data.iloc[:,1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


    st.write(x_train.shape)
    st.write(type(x_train))

    x_train=np.array(list(x_train))
    x_test=np.array(list(x_test))
    model=LinearRegression()
    model.fit(x_train.reshape(-1,1),y_train)

    predict=model.predict(x_test.reshape(-1,1))
    df2=pd.DataFrame({'Actual salary':y_test,'predictsalary':predict})
    st.dataframe(df2)
    mse=mean_squared_error(y_test,predict)
    r2=r2_score(y_test,predict)
    st.info(f'the mse value is {mse}')
    st.warning(f'the r2_score is {r2}')
    st.success('model training is done')


     
     

import sqlite3
st.title('streamlit with sqlite3')

page=st.sidebar.selectbox("pages",['sql injection','sql table viewer'])
if page=='sql injection':
            st.write('this is to injection infor to the sql database')
            
            sepal_length=st.number_input("insert sepal length number")
            sepal_width=st.number_input("insert sepal_width number")
            petal_length=st.number_input("insert petal_length number")
            petal_width=st.number_input("insert petal_width number")

            species=st.selectbox('species',('setosa','versicolor','viriginica'))

            submit=st.button('sql_injection')
            if submit:

                #connect to sqlite3( create a new database)
                conn=sqlite3.connect(r"C:\Users\kutlc\OneDrive\Documents\infor.db")

                #create a cursor
                cursor=conn.cursor()

                    #insert the random values into table
                cursor.execute('INSERT INTO infor (sepal_length,sepal_width,petal_length,petal_width,species)VALUES(?,?,?,?,?)',(sepal_length,sepal_width,petal_length,petal_width,species))

                #commit the chabges close the database connection
                conn.commit()
                conn.close()




else:
    
    
        conn=sqlite3.connect(r"C:\Users\kutlc\OneDrive\Documents\infor.db")
        cursor=conn.cursor()
        g=cursor.execute('select * from infor')
        data=g.fetchall()
        columns=[desc[0] for desc in cursor.description]
        #create a pandas DataFrame from fetched data and column names
        df=pd.DataFrame(data,columns=columns)
        st.dataframe(df)


st.title('Layers and Containers')
tab1,tab2,tab3,tab4=st.tabs(['RE','MT','R15','RX'])
with tab1:
     st.header('this is bikenum1')

with tab2:
     st.header('this is bikenum2')
    
with tab3:
     st.header('this is bikenum3')

col1,col2,col3=st.columns(3)
with col1:
     b1=st.button('Submit')
with col2:
     b2=st.button('login')

with col3:
     b3=st.button('logout')

with st.expander('open me'):
     st.header('Hello GoodMorning I am kuttalam')
     st.image(r"C:\Users\kutlc\OneDrive\Documents\Resume3\Image20240821174114.jpg")


st.title('Streamlit Session State')
st.session_state
st.session_state['a_counter']=0
st.session_state.boolean=False

for the_values in st.session_state.keys():
     st.write(the_values)


for the_values in st.session_state.values():
     st.write(the_values)

for the_values in st.session_state.items():
     st.write(the_values)


slider=st.slider('my slider',0,10)
st.session_state.a_counter=slider

st.write('the value of the slider is',st.session_state.a_counter)

st.title('Streamlit Secrets Management')
S=pd.read_csv(r'C:\Users\kutlc\OneDrive\Documents\SQLDB\.streamlit\secrets.toml')
st.write(S)
