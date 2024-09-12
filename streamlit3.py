import streamlit as st
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
