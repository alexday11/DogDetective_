import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from keras.applications.vgg16 import preprocess_input

st.set_page_config(
    page_title="DogDetective",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded",
)

my_model = load_model('./Data/best_model.h5')
labels = pickle.load(open('./Data/labels.pkl','rb'))
data = pickle.load(open('./Data/url.pkl','rb'))
#st.selectbox('What Would you like spicies dogs',data['Species'].values)



def predicts(img_fname):
  img = load_img(img_fname, target_size=(224,224))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  img  = preprocess_input(img)
  pred = my_model.predict(img)
  pred_cls = labels[np.argmax(pred,-1)[0]]
  return pred_cls

def recommend(feature,name_species):
   ref_index = data[data['species']==name_species].index[0]
   url = data[feature].iloc[ref_index]
   return url

def process_img(filename,resize=(334,500)):
    img = Image.open(filename)
    img = img.resize(resize)
    return img

def create_columns2():
   col1, col2, col3 = st.columns(3)

   with col1:
      image1 = process_img(img1)
      st.image(image1)
      #st.markdown("[![Clickable Image](./Image2/n02085620-Chihuahua/n02085620_588.jpg)](https://google.com)")
      st.link_button(f'วิธีดูแลสุนัขพันธุ์: {names}',link_url)
   with col2:
      image2 = process_img(img2)
      st.image(image2)
      st.link_button(f'ผลิตภัณฑ์ที่เหมาะสำหรับ: {names} วัยโต',link_url_l)
   with col3:
      image3 = process_img(img3)
      st.image(image3)
      st.link_button(f'ผลิตภัณฑ์ที่เหมาะสำหรับ: {names} ลูกสุนัข',link_url_s)

#Image 
st.image('img_header2.png',use_column_width=True)
st.title('อัปโหลดรูปน้องสุนัขของคุณได้เลย :point_down:')

# Create a file upload button for jpg and png files
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#test_img = './Image2/n02085620-Chihuahua/n02085620_588.jpg'
#test_img2 = 'https://static.streamlit.io/examples/owl.jpg'

# Display the uploaded image

if uploaded_image is not None:
   names = predicts(uploaded_image)
   link_url = recommend('url',names)
   link_url_l = recommend('url_l',names)
   link_url_s = recommend('url_s',names)
   link_img = recommend('image',names)
   specific = recommend('specific',names)
   care = recommend('care',names)
   note = recommend('note',names)
   col = st.columns(2)
   img1 = recommend('img_1',names)
   img2 = recommend('img_2',names)
   img3 = recommend('img_3',names)
   with col[0]:
      st.header("สุนัขพันธุ์: {}".format(names))
      st.image(uploaded_image)
   with col[1]:
      st.header('ข้อมูลเบื้องต้นสำหรับน้องสุนัขพันธุ์: {}'.format(names))
      st.subheader('ข้อมูลจำเพาะ',divider='rainbow')
      st.markdown(specific)
      st.subheader('วิธีการดูแล',divider='rainbow')
      st.markdown(care)
      st.subheader('ข้อควรระวัง',divider='rainbow')
      st.markdown(note)





if st.button('ดูข้อมูลเพิ่มเติม'):
  st.subheader('สุขภาพและความเป็นอยู่ที่ดีของสุนัข')
  create_columns2()
  #st.markdown("[![Clickable Image](https://static.streamlit.io/examples/owl.jpg)](https://google.com)")

def create_columns3():
   col1, col2, col3 = st.columns(3)

   with col1:
      image1 = process_img(img_box1)
      st.image(image1)
      #st.markdown("[![Clickable Image](./Image2/n02085620-Chihuahua/n02085620_588.jpg)](https://google.com)")
      st.link_button(f'วิธีดูแลสุนัขพันธุ์: {name_boxes}',link_url_box)
   with col2:
      image2 = process_img(img_box2)
      st.image(image2)
      st.link_button(f'ผลิตภัณฑ์ที่เหมาะสำหรับ: {name_boxes} วัยโต',link_url_box_l)
   with col3:
      image3 = process_img(img_box3)
      st.image(image3)
      st.link_button(f'ผลิตภัณฑ์ที่เหมาะสำหรับ: {name_boxes} ลูกสุนัข',link_url_box_s)



st.header('สุนัขสายพันธุ์ต่างๆ :guide_dog:',divider='gray')
name_boxes = st.selectbox('What would you recommend',data['species'].values)
if st.button('Select Species'):
   col = st.columns(2)
   link_url_box = recommend('url',name_boxes)
   link_url_box_l = recommend('url_l',name_boxes)
   link_url_box_s = recommend('url_s',name_boxes)
   img_species = recommend('image',name_boxes)
   specific_2 = recommend('specific',name_boxes)
   care_2 = recommend('care',name_boxes)
   note_2 = recommend('note',name_boxes)
   img_box1 = recommend('img_1',name_boxes)
   img_box2 = recommend('img_2',name_boxes)
   img_box3 = recommend('img_3',name_boxes)
   with col[0]:
      st.header("สุนัขพันธุ์: {}".format(name_boxes))
      st.image(img_species)
   with col[1]:
      st.header('ข้อมูลเบื้องต้นสำหรับน้องสุนัขพันธุ์: {}'.format(name_boxes))
      st.subheader('ข้อมูลจำเพาะ',divider='rainbow')
      st.markdown(specific_2)
      st.subheader('วิธีการดูแล',divider='rainbow')
      st.markdown(care_2)
      st.subheader('ข้อควรระวัง',divider='rainbow')
      st.markdown(note_2)
   st.subheader('สุขภาพและความเป็นอยู่ที่ดีของสุนัข')
   create_columns3()






