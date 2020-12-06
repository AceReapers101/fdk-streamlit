import streamlit as st
import os

st.title('1stDayKit')
st.write('A high-level Deep Learning toolkit for solving generic tasks.')

option = st.sidebar.selectbox(
    'Select a usage',
     ['----','Machine Translation','Super Resolution','Object Detection'])

if option == "----":
    st.subheader('Select a function from the sidebar')

elif option == "Machine Translation":

    st.header('Machine Translation')

    st.echo()
    with st.echo():
        from src.core.translate import Translator_M
        from src.core.utils import utils

    st.echo()
    with st.echo():
        Trans = Translator_M(task='Helsinki-NLP/opus-mt-en-ROMANCE')

    st.echo()
    with st.echo():
        text_to_translate = ['>>fr<< this is a sentence in english that we want to translate to french',
                            '>>pt<< This should go to portuguese',
                            '>>es<< And this to Spanish']

        output = Trans.predict(text_to_translate)
        st.subheader('Output')
        st.write(output)

    st.subheader('Translated Text')
    left_column, right_column = st.beta_columns(2)
    left_column.write(text_to_translate)
    right_column.write(output)

elif option == "Super Resolution":
    
    st.header('Super Resolution')

    st.echo()
    with st.echo():
        import torch
        from src.core.super_res import SuperReser
        from src.core.utils import utils
        from PIL import Image
        import cv2

    st.echo()
    with st.echo():
        super_res = SuperReser(name="SuperResssss")

    st.echo()
    with st.echo():
        img = Image.open("src/core/base_libs/ESRGAN/ny/rsz_china-street-changsha-city.jpg")
        st.image(img,width=700)

    st.echo()
    with st.echo():
        img_cv = utils.pil_to_cv2(img)
        output = super_res.predict(img_cv)
        cv2.imwrite('tmp/tempImage.jpg', output)
        st.image('tmp/tempImage.jpg',width=700)
    
    st.subheader('Comparison')
    left_column, right_column = st.beta_columns(2)
    left_column.write("Before")
    left_column.image(img,width=350)
    right_column.write("After")
    right_column.image('tmp/tempImage.jpg',width=350)

elif option == "Object Detection":
    
    st.header('Object Detection')

    st.echo()
    with st.echo():
        import torch
        from src.core.detect import Detector
        from src.core.utils import utils
        from PIL import Image
        import cv2

    st.echo()
    with st.echo():
        det = Detector(name="DemoDet")

    st.echo()
    with st.echo():
        img = Image.open("misc/nyc_superres/out0166.jpg.jpg")
        st.image(img,width=700)

    st.echo()
    with st.echo():
        img_cv = utils.pil_to_cv2(img)
        output = det.predict(img_cv)
        out_img = det.visualize(img_cv,output,figsize=(18,18))
        cv2.imwrite('tmp/tempImage.jpg', out_img)
        st.image('tmp/tempImage.jpg',width=700)