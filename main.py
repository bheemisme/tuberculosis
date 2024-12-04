import streamlit as st

from PIL import Image
from model import tb_call

def uploader():
    uploaded_file = st.file_uploader(
                    label='Upload the image', 
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=False,
                    key='image-uploader'
                )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        classname, prob = tb_call(image)
        st.markdown(f'#### Class: {classname}')
        st.markdown(f'#### Probability: {prob}')


def app():

    st.title('Tuberculosis detection')
    
    uploader()


if __name__ == '__main__':
    app()
    