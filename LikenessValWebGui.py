import streamlit as st
import clip
import torch
import numpy as np
from PIL import Image

bg = '''
<style>
.stApp {
  background: radial-gradient(50% 50% at 50% 50%,#9ae0f6 0,#e6e6e6 100%);
}
</style>
'''

likenessArr = np.load('LikenessArray.npy')

st.markdown(bg, unsafe_allow_html=True)


header = st.container()
body = st.container()
imagebox = st.empty()

likenesseval = st.empty()
likenesseval.markdown("<br><br><br><br><br><br><br><br>Current Likeness: ", unsafe_allow_html = True)

@st.cache(ttl=60)
def update_image():
    imagebox.image(picture)
    image1 = Image.open(picture)
    with torch.no_grad():
        image = preprocess(image1).unsqueeze(0).to(device)
        image_features = model.encode_image(image).numpy()
    likenesseval.write("Current Likeness: "+ str(round(likeness_score(image_features)[0],4)))

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache
def load_model():
  return clip.load("ViT-B/32", device=device)

model, preprocess = load_model()

def sigmoid(x):
    return 1/(np.exp(-x) + 1)

def likeness_score(embed):
    return sigmoid(embed @ likenessArr + 3.1309943199157715)


with header:
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        st.title("Likeness Demo")
    with col2:
        st.write('')
        st.write('')
        st.image("microsoft_logo.png")
        
    with col3:
        st.write('')
        st.write('')
        st.image("cambridge_logo.png")

with body:
    #global picture
    picture = st.file_uploader(label="Upload Image Here:", type=['png', 'jpg', 'tiff'])
    if picture is not None:
        update_image()
