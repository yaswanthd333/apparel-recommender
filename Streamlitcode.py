import streamlit as st
import numpy as np
import pandas as pd
import warnings
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words) 


def tfidf_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    k=[]
    for i in range(1,len(indices)):
        k.append([data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]]])
    return k



html_temp = """
  <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Apparel Recommendation System</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)


data = pd.read_pickle('16k_apparel_data_preprocessed')


t=st.text_input("","")
#df = pd.DataFrame(1,{'asin brand':'','color':'','medium_image_url':'','product_type_name':'','title':t,'formatted_price':''})
df=['','','','','',t,'']
data.loc[-1]=df


tfidf_title_vectorizer = TfidfVectorizer(min_df=1)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])



k=tfidf_model(-1, 23)



m = st.markdown(""" <style> 
div.stButton > 
    button {
  appearance: none;
  background-color: #000000;
  border: 2px solid #1A1A1A;
  border-radius: 15px;
  box-sizing: border-box;
  color: #FFFFFF;
  cursor: pointer;
  font-family: Roobert,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol";
  font-size: 16px;
  font-weight: 600;
  line-height: normal;
  margin: 0;
  min-height: 60px;
  min-width: 0;
  display:block;
  margin:auto;
  outline: none;
  padding: 16px 24px;
  text-align: center;
  text-decoration: none;
  transition: all 300ms cubic-bezier(.23, 1, 0.32, 1);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  will-change: transform;
}

button : disabled {
  pointer-events: none;
}

button :hover {
  box-shadow: rgba(0, 0, 0, 0.25) 0 8px 15px;
  transform: translateY(-2px);
}

button:active {
  box-shadow: none;
  transform: translateY(0);
}

</style>""", unsafe_allow_html=True)
stat =st.button("Show me the apparel")
st.write("\n")
st.write("\n")
st.markdown("""<style>
cols{
  backdrop-filter: blur(10px);
  text-align:center;
  justify-content:cente;
  margin-auto:center;
}
div.stImage>img{
    display: flex;
    justify-content: center;
}

</style>""",unsafe_allow_html=True)
if stat==True:
      for i in range(1,len(k),3):
        cols=st.columns(3)
        cols[0].image(k[i][1])
        cols[0].write(k[i][0])
        cols[1].image(k[i+1][1])
        cols[1].write(k[i+1][0])
        cols[2].image(k[i+2][1])
        cols[2].write(k[i+2][0])
        #st.image(k[i][1])
        #st.write(k[i][0])