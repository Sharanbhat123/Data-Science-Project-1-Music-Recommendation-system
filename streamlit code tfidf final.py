import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

rad=st.sidebar.radio('Choose',['Home', 'About'])


song_df_1 = pd.read_csv('triplets_file.csv')
song_df_1.rename(columns={'User id':'User_id'},inplace=True)
song_df_2 = pd.read_csv('song_data.csv')
song_df=pd.merge(song_df_1,song_df_2, on='Song_id', how='left')
song_df['song']=song_df['Title']+'-'+song_df['Artist_name']

if rad=='Home':

    st.title('Music Recommendation system: Relaxation and Meditation')
    song_input = st.text_input("Whats  Your  Favorite  Song ?")
    search_button = st.button("Search")

    col1, col2=st.columns(2)
    with col1:
        if search_button:
            st.write('Music Selected: "', song_input, '"\n')
            st.write('Searching for recommendation......')

            tfv = TfidfVectorizer(max_df=3, max_features=None, strip_accents='unicode', token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3), stop_words='english')
            song_df['song'] = song_df['song'].fillna('')
            tfv_matrix = tfv.fit_transform(song_df['song'])
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
            indices = pd.Series(song_df.index, index=song_df['Title']).drop_duplicates()


            def give_rec(Title, sig=sig):
                # get the index corresponding to original title
                idx = indices[Title]

                # get the pairwise similarity scores
                sig_scores = list(enumerate(sig[idx]))

                # sort the title
                sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

                # scores of the 20 most similar song titles
                sig_scores = sig_scores[1:21]

                # title indices
                song_indices = [i[0] for i in sig_scores]

                # top 10 most similar movies
                return song_df['Title'].iloc[song_indices]


            fn_call = give_rec(song_input)
            st.write(fn_call)

    with col2:
        if search_button:
            st.write('These are the respective song links')
            st.write('Please use the links to play the song')

            tfv = TfidfVectorizer(max_df=3, max_features=None, strip_accents='unicode', token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3), stop_words='english')
            song_df['song'] = song_df['song'].fillna('')
            tfv_matrix = tfv.fit_transform(song_df['song'])
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
            indices = pd.Series(song_df.index, index=song_df['Title']).drop_duplicates()


            def give_rec(Title, sig=sig):
                # get the index corresponding to original title
                idx = indices[Title]

                # get the pairwise similarity scores
                sig_scores = list(enumerate(sig[idx]))

                # sort the title
                sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

                # scores of the 20 most similar song titles
                sig_scores = sig_scores[1:21]

                # title indices
                song_indices = [i[0] for i in sig_scores]

                # top 10 most similar movies
                return song_df['Album Link'].iloc[song_indices]


            fn_call = give_rec(song_input)
            st.write(fn_call)

if rad=='About':
    st.title("Group Members")
    st.write("J Rani")
    st.write("Lakshmi Tejaswi Akella")
    st.write("Rushikesh Ram Malwade")
    st.write("Sharan S")
    st.write("Sudharani G")
    st.write("Sukla Pattnaik")
    st.write("Sathwik J")