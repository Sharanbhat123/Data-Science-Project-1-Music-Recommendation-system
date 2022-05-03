import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
import numpy as np

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

        music_users_features = song_df.pivot(index='Song_id', columns='User_id', values='Listen_count').fillna(0)
        mat_music_users = csr_matrix(music_users_features.values)

        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        model_knn.fit(mat_music_users)


        def get_recommendations(music_name, data=mat_music_users, model=model_knn, n_recommendations=10):
            model.fit(data)
            index = process.extractOne(music_name, song_df['Title'])[2]
            print('Music Selected:', song_df['Title'][index], 'Index:', index)
            print('Searching for recommendation......')
            distances, indices = model.kneighbors(data[index], n_neighbors=n_recommendations)
            for i in indices:
                print(song_df['Title'][i].where(i != index))
            return song_df['Title'][i].where(i != index)


        fn_call = get_recommendations(song_input)
        st.write(fn_call)

    with col2:
        if search_button:
            st.write('These are the respective song links')
            st.write('Please use the links to play the song')


        def get_recommendations(music_name, data=mat_music_users, model=model_knn, n_recommendations=10):
            model.fit(data)
            index = process.extractOne(music_name, song_df['Title'])[2]
            print('Music Selected:', song_df['Title'][index], 'Index:', index)
            print('Searching for recommendation......')
            distances, indices = model.kneighbors(data[index], n_neighbors=n_recommendations)
            for i in indices:
                print(song_df['Album Link'][i].where(i != index))
            return song_df['Album Link'][i].where(i != index)


        fn_call = get_recommendations(song_input)
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



