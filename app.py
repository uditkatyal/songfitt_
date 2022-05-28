import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from load_css import local_css
from PIL import Image
import pydeck as pdk
import plotly.figure_factory as ff
import base64
import streamlit.components.v1 as components


local_css("style.css")



def spr_sidebar():
    with st.sidebar:
        # st.image(SPR_SPOTIFY_URL, width=60)
        st.info('**SONGFITT**')
        home_button = st.button("Home")
        data_button = st.button("About Dataset")
        rec_button = st.button('Recommendations')
        trends_button = st.button('Features')
        conc_button = st.button('Conclusions')
        blog_button = st.button('My 4 weeks Progress Report')
        st.success('By Udit Katyal')
        st.session_state.log_holder = st.empty()
        # log_output('None')
        if home_button:
            st.session_state.app_mode = 'home'
        if data_button:
            st.session_state.app_mode = 'dataset'
        if trends_button:
            st.session_state.app_mode = 'trends'
        if rec_button:
            st.session_state.app_mode = 'recommend'
        if conc_button:
            st.session_state.app_mode = 'conclusions'
        if blog_button:
            st.session_state.app_mode = 'blog'


def dataset_page():
    st.markdown("<br>", unsafe_allow_html=True)
    """
    # Spotify Gen Track Dataset
    -----------------------------------
    Here I am using Spotity Gen Track Dataset, this dataset contains n number of songs and some metadata is included as well such as name of the playlist, duration, number of songs, number of artists, etc.
    """

    dataset_contains = Image.open('images/dataset_contains.png')
    st.image(dataset_contains, width =900)

    """
   
    - The data has three files namely spotify_albums, spotify_artists and spotify_tracks from which I am extracting songs information.
    - There is spotify features.csv files which contains all required features of the songs that I am using. 
    - Does not have an offensive title

    """
    """
    # Enhancing the data:
    These are some of the features that are available to us for each song and I am going to use them to enhance our dataset and to help matching
    the user's favorite song as per his/her input.

    ### These features value is measured mostly in a scale of 0-1:
    - **acousticness:** Confidence measure from 0.0 to 1.0 on if a track is acoustic.
    - **danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo,
    rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
    - **energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically,
    energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.
    Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
    - **instrumentalness:** Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or
    spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
    Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
    - **liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability
    that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
    - **loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful
    for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical
    strength (amplitude). Values typical range between -60 and 0 db.
    - **speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording
    (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably
    made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in
    sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
    - **tempo:** The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the
    speed or pace of a given piece and derives directly from the average beat duration.
    - **valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound
    more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

    Refered docs: [link](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """

    st.markdown("<br>", unsafe_allow_html=True)
    '''
    # Final Dataset 
    '''
    '''
    - Enhancing data
    '''
    dataframe1 = pd.read_csv('filtered_track_df.csv')
    st.dataframe(dataframe1)
    st.markdown("<br>", unsafe_allow_html=True)

    # st.subheader('Total VS New Tracks in each json file')
    # st.plotly_chart(get_num_tracks_fig('total'), use_container_width=True)
    # st.subheader('Existing VS New Tracks in each json file')
    # st.plotly_chart(get_num_tracks_fig('existing'), use_container_width=True)


def spr_footer():
    st.markdown('---')
    st.markdown(
        '© Copyright 2021 - SongFitt By Udit Katyal')


def blog_page():
    st.markdown("<br>", unsafe_allow_html=True)

    # base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="1000" type="application/pdf"></iframe>'
        
    # st.markdown(pdf_display, unsafe_allow_html=True)

    def show_pdf(file_path):
        with open(file_path,"rb") as f:
          base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    show_pdf("4_weeks_progress_report.pdf")

    # def st_display_pdf(pdf_file):
    #      with open(pdf_file,"rb") as f:
    #          base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    #      pdf_display = f'<embed src=”images/abc.pdf;base64,{base64_pdf}” width=”700″ height=”1000″ type=”application/pdf”>’
    #      st.markdown(pdf_display, unsafe_allow_html = True)


        
   

    
# @st.cache(suppress_st_warning=True)


def trends_page():
    st.header("Features")
    st.subheader("Song Popularity Prediction")
    st.markdown(
        'On the basis of the features that we have in our dataframe, I will try to predict the Popularity of Songs.')

    popularity_distrubution = Image.open(
        'images/popularity_distribution.png') 

    st.image(popularity_distrubution,
             caption='Popularity Distribution', width=500)       

    
    st.write("Popularity based on Time Signature")
    st.code("Indicates how many beats are in each measure of a piece of music")   
    st.code("4/4 has four quarter note beats and 3/4 has three quarter note beats likewise")  

    st.write("#")

    st.write("Popularity based on key")
    st.code("Key is the major or minor scale around which a piece of music revolves")   
    st.code("12 keys C#, C, D#, D, E, F#, F, G#, G, A#, A, B.")  

    st.write("#")

    st.write("Popularity based on Mode")
    st.code("Corresponds to melodic and harmonic behaviors of music")   
    st.code("Major and Minor Modes") 

    st.write("#")

    st.write("Popularity based on Mode and Key")
    st.code("Popularity of songs collectively")   
     
        #  Popularity based on Key and Mode

    # t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"
    # st.markdown(t, unsafe_allow_html=True)
    
    # genre = st.radio(
    #  "What's your favorite movie genre",
    #  ('Comedy', 'Drama', 'Documentary'))  
    

    popularity_timesignature = Image.open(
        'images/popularity_timesignature.png')

    popularity_key = Image.open(
        'images/popularity_key.png')

    popularity_mode = Image.open(
        'images/popularity_mode.png')

    popularity_key_mode = Image.open(
        'images/popularity_key_mode.png')

   
    col1, col2 = st.columns(2)
    with col1:
        st.image(popularity_timesignature,
                 caption='Popularity Based on Time Signatures', width=400)
    with col2:
        st.image(popularity_key,
                 caption='Popularity Based on Key', width=400)

    col1, col2 = st.columns(2)
    with col1:
        st.image(popularity_mode,
                 caption='Popularity Based on Mode', width=400)
    with col2:
        st.image(popularity_key_mode,
                 caption='Popularity Based on Key and Mode', width=400)

    st.subheader('Feature Correlation')
    st.markdown('**Feature Correlation** talks about dependencies between features. It is a way to understand how features are correlated with each other.')
    st.markdown(
        'So here value **1** depicts that the features are highly correlated with each other.')
    st.markdown(
        'From here we can depict that **ENERGY** and **LOUNDNESS** are much related to eachother with value of **0.82**')
    feature_correlation = Image.open(
        'images/feature_corelation.png')
    st.image(feature_correlation, caption='Feature Correlation', width=900)

    st.header("Algorithms")
    st.subheader("Linear Regression")
    code = '''LR_Model = LogisticRegression()
LR_Model.fit(X_train, y_train)
LR_Predict = LR_Model.predict(X_valid)
LR_Accuracy = accuracy_score(y_valid, LR_Predict)
print("Accuracy: " + str(LR_Accuracy))

LR_AUC = roc_auc_score(y_valid, LR_Predict)
print("AUC: " + str(LR_AUC))

Accuracy: 0.7497945543198379
AUC: 0.5'''
    st.code(code, language='python')
   
    st.subheader("K-Nearest Neighbors Classifier")
    code = '''KNN_Model = KNeighborsClassifier()
KNN_Model.fit(X_train, y_train)
KNN_Predict = KNN_Model.predict(X_valid)
KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
print("Accuracy: " + str(KNN_Accuracy))

KNN_AUC = roc_auc_score(y_valid, KNN_Predict)
print("AUC: " + str(KNN_AUC))

Accuracy: 0.7763381361967896
AUC: 0.6890904291795135'''
    st.code(code, language='python')

    st.subheader("Decision Tree Classifier")
    code = '''DT_Model = DecisionTreeClassifier()
DT_Model.fit(X_train, y_train)
DT_Predict = DT_Model.predict(X_valid)
DT_Accuracy = accuracy_score(y_valid, DT_Predict)
print("Accuracy: " + str(DT_Accuracy))

DT_AUC = roc_auc_score(y_valid, DT_Predict)
print("AUC: " + str(DT_AUC))

Accuracy: 0.8742672437407549
AUC: 0.8573960839474465
'''

    st.code(code, language='python')
    st.subheader("Linear Support Vector Classification")
    code = '''
    training_LSVC = training.sample(10000)
X_train_LSVC = training_LSVC[features]
y_train_LSVC = training_LSVC['popularity']
X_test_LSVC = dataframe.drop(training_LSVC.index)[features]
X_train_LSVC, X_valid_LSVC, y_train_LSVC, y_valid_LSVC = train_test_split(
    X_train_LSVC, y_train_LSVC, test_size = 0.2, random_state = 420)

LSVC_Model = DecisionTreeClassifier()
LSVC_Model.fit(X_train_LSVC, y_train_LSVC)
LSVC_Predict = LSVC_Model.predict(X_valid_LSVC)
LSVC_Accuracy = accuracy_score(y_valid_LSVC, LSVC_Predict)
print("Accuracy: " + str(LSVC_Accuracy))

LSVC_AUC = roc_auc_score(y_valid_LSVC, LSVC_Predict)
print("AUC: " + str(LSVC_AUC))

Accuracy: 0.6815
AUC: 0.5881201752103391
'''

    st.code(code, language='python')
    st.subheader("Random Forest")
    code = '''RFC_Model = RandomForestClassifier()
RFC_Model.fit(X_train, y_train)
RFC_Predict = RFC_Model.predict(X_valid)
RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy))

RFC_AUC = roc_auc_score(y_valid, RFC_Predict)
print("AUC: " + str(RFC_AUC))

Accuracy: 0.9357365912452748
AUC: 0.879274665020435'''
    st.code(code, language='python')

    st.header("Music Trends Analysis By Location")
    '''
    For Input purpose I took the most listened song from my dataset **Blinding Lights**  and predicted it's popularity score
    
    '''

    top_10_tracks = Image.open("images/top_tracks.png")
    st.image(top_10_tracks , caption ="Top 1o Tracks", width = 800)
    
    st.code("Popularity ranges from 0 - 100 but to make visible on map 1 Unit = 1000 Unit, so 32200 score = 32.2 popularity")
    components.iframe("http://threegraphs.com/charts/preview/9036/embed/", width = 1000, height = 700)

   
 

# @st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(
        "filtered_track_df.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios



def rec_page():
    
    st.header("RECOMMENDATION ENGINE")
   
    with st.container():
        col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        acousticness = st.slider(
            'Acousticness',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)
        tracks_per_page = 12
        test_feat = [acousticness, danceability,
                     energy, instrumentalness, valence, tempo]
        uris, audios = n_neighbors_uri_audio(
            genre, start_year, end_year, test_feat)
        tracks = []
        for uri in uris:
            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
                uri)
            tracks.append(track)
    # st.write(tracks)
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [
            genre, start_year, end_year] + test_feat
    current_inputs = [genre, start_year, end_year] + test_feat

    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page

    current_tracks = tracks[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i % 2 == 0:
                with col1:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

            else:
                with col3:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

    else:
        st.write("No songs left to recommend")

    st.subheader("Algorithms that I have used in this filtering are the k-nearest neighbours and Random Forest")

   
    # Plot!
    random_forest_audio_importance = Image.open('images/random_forest_audio_importance_feature.jpg')
    st.image(random_forest_audio_importance, caption ="random_forest_audio_feature_importance", width = 900)

   


def home_page():
    st.header('Welcome to SONGFITT (SS)')
    st.markdown('---')
    st.write(
        'Hi Microsoft, I am Udit Katyal and this is my webApp for Microsoft engage 2022 Program')
    st.write('Check this demo video to see how to use this Web App:')

    st.subheader('About Me')

    st.subheader('Udit Katyal')
    image = Image.open(
        'images/img1.jpg')
    st.image(image, caption='Udit Katyal', width=200)

    st.write("Check out my [Github Repository](https://github.com/uditkatyal/songfitt_)")
    st.write('Hi I am Udit Katyal, a sophomore pursuing Btech IT from Akhilesh Das Gupta Institute of Technology and Management, New Delhi India. ')


def conclusions_page():
    st.header('Conclusions')
    
    '''
    #### Problems I ran through
    '''
    '''
    - Data Collection -- Even though the core dataset I used was provided by Spotify, it was needed to go and look for other data sources to enhance the data and combine it with the core data set. 

    - Efficient Data Processing -- Initially I decided to take complete SpotifyGenTrack dataset (approximately 773.84 MB of data). But it became very important to enhance the data quality unless the recommendation might run to unnecessary bugs and outcomes. Therefore after many attempts, generated a filteredTrack csv which had the exact and refined dataset that was required for recommendation and predicting popularity of songs.

    - Unsupervised Learning -- Decided to take an different approach where I Explored different families of cluster algorithms and learning about advantages and disadvantages to make the best selection as well as deciding which measure distance makes the most sense for our purposes.

    '''
    st.success("Performed the accuracy tests and these are the results received")

    st.header("Model Perfomance Summary")

    algo_accuracy = Image.open(
        'images/algos_accuracy.png')
    st.image(algo_accuracy, width=400)
    st.write('Using a dataset of 228, 000 Spotify Tracks, I was able to predict popularity(greater than 57 popularity) using audio-based metrics such as key, mode, and danceability without external metrics such as artist name, genre, and release date. The Random Forest Classifier was the best performing algorithm with 92.0 % accuracy and 86.4 % AUC. The Decision Tree Classifier was the second best performing algorithm with 87.5 % accuracy and 85.8 % AUC.')

    st.write('Moving forward, I will use a larger Spotify database by using the Spotify API to collect my own data, and explore different algorithms to predict popularity score rather than doing binary classification.')

    algo_auc = Image.open(
        'images/models_auc_area_under_curve.png')
    st.image(algo_auc, width=400)
    st.write("The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classess. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.")
     
    st.subheader("Music Trend Analysis Conclusion")
    '''
    - Dataset is imbalanced with more Europian countries.
    - Few songs have managed to make in 96% of Top Charts of all countries.
    - Even though few artists have many occurances in the Top charts, they don't have any song in Top 10 tracks occurances.
    - Average song duration preferred by most is around 3:00 minutes to 3:20 minutes
    - People in Asian countries prefer longer song duration. Europian countries mostly listen to songs close to or less than 3 minutes.
    - Asian countries prefer less of explicit songs, only 18%, compared to world average of 35%. "Global Top 50 Chart" has 23 explicit songs.
    '''
    



st.session_state.app_mode = 'recommend'

# @st.cache()
def main():
    
    spr_sidebar()
    st.header("SONGFITT (SS)")
    st.markdown(
        '**SONGFITT** is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.') 

    cover_image = Image.open("images/cover.jpg")
    st.image(cover_image , width = 600)     
    # st.code("<- Go to the Home Button ")
    if st.session_state.app_mode == 'dataset':
        dataset_page()

    if st.session_state.app_mode == 'trends':
        trends_page()

    if st.session_state.app_mode == 'recommend':
        # if 'spr' not in st.session_state:
        #  st.error('Please select an option in User Input page')
        rec_page()

        with st.form(key="form"):
            files = st.file_uploader("Files", accept_multiple_files=True)
            submit_button = st.form_submit_button(label="Submit choice")

        if submit_button:
            if files:
             st.markdown("You chose the files {}".format(", ".join([f.name for f in files])))
            else:
             st.markdown("You did not choose any file but clicked on 'Submit choice' anyway")
        else:
          st.markdown("You did not click on submit button.")
        

    if st.session_state.app_mode == 'blog':
        blog_page()

    if st.session_state.app_mode == 'conclusions':
        conclusions_page()

    if st.session_state.app_mode == 'home':
        home_page()

    spr_footer()


# Run main()
if __name__ == '__main__':
    main()


# # HOME PAGE
