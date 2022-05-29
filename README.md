# SONGFITT - Music Recommendation System
## Submission for Microsoft Intern Engage 2022

# Overview
SONGFITT is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.

Along with the rapid expansion of digital music formats, managing and searching for songs has become signiﬁcant. The purpose of this project is to build a recommendation system to allow users to discover music based on their listening preferences. Therefore in this model I focused on the public opinion to discover and recommend music.

## Features: 
1.	Song Recommendation (minimalistic feature)
	
2.	Recommendation on the basis of Genre and Year of Release (old or new)
	
3.	Depicting the Importance of Acousticness, Loudness, Tempo, Liveness, danceability and valence using Feature Correlation
	
4.	Used Spotify API to play songs on the WebApp

5.	Redirecting to the recommended Songs on the your personal Spotify app with one click.
	
6.	Calculated the Accuracy and Area under curve of each algorithm to see which is best for Prediction and Recommendation
	
7.	Depicted a 3-D earth Model predicting popularity of most famous song(Blinding Lights) in my dataset in different location.

8. Continious update on the deployed site

## Tech Stack and Softwares used
1. `Frontend` : Streamlit
2. `Backend` : Python, scikit-learn, pandas, Numpy and Plotly.
3. `ML model` : Jupyter Notebook
4. `IDE` : VsCode
6. `Version Control` : Git
7. `Deployment` : Streamlit-Share

## Screenshots
### App UI
![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/screenshot1.png)

### 3-D Model Depicting Popularity of the Most popular song in my dataset by location

![Screenshot (107)](https://github.com/uditkatyal/songfitt_/blob/main/images/screenshot2.png)

## Installation/Environment Setup
1. Clone this repository in your local system.
* Open terminal in a new folder and enter the command given below.
   ```
   git clone https://github.com/uditkatyal/songfitt_
   ```

2. Make sure that Python is installed and updated in your machine.

3. Install dependencies.
* Open terminal in the cloned folder and enter the command given below.
   ```
   pip install -r requirements.txt
   ```
  
4. Run the project.
* Write the following command in terminal to run the website locally. 
   ```
   streamlit run app.py
   ```
   
5. If everything is done in order, the app will be running at "http://localhost:8501/"

## If You want to run the ML Models
- Do add the SpotGenTrack file in the main folder 
- Zip file of SpotGenTrack is Attached in the folder 
- This is how file structure should look like-
- C:\Users\UditKatyal\Desktop\songfitt_\SpotGenTrack

![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/file_structure.png)
![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/sub_files_1.png)






##  Proposed methodology 
  In this Project I have used K Nearest Neighbours and Random Forest Machine Learning Algorithms.
  The model is initially trained with dataset provided and then recommends the songs as per user input.
  
  ![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/workflow_model.png)


## Dataset Link
- SpotifyGenTrack - https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset
- Music Trend Analysis (by location) - https://www.kaggle.com/code/hkapoor/music-trends-analysis-by-location/data

## Data cleaning and pre-processing:
- read playlists/track info from json files
- extract audio features for each track
- optimise the features 

## Research Papers 
- https://www.researchgate.net/publication/277714802_A_Survey_of_Music_Recommendation_Systems_and_Future_Perspectives
- https://www.researchgate.net/publication/324652918_Recommendation_of_Job_Offers_Using_Random_Forests_and_Support_Vector_Machines


## Video Demo
The demo video for the Web Application: [SONGFITT | Video](https://www.youtube.com/watch?v=hLn_rFlkQME)

## Conclusion and Future Scope

- In this project, I have presented a novel framework for Music recommendation that is driven by data
and simple effective recommendation system for generating songs as per users choice.
- Moving forward, I will use a larger Spotify database by using the Spotify API to collect my own data, and explore different algorithms to predict popularity score rather than doing binary classification.

Thank you, Microsoft and Acehacker Team for such an amazing program ❤️
