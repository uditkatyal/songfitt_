# songfitt_
SONGFITT is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.

Along with the rapid expansion of digital music formats, managing and searching for songs has become signiﬁcant. The purpose of this project is to build a recommendation system to allow users to discover music based on their listening preferences. Therefore in this model I focused on the public opinion to discover and recommend music.

## Installation

Use pip to install the requirements.

~~~bash
pip install -r requirements.txt
~~~

## Usage

To run the web server, simply execute streamlit with the app.py file:

```bash
streamlit run app.py
```

##  Proposed methodology 
  In this Project I have used K Nearest Neighbours and Random Forest Machine Learning Algorithms.
  The model is initially trained with dataset provided and then recommends the songs as per user input.
  
  ![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/workflow_model.png)
## Screenshots
### App UI
![Screenshot (105)](https://github.com/uditkatyal/songfitt_/blob/main/images/screenshot1.png)

### 3-D Model Depicting Popularity of the Most popular song in my dataset by location

![Screenshot (107)](https://github.com/uditkatyal/songfitt_/blob/main/images/screenshot2.png)


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


## Built With

- [streamlit]() - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.
- [pandas]() - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- [Pillow]() - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
- [scikit-learn]() - Scikit-learn is a free software machine learning library for the Python programming language.
- [Plotly]() - An open-source, interactive data visualization library for Python.
- [Numpy]() - NumPy is the fundamental package for array computing with Python.
- [Three Graphs]() - 3-D Charts for the web.

## Conclusion

In this project, I have presented a novel framework for Music recommendation that is driven by data
and simple effective recommendation system for generating songs as per users choice.
