
"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
# Data dependencies
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import spacy
import string
import re
import os



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

st.set_option('deprecation.showPyplotGlobalUse', False)


# Load your raw data
raw = pd.read_csv("resources/train.csv")
train_df = raw



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	html_temp = """
	<div style="background-color:{};padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:{};text-align:center;">EDSA 2020:Climate Change Belief Analysis</h1>
	</div>
	"""

	title_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:white;text-align:center;">CLASSIFICATION PREDICT</h1>
	<h2 style="color:white;text-align:center;">TEAM:4</h2>
	<h3 style="color:white;text-align:right;">2020/11/18</h3>
	</div>
	"""
	

	# Creates a main title and subheader on your page -
	# these are static across all pages


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	menu = ["Home", "About The Predict", "Explanatory Design Analysis", "Predictions","Model Performance"]
	selection = st.sidebar.selectbox("Menu", menu)

	if selection == "Home":
		st.markdown(html_temp.format('royalblue','white'), unsafe_allow_html=True)
		st.markdown(title_temp, unsafe_allow_html=True)


	# Building out the "About" page
	if selection == "About The Predict":
		markup(selection)
		st.info("Project Oveview : Predict an individual’s belief in climate change based on historical tweet data")
		
		# You can read a markdown file from supporting resources folder
		if st.checkbox("Introduction"):
			st.subheader("Introduction to Classification Predict")
			st.info("""Many companies are built around lessening one’s environmental impact or carbon footprint.They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals.
			 They would like to determine how people perceive climate change and whether or not they believe it is a real threat.
			 This would add to their market research efforts in gauging how their product/service may be received.We are creating  a Machine Learning Classifier models based on multiple Classifiers algorithhms including Logistic Regression,
			 Decision Tree classifier, Xgboost classifier, Lightxboost classifiers, Linear SVC classifier and etc.The models should classify the text into
			 whether the person believes in climate change  or not or based on their news tweet""")

		
		if st.checkbox("Problem Statement"):
			st.subheader("Problem Statement of the Classification Predict")
			st.info("Build a Natural Language Processing models to classify whether or not a person believes in climate change or based on their novel tweet data")

		if st.checkbox("Data"):
			st.subheader("Data Decription")
			st.info("""For this exercise we will be using the data that was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch,
			 University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:
			 
>2- News: the tweet links to factual news about the climate change

>1 - Pro: the tweet support the believe of man made climate change

>0 - Neutral: the tweet neither supports nor refutes the believe of manmade climate change

>1 - Anti: the tweet does not believe in man made climate change

Variable definitions:
>sentiment: (Sentiment of tweet)
>message: (Tweet body)
>tweetid: (Twitter unique id)""")

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page



	# Building out the predication page
	if selection == "Predictions":
		markup(selection)
		option = st.selectbox('Select a model',['LinearSVC','Decision Tree','Random Forest','KNeighbors Classifiers','Light gbm','XGBoost','Logistic Regression'])
		if option == 'LinearSVC':
			st.info('Linear SVC - it fits to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is.')
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LinearSVC_model.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'Decision Tree':
			st.info("Decision Tree - A machine learning model that operates by partitioning data into smaller subsets. It considers all possible binary data splits and selects the data split with the best separation of the data")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/DecisionTreeClassifier.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'Random Forest':
			st.info("Random Forest - An ensemble machine learning method that operates by constructing multiple decision trees at training time and outputting the mode classifications for the individual trees")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Random_Forest.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'KNeighbors Classifiers':
			st.info("KNeighbors Classifiers - implements classification based on voting by nearest k-neighbors of target point, t, while RadiusNeighborsClassifier implements classification based on all neighborhood points within a fixed radius, r, of target point, t")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/KNeighborsClassifier.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'Light gbm':
			st.info("Light gbm -  is a gradient boosting framework that uses tree based learning algorithm. It grows tree vertically while other algorithm grows trees horizontally.")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/lightgbmClassifier.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'XGBoost':
			st.info("XGBoost - An ensemble machine learning model technique that uses gradient boosting framework for machine learning.")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/XgboostClassifier.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		if option == 'Logistic Regression':
			st.info("Logistic Regression - A machine learning method that computes the probability of an event occuring and places it in the relevant class or category")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

	







	# Building out the EDA page
	if selection == "Explanatory Design Analysis":
		markup(selection)
		train_df = raw


		#plot and visualize the most used words in wordcloud
		if st.checkbox("word cloud"):
			st.subheader("Most used words per each sentiment")
			st.image('resources/wrdc.png', channels="BGR")

			st.subheader("Most used words per each sentiment with clean data")
			st.image('resources/wrdclean.png', channels="BGR")


		if st.checkbox("count of words per each sentiment"):
			st.subheader("count of words per each sentiment with clean data")
			st.image('resources/observations_sent.png', channels="BGR")

			st.subheader("count of words per each sentiment with resampled data")
			st.image('resources/classimb.png', channels="BGR")



		if st.checkbox("Punctuation in messages "):
			st.subheader("number of punctuations in messages per sentiment")
			st.image('resources/punc.png', channels="BGR")


		if st.checkbox("handles of the most tweets"):
			st.subheader("handles of the tweets per message(news)")
			st.image('resources/hand_news.png', channels="BGR")
			

		if st.checkbox("frequent words used most in tweets"):
			st.subheader("frequent words used most in tweets(news)")
			st.image('resources/words_news.png', channels="BGR")



		if st.checkbox("frequent # used most in tweets"):
			st.subheader("frequent # used most in tweets(news)")
			st.image('resources/#_news.png', channels="BGR")



		if st.checkbox("metrics"):
			st.subheader('plot the PCC figure')
			corr = train_df.corr(method = 'pearson')
			f, ax = plt.subplots(figsize=(11, 9))
			cmap = sns.diverging_palette(10, 275, as_cmap=True)
			sns.heatmap(corr, cmap=cmap, square=True,
			linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
			st.pyplot()


	# Building out the "Model Perfomance" page
	if selection == "Model Performance":
		markup(selection)
		st.subheader('Summary of all our models perfomance')
		
		
		if st.checkbox("Base Models"):
			st.subheader("F1Score, Accuracy and Train Time of our base models")
			st.info("We trained our models on raw data and found that that LinearSVC, Logistic Regression, XGboost and Lightgm were our top 4 models in terms of accuracy.")
			st.image('resources/f1score.png', channels="BGR")


		if st.checkbox(" Models Based on clean Data"):
			st.subheader("F1Score, Accuracy and Train Time of models trained with clean data")
			st.info("We trained our models on clean data and found that that LinearSVC, Logistic Regression, Lightgm and XGboost  were our top 4 models in terms of accuracy.")
			st.image('resources/f1clean.png', channels="BGR")	


		if st.checkbox(" Ensemble Predictions"):
			st.subheader("Ensemble Predictions")
			st.info(""" Combine Model Predictions Into Ensemble Predictions. Using a Voting classifier simply means building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions. It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data""")
			st.image('resources/esembled.png', channels="BGR")


def markup(heading):
	html_temp = """<div style=background-color:{};padding:10px;boarder-radius:10px"><h1 style="color:{};text-align:center;">"""+heading+"""</h1>"""
	st.markdown(html_temp.format('royalblue','white'), unsafe_allow_html=True)


#Function to lable our Sentiments
def getAnalysis(score):
    if score == 2:
        return 'News'
    elif score == 1:
        return 'Pro'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Anti'
train_df['Analysis'] = train_df['sentiment'].apply(getAnalysis)
train_df['msg_len'] = train_df['message'].apply(lambda x: len(x))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
