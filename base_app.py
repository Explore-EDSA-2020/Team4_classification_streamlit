
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
#import spacy_streamlit
from PIL import Image
import spacy
import string
import re
import os

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# porterStemmer = nltk.PorterStemmer()
# wordNetLemma = nltk.WordNetLemmatizer()
# stopword = stopwords.words('english')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

st.set_option('deprecation.showPyplotGlobalUse', False)

# nlp = spacy.load('en')

# Load your raw data
raw = pd.read_csv("resources/train.csv")
train_df = raw



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	html_temp = """
	<div style="background-color:{};padding:10px;border-radius:10px;margin:10px;">
	<img src="resources/imgs/EDSA_LOGO.PNG" alt="edsa logo" width="500" height="600">
	<h1 style="color:{};text-align:center;">EDSA 2020:Climate Change Belief Analysis</h1>
	</div>
	"""

	title_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:white;text-align:center;">CLASSIFICATION PREDICT</h1>
	<h2 style="color:white;text-align:center;">TEAM:4</h2>
	<h3 style="color:white;text-align:right;">2020/11/03</h3>
	</div>
	"""
	article_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<h6>Author:{}</h6> 
	<h6>Post Date: {}</h6>
	<p style="text-align:justify">{}</p>
	</div>
	"""
	head_message_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<h6>Author:{}</h6> 
	<h6>Post Date: {}</h6> 
	</div>
	"""
	full_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
	<p style="text-align:justify;color:black;padding:10px">{}</p>
	</div>
	"""

	
	# st.markdown(title_temp, unsafe_allow_html=True)
	# st.markdown(article_temp, unsafe_allow_html=True)
	# st.markdown(head_message_temp, unsafe_allow_html=True)
	# st.markdown(full_message_temp, unsafe_allow_html=True)

	# Creates a main title and subheader on your page -
	# these are static across all pages
	# st.title("Tweet Classifer")
	# st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	menu = ["Home", "About The Predict", "Text Classification", "Explanatory Design Analysis", "Predictions","Model Performance"]
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


# Building out the "Text Classification" page
	if selection == "Text Classification":
		# markup(selection)
		
		our_image = Image.open(os.path.join('resources/imgs/logo.png'))
		st.image(our_image)

		menu = ["Home","NER"]
		choice = st.selectbox("Menu",menu)

		# if choice == "Home":
		# 	st.subheader("Tokenization")
		# 	raw_text = st.text_area("Your Text","Enter Text Here")
		# 	docx = nlp(raw_text)
		# 	if st.button("Tokenize"):
		# 		spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])

		# elif choice == "NER":
		# 	st.subheader("Named Entity Recognition")
		# 	raw_text = st.text_area("Your Text","Enter Text Here")
		# 	docx = nlp(raw_text)
		# 	spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

	# Building out the predication page
	if selection == "Predictions":
		
			markup(selection)
			st.info("Prediction with ML Models")
			if st.checkbox("LinearSVC"):
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

			if st.checkbox("DecisionTreeClassifier"):
				st.info("Decision Tree - A machine learning model that operates by partitioning data into smaller subsets. It considers all possible binary data splits and selects the data split with the best separation of the data")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for DecisionTree","Type Text for Decision Tree")
			

				if st.button("Classify with DecisionTree"):
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
			

			if st.checkbox("Random_Forest"):
				st.info("Random Forest - An ensemble machine learning method that operates by constructing multiple decision trees at training time and outputting the mode classifications for the individual trees")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for random forest","Type Here for RF")
			

				if st.button("Classify for random forest"):
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


			if st.checkbox("KNeiobors Classifier"):
				st.info("KNeighbors Classifiers - implements classification based on voting by nearest k-neighbors of target point, t, while RadiusNeighborsClassifier implements classification based on all neighborhood points within a fixed radius, r, of target point, t")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for knb","Type Here for knb")
			

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
	

			if st.checkbox("Lightgbm"):
				st.info("Light gbm -  is a gradient boosting framework that uses tree based learning algorithm. It grows tree vertically while other algorithm grows trees horizontally.")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for lightgbm","Type Here for lightgbm")
			

				if st.button("Classify with lightgbm"):
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

			if st.checkbox("Xgboost"):
				st.info("XGBoost - An ensemble machine learning model technique that uses gradient boosting framework for machine learning.")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for xgboost","Type Here for xgboost")
			

				if st.button("Classify for xgboost"):
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

			if st.checkbox("Logistic regression"):
				st.info("Logistic Regression - A machine learning method that computes the probability of an event occuring and places it in the relevant class or category")
				# Creating a text box for user input
				tweet_text = st.text_area("Enter Text for logistic regression","Type Here for logistic regression")
			

				if st.button("Classify for logistic regression"):
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

	# Building out the EDA page
	if selection == "Explanatory Design Analysis":
		markup(selection)
		train_df = raw

		#Show the counts

		#plot and visualize the counts
		if st.checkbox("Models and their Accuracy score"):
			st.subheader("Models based on raw data and the accuracy score")
			st.image('resources/accuracy.png', channels="BGR")


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

			st.subheader("handles of the tweets per message(Pro)")
			st.image('resources/hand_pro.png', channels="BGR")


			st.subheader("handles of the tweets per message(Anti)")
			st.image('resources/handl_anti.png', channels="BGR")


			st.subheader("handles of the tweets per message(Neutral)")
			st.image('resources/hand_neutr.png', channels="BGR")
			


		if st.checkbox("frequent words used most in tweets"):
			st.subheader("frequent words used most in tweets(news)")
			st.image('resources/words_news.png', channels="BGR")


			st.subheader("frequent words used most in tweets(Pro)")
			st.image('resources/Pro_news.png', channels="BGR")


			st.subheader("frequent words used most in tweets(Anti)")
			st.image('resources/anti.png', channels="BGR")


			st.subheader("frequent words used most in tweets(Neutral)")
			st.image('resources/neutral.png', channels="BGR")


		if st.checkbox("frequent # used most in tweets"):
			st.subheader("frequent # used most in tweets(news)")
			st.image('resources/#_news.png', channels="BGR")


			st.subheader("frequent # used most in tweets(Pro)")
			st.image('resources/Pro_#.png', channels="BGR")


			st.subheader("frequent # used most in tweets(Anti)")
			st.image('resources/anti_#.png', channels="BGR")


			st.subheader("frequent # used most in tweets(Neutral)")
			st.image('resources/neutral_#.png', channels="BGR")	

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

# Data Cleaning
# def clean_message(message):
#     str(message).lower()
#     regrex_pattern = re.compile(pattern = "["
#       u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002500-\U00002BEF"  # chinese char
#         u"\U00002702-\U000027B0"
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u"\U00010000-\U0010ffff"
#         u"\u2640-\u2642" 
#         u"\u2600-\u2B55"
#         u"\u200d"
#         u"\u23cf"
#         u"\u23e9"
#         u"\u231a"
#         u"\ufe0f"  # dingbats
#         u"\u3030"
#          "]+", flags = re.UNICODE)
#     message = regrex_pattern.sub(r'',message) # remove emojis
#     # Remove user @ references and '#' from tweet
#     message = re.sub(r'@[A-Za-z0-9]+','',message) ##Remove @aderate
#     message = re.sub(r'RT[\s]+', '', message) ## remove RT Retweets
#     message = re.sub(r'https?:\/\/\S+', '', message) ##remove hyperlink
#     message =  ''.join([char for char in message if char not in string.punctuation]) ## remove puntuations i.e. ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
#     message = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', message) # remove URLs
#     message = re.sub(r'@[^\s]+', '', message) # remove usernames
#     message = re.sub(r'#[A-Za-z0-9]+', '', message) #get rid of hashtags
#     message = message.translate(str.maketrans('', '', string.punctuation))
#     message_tokens = word_tokenize(message)
#     filtered_message = [word.lower() for word in message_tokens if word not in stopword]

#     stemmed_words = [porterStemmer.stem(word) for word in filtered_message]
#     lemma_words = [wordNetLemma.lemmatize(word) for word in stemmed_words]

#     return ' '.join(lemma_words)
# train_df['message'] = train_df['message'].apply(clean_message)

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

# def punctuation_count(txt):
# 	count = sum([1 for c in txt if c in string.punctuation])
# 	return 100*count/(len(txt))	

# train_df['punctuation_%'] = train_df['message'].apply(lambda x: punctuation_count(x))

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
