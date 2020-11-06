
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


import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy_streamlit
import string
import re

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

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
	
	menu = ["Home", "Predictions", "Explanatory Design Analysis", "About The Predict"]
	selection = st.sidebar.selectbox("Menu", menu)

	if selection == "Home":
		st.markdown(html_temp.format('royalblue','white'), unsafe_allow_html=True)
		st.markdown(title_temp, unsafe_allow_html=True)



	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Predictions":
		markup(selection)
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("C:/Users/mojel/Documents/EDSA/Predict/Classification/climate-change-edsa2020-21/LinearSVC_model.pkl"),"rb"))
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
		if st.checkbox("barchat"):
			st.subheader("Analys per count")
			plt.title('Sentiment Analysis')
			plt.xlabel('Sentiments')
			plt.ylabel('Counts')
			train_df['Analysis'].value_counts().plot(kind='bar')
			plt.show()
			st.pyplot()


		if st.checkbox("word cloud"):
			st.subheader("word cloud")
			allwords = ' '.join( [tweets for tweets in train_df['message']])
			wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allwords)
			plt.imshow(wordCloud, interpolation = "bilinear")
			plt.axis('off')
			plt.show()
			st.pyplot()


		if st.checkbox("messagen length"):
			st.subheader("message length per each class")
			bins = np.linspace(0, 150,50)
			plt.hist(train_df['msg_len'], bins)
			plt.legend(loc='upper right')
			plt.title('Message length')
			plt.show()
			st.pyplot()


		if st.checkbox("metrics"):
			st.subheader('plot the PCC figure')
			corr = train_df.corr(method = 'pearson')
			f, ax = plt.subplots(figsize=(11, 9))
			cmap = sns.diverging_palette(10, 275, as_cmap=True)
			sns.heatmap(corr, cmap=cmap, square=True,
			linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
			st.pyplot()

		if st.checkbox("sentiment analysis"):
			st.subheader('sentiment analysis')
			sia = SentimentIntensityAnalyzer()
			train_df['polarity scores'] = train_df["message"].apply(lambda x: sia.polarity_scores(x)["compound"])
			train_df['sentiment'] = train_df['polarity scores'].apply(lambda y:'Anti' if y<0 else ('neutral' if y ==0 else 'positive' if y==1 else 'news') )
				
			total_tweets = len(train_df.index)
			anti_sent=len(train_df[train_df['polarity scores']<0])
			neutral_sent=len(train_df[train_df['polarity scores']==0])
			pro_sent=len(train_df[train_df['polarity scores']==1])
			news_sent=len(train_df[train_df['polarity scores']==2])
			labels=['anti','neutral','pro','news']
			colors=['red','blue','green','yellow']
			explode = (0, 0, 0.1, 0.5) 
			fig1, ax = plt.subplots()
			wedges = ax.pie([anti_sent,neutral_sent,pro_sent,news_sent], explode=explode,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
			ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
			ax.set_title('Sentiment analysis for the tweets')
			plt.legend(labels=['anti','neutral','pro','news'], loc="upper right")
			plt.tight_layout()
			plt.show()
			#st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()
			
		# #eda(train_df)
		# st.subheader("Data Analysis")
		# st.write("climate_data")
		# st.bar_chart(train_df['sentiment'])



	# Building out the "About" page
	if selection == "About The Predict":
		markup(selection)
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

def markup(heading):
	html_temp = """<div style=background-color:{};padding:10px;boarder-radius:10px"><h1 style="color:{};text-align:center;">"""+heading+"""</h1>"""
	st.markdown(html_temp.format('royalblue','white'), unsafe_allow_html=True)

# Data Cleaning
def cleanText(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Remove @mentions
    text = re.sub(r':[\s]+', '', text)
    text = re.sub(r'#', '', text) #Remove # symbol
    text = re.sub(r'rt[\s]+', '', text) #Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text) #Remove hyper-links
    return text
train_df['message'] = train_df['message'].apply(cleanText)

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
