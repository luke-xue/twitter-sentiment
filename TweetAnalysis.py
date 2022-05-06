import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Twitter API credentials
consumerKey = ''
consumerSecret = ''
accessToken = ''
accessTokenSecret = ''

#create authenticate object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(authenticate, wait_on_rate_limit= True)

search_term = 'Tesla earnings'
tweets = tweepy.Cursor(api.search,
                       q = search_term,
                       lang='en',
                       since = '2022-03-27',
                       tweet_mode = 'extended'
                       ).items(1500)

#store tweets
all_tweets = [tweet.full_text for tweet in tweets]
all_tweets

#create dataframe to store tweets
df = pd.DataFrame(all_tweets, columns=['Tweets'])
df.head(6)



def cleanTwt(twt):
  twt = re.sub('RT', '', twt)
  twt = re.sub('#[A-Za-z0-9]+', '', twt)
  twt = re.sub('\\n', '', twt)
  twt = re.sub('https?:\/\/\S+', '', twt)
  twt = re.sub('@[\s]*', '', twt)
  twt = re.sub('^[\s]+|[\s]+$', '', twt)
  return twt

#create column of cleaned tweets
df['Cleaned_Tweets'] = df['Tweets'].apply(cleanTwt)
df.head()

#create a new dataframe
df = pd.DataFrame(df['Cleaned_Tweets'], columns=['Cleaned_Tweets'])
df.drop_duplicates(inplace=True)
idx = list(range(0, len(df)))
df = df.set_index(pd.Index(idx))
df

#get subjectivity of tweet
def getSubjectivity(twt):
  return TextBlob(twt).sentiment.subjectivity

#get polarity of tweet (positive, negative, or neutrality of tweet)
def getPolarity(twt):
  return TextBlob(twt).sentiment.polarity

#Create 2 new columns to dataframe
df['Subjectivity'] = df['Cleaned_Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Cleaned_Tweets'].apply(getPolarity)
df.head(6)

def getSentiment(value):
  if value < 0:
    return 'Negative'
  elif value > 0:
    return 'Positive'
  else:
    return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(getSentiment)
df.head()

#scatterplot!
plt.figure(figsize = (8,6))
for i in range(0, len(df)):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color = 'green')
plt.title('Scatter Plot')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()
