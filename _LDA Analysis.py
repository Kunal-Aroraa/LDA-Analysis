#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and dataset

# In[1]:


import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('/Users/hp/Downloads/savedrecs.csv')
df=pd.DataFrame(df)


# In[3]:


df.head()


# In[4]:


df


# # Data Preparation

# In[5]:


abstracts = df['Abstract'].dropna()

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(abstracts)


# # LDA Implementation

# In[6]:


lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda.fit(X)

#Function to display words
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print("\n")


# In[7]:


#Display 
no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)


# In[8]:


# Function to find topics and displaying
def dominant_topic(ldamodel, document_term_matrix):
    topics = []
    for i, row in enumerate(ldamodel.transform(document_term_matrix)):
        topic = np.argmax(row)
        topics.append(topic)
    return topics
dominant_topics = dominant_topic(lda, X)
data_with_topics = df.copy()
data_with_topics = data_with_topics.dropna(subset=['Abstract'])  
data_with_topics['Dominant Topic'] = dominant_topics
data_with_topics.head()


# # Data Visualisation 

# In[9]:


from wordcloud import WordCloud

# Function to generate word clouds for each topic
def generate_wordclouds(lda_model, feature_names, num_topics):
    for topic_idx in range(num_topics):

        topic_words = {feature_names[i]: lda_model.components_[topic_idx][i] for i in lda_model.components_[topic_idx].argsort()[:-no_top_words - 1:-1]}

        wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(topic_words)
        
        plt.figure(figsize=(9, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Word Cloud for Topic {topic_idx}')
        plt.show()

generate_wordclouds(lda, vectorizer.get_feature_names_out(), lda.n_components)


# In[10]:


sns.set_style("whitegrid")

topic_counts = data_with_topics['Dominant Topic'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette="viridis")
plt.title('Distribution of Dominant Topics in the Dataset')
plt.xlabel('Topic Number')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45)
plt.show()


# In[11]:


pip install pyldavis


# In[12]:


pyLDAvis.enable_notebook()

lda_vis = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne')
pyLDAvis.display(lda_vis)


# In[13]:


lda_vis = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne')
pyLDAvis.save_html(lda_vis, 'lda_visualization.html')


# # Using K-Means 

# In[14]:


topic_distributions = lda.transform(X)
num_clusters = 10 

kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(topic_distributions)

cluster_assignments = kmeans.labels_

data_with_clusters = df.copy()
data_with_clusters = data_with_clusters.dropna(subset=['Abstract'])  
data_with_clusters['Cluster'] = cluster_assignments

data_with_clusters.head()


# In[15]:


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(topic_distributions)
plt.figure(figsize=(12, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap='viridis', marker='o')
plt.colorbar(label='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Document Clusters')
plt.show()


# In[16]:


data_with_clusters.to_csv('clustered_dataset.csv', index=False)


# In[17]:


# Counting the number of documents in each cluster
cluster_counts = data_with_clusters['Cluster'].value_counts()

# Plotting the distribution of documents across clusters
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
plt.title('Distribution of Documents Across Clusters')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45)
plt.show()


# # Using Sentiment Analysis

# In[19]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Load your dataset
df = pd.read_csv('/Users/hp/Downloads/updated_dataset_with_topics.csv')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment score
def calculate_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Apply the function to your text data
df['Sentiment Score'] = df['Abstract'].apply(calculate_sentiment)

# Group by 'Dominant Topic' and calculate average sentiment score for each topic
average_sentiment_by_topic = df.groupby('Dominant Topic')['Sentiment Score'].mean()

# Display the average sentiment scores by topic
print(average_sentiment_by_topic)


# In[20]:


# Correcting the data format for the seaborn barplot function
topics = [f"Topic {i}" for i in range(9)]
sentiment_scores = [0.716485, 0.816680, 0.827357, 0.686636, 0.859758, 0.777115, 0.824950, 0.819748, 0.902900]

# Creating a DataFrame for the plot
df_plot = pd.DataFrame({'Topic': topics, 'Sentiment Score': sentiment_scores})

# Plotting the bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='Topic', y='Sentiment Score', data=df_plot)
plt.title('Average Sentiment Score for Each Topic')
plt.xlabel('Dominant Topic')
plt.ylabel('Average Sentiment Score')
plt.ylim(-1, 1)  # Sentiment scores range from -1 to 1
plt.axhline(0, color='grey', lw=0.8)  # Line to indicate neutral sentiment
plt.show()


# In[ ]:




