#!/usr/bin/env python
# coding: utf-8

# # Recommendation System

# # Importing Library

# code di bawah melakukan importing library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime

now = datetime.now()

from sklearn.model_selection import train_test_split

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Lambda
from tensorflow.keras.layers import Embedding, Flatten, dot, Dot
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping


# # Data Understanding

# code di bawah melakukan pembacaan terhadap data dan menunjukkan informasi mengenai data

# In[3]:


movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')
links = pd.read_csv('ml-latest-small/links.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')


# In[3]:


movies.info()


# In[4]:


tags.info()


# In[5]:


links.info()


# In[6]:


ratings.info()


# In[7]:


print('jumlah film tersedia dalam dataset: ', len(movies.movieId.unique()))
print('jumlah link based movies ID: ', len(links.movieId.unique()))
print('jumlah tags based movies ID: ', len(tags.movieId.unique()))
print('jumlah rating based movies ID: ', len(ratings.movieId.unique()))
print('jumlah tags per user: ', len(tags.userId.unique()))
print('jumlah rating per user: ', len(ratings.userId.unique()))


# # Univariate EDA

# Code di bawah melakukan EDA terhadap data yang akan digunakan

# In[8]:


print("banyak genre dari movies: ", len(movies.genres.unique()))


# In[9]:


print("banyak tag dari movies: ", len(tags.tag.unique()))
print("jenis tag dari movies: ", tags.tag.unique())


# In[10]:


print("banyak rating dari movies: ", len(ratings.rating.unique()))
print("jenis rating dari movies: ", ratings.rating.unique())


# In[4]:


df_movie_rate_merge = pd.merge(ratings, movies, on='movieId', how='left')


# In[5]:


df_movie_rate_merge


# In[6]:


sns.pairplot(df_movie_rate_merge[['movieId','userId','rating']])


# ## Movies

# pada data di bawah dapat diketahui bahwa di dalam data movies film emma (1996) memiliki data paling banyak dengan jumlah 1053

# In[11]:


movies.head()


# In[12]:


movies.describe(include=object)


# ## Ratings

# Pada data di bawah dapat diketahui bahwa nilai minimal rating 0.5 dan paling tinggi 5

# In[13]:


ratings.head()


# In[14]:


ratings.describe()


# ## Links

# In[15]:


links.head()


# In[16]:


links.describe()


# ## Tags

# Pada data di bawah dapat diketahui bahwa data tags paling banyak adalah in netflix queue dengan total 131, data ini mungkin mengartikan bahwa user melakukan tag antrian terhadap suatu film pada platform netflix

# In[17]:


tags.head()


# In[18]:


tags.describe(include=object)


# # Data preprocessing

# Pada tahap ini dilakukan penggabungan data dan pemrosesan terhadap data agar data dapat diolah.

# In[19]:


movies_all = np.concatenate((
    movies.movieId.unique(),
    links.movieId.unique(),
    tags.movieId.unique(),
    ratings.movieId.unique()
))

movies_all = np.sort(np.unique(movies_all))

print("jumlah seluruh data movies based movieId: ", len(movies_all))


# In[20]:


user_all = np.concatenate((
    ratings.userId.unique(),
    tags.userId.unique()
))

user_all = np.sort(np.unique(user_all))

print("jumlah seluruh data user : ", len(user_all))


# In[21]:


movies_df = pd.merge(tags, movies, on='movieId', how='left')
movies_df.info()


# In[22]:


movies_df = pd.merge(ratings.drop(['userId','timestamp'],axis=1), movies_df, on='movieId', how='left')
movies_df.info()


# In[23]:


movies_df.head()


# In[24]:


movies_df.shape


# In[25]:


movies_df.isnull().sum()


# In[26]:


movies_df.dropna(inplace=True)
movies_df.shape


# In[27]:


movies_df.isnull().sum()


# In[28]:


fix_movies = movies_df.sort_values('movieId', ascending=True)
fix_movies


# In[29]:


len(fix_movies.movieId.unique())


# In[30]:


fix_movies.drop('tag',axis=1, inplace=True)


# In[31]:


preparation = fix_movies
preparation = preparation.drop_duplicates('movieId')
preparation


# In[32]:


movie_id = preparation['movieId'].tolist()
title_movie = preparation['title'].tolist()
rating_movie = preparation['rating'].tolist()
genres_movie = preparation['genres'].tolist()


# In[33]:


movie_new = pd.DataFrame({
    'id':movie_id,
    'title':title_movie,
    'genre':genres_movie,
    'rating':rating_movie
})

movie_new


# # Model Development Content Based Filtering

# Pada tahap ini merupakan tahap membuat sistem rekomendasi menggunakan content based filtering

# # Vectorizing the data

# Tahap ini dilakukan vektorirasi terhadap data

# In[34]:


genres = movie_new["genre"].str.get_dummies(sep="|")
movie_genres = pd.concat([movie_new['title'], genres], axis=1)
movie_genres


# In[35]:


movie_genres.drop('(no genres listed)',axis=1,inplace=True)


# ## Cosine Similarity

# Tahap ini melakukan perhitungan kesamaan kosinus menggunakan library sklearn

# In[36]:


from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


cs = cosine_similarity(np.array(genres))
cs


# In[38]:


cosine_sim_df = pd.DataFrame(cs, index=movie_new['title'], columns=movie_new['title'])
print('shape: ', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)


# code di bawah menunjukkan fungsi untuk melakukan rekomendasi dan hasil dari rekomendasi

# In[39]:


def movie_recommendations(title, similarity_data=cosine_sim_df, items=movie_new[['title', 'genre']], k=5):
    index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))
    
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    closest = closest.drop(title, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)


# In[44]:


movie_new[movie_new.title.eq('Secrets & Lies (1996)')]


# In[45]:


movie_recommendations('Secrets & Lies (1996)')


# # Data Preparation collaborative filtering

# Pada tahap ini dilakukan pemrosesan untuk collaborative filtering

# In[4]:


ratings = pd.merge(movies.drop('genres',axis=1), ratings, on='movieId', how='left')
ratings


# In[5]:


ratings['userId'] = ratings['userId'].astype(float)
ratings['movieId'] = ratings['movieId'].astype(str).astype(int)
ratings['rating'] = ratings['rating'].astype(str).astype(float)
ratings['timestamp'] = ratings['timestamp'].apply(lambda x: now.strftime("%m/%d/%Y, %H:%M:%S"))
ratings


# In[6]:


number_users = ratings['userId'].unique().shape[0]
number_movies = ratings['movieId'].unique().shape[0]
number_ratings = len(ratings)
avg_ratings_per_user = number_ratings/number_users


# In[118]:


print('Number unique users: ', number_users)
print('Number unique movies: ', number_movies)
print('Number total ratings: ', number_ratings)
print('Average number ratings per user: ', avg_ratings_per_user)


# In[119]:


movieId = ratings.groupby("movieId").count().sort_values(by="rating",ascending=False)[0:1000].index
ratings_new = ratings[ratings.movieId.isin(movieId)]
ratings_new.count()


# In[120]:


ratings_new.shape


# In[121]:


userId = ratings_new.groupby("userId").count().sort_values(by="rating",ascending=False).sample(n=500, random_state=42).index
ratings_new_cop = ratings_new[ratings_new.userId.isin(userId)]
ratings_new_cop.count()


# ## Generate New Id for movie and user

# Pada tahap ini dilakukan pembuatan identifier baru atau biasa disebut encoding

# In[122]:


movies = ratings_new_cop['movieId'].unique()
moviesDF = pd.DataFrame(data=movies,columns=['originalMovieId'])
moviesDF['newMovieId'] = moviesDF.index+1


# In[123]:


users = ratings_new_cop['userId'].unique()
usersDF = pd.DataFrame(data=users,columns=['originalUserId'])
usersDF['newUserId'] = usersDF.index+1


# In[124]:


ratings_new_cop = ratings_new_cop.merge(moviesDF,left_on='movieId',right_on='originalMovieId')
ratings_new_cop.drop(labels='originalMovieId', axis=1, inplace=True)
ratings_new_cop = ratings_new_cop.merge(usersDF,left_on='userId',right_on='originalUserId')
ratings_new_cop.drop(labels='originalUserId', axis=1, inplace=True)


# In[125]:


number_users = ratings_new_cop.userId.unique().shape[0]
number_movies = ratings_new_cop.movieId.unique().shape[0]
number_ratings = len(ratings_new_cop)
avg_ratings_per_user = number_ratings/number_users

print('Number of unique users: ', number_users)
print('Number of unique movies: ', number_movies)
print('Number of total ratings: ', number_ratings)
print('Average number of ratings per user: ', avg_ratings_per_user)


# In[126]:


throwback_df = ratings_new_cop
throwback_df


# ## Splitting the data

# Pada tahap ini data dibagi 20% untuk testing dan 80% untuk training

# In[127]:


X_train, X_test = train_test_split(ratings_new_cop.drop('title',axis=1), test_size=0.20, shuffle=True, random_state=42)
X_validation, X_test = train_test_split(X_test, test_size=0.50, shuffle=True, random_state=42)


# In[128]:


print('train set:', X_train.shape)
print('validation set:',X_validation.shape)
print('test set: ',X_test.shape)


# ## Melakukan pengecekan sparsity

# Pada tahap dilakukan pengecekan sparsity pada data dan dapat diketahui bahwa untuk data train sparsity dimiliki sebesar 7.93%

# In[129]:


ratings_train = np.zeros((number_users, number_movies))
for row in X_train.itertuples():
    ratings_train[row[6]-1, row[5]-1] = row[3]


# In[130]:


sparsity = float(len(ratings_train.nonzero()[0]))
sparsity /= (ratings_train.shape[0] * ratings_train.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))


# In[131]:


ratings_validation = np.zeros((number_users, number_movies))
for row in X_validation.itertuples():
    ratings_validation[row[6]-1, row[5]-1] = row[3]


# In[132]:


ratings_test = np.zeros((number_users, number_movies))
for row in X_test.itertuples():
    ratings_test[row[6]-1, row[5]-1] = row[3]


# ## Model development collaborative filtering

# Pada tahap dilakukan pembuatan model collaborative filtering menggunakan loss MSE dan Earlystopping callback

# In[7]:


class CollaborativeFilteringModel(Model):
    def __init__(self, number_users, number_movies, n_latent_factors=1, **kwargs):
        super(CollaborativeFilteringModel, self).__init__(**kwargs)
        self.n_latent_factors = n_latent_factors
        self.user_input = Input(shape=[1], name='user')
        self.user_embedding = Embedding(input_dim=number_users + 1, output_dim=n_latent_factors,
                                        name='user_embedding')(self.user_input)
        self.user_vec = Flatten(name='flatten_users')(self.user_embedding)
        self.movie_input = Input(shape=[1], name='movie')
        self.movie_embedding = Embedding(input_dim=number_movies + 1, output_dim=n_latent_factors,
                                         name='movie_embedding')(self.movie_input)
        self.movie_vec = Flatten(name='flatten_movies')(self.movie_embedding)
        self.product = Dot(axes=1)([self.movie_vec, self.user_vec])
        super(CollaborativeFilteringModel, self).__init__(inputs=[self.user_input, self.movie_input], outputs=self.product, **kwargs)
    
    def compile(self, optimizer='adam', loss='mean_squared_error', **kwargs):
        super(CollaborativeFilteringModel, self).compile(optimizer=optimizer, loss=loss, **kwargs)

model = CollaborativeFilteringModel(number_users, number_movies)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# In[134]:


x_train = [X_train['newUserId'], X_train['newMovieId']]
y_train = X_train['rating']

x_val = [X_validation['newUserId'], X_validation['newMovieId']]
y_val = X_validation['rating']

early_stopping = EarlyStopping(patience=3)
history = model.fit(x_train,
                    y_train, epochs=50,
                    validation_data=(x_val,y_val),
                    callbacks=[early_stopping])


# Pada code di bawah diketahui bahwa model train memiliki nilai MSE train sebesar 0.67 dan MSE validation sebesar 0.74

# In[135]:


print('score MSE train: ', history.history['loss'][-1])
print('score MSE Validation: ', history.history['val_loss'][-1])
plt.plot(history.history['loss'][10:])
plt.plot(history.history['val_loss'][10:])
plt.title('model_metrics')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Code di bawah merupakan fungsi dan output dari rekomendari berdasarkan user rating (collaborative filtering)

# In[141]:


def output_recommendation(x):
    
    movie_watched = throwback_df[throwback_df.newUserId == user_id]

    movie_not_watched = throwback_df[~throwback_df['newMovieId'].isin(movie_watched.newMovieId.values)][['title','rating','newMovieId']]

    movie_id_pred = model.predict(x)
    user_rate = movie_id_pred.flatten()

    top_rate_indices = user_rate.argsort()[-20:][::1]
    
    user_top_movie = movie_watched.sort_values(by='rating',ascending=False)['title'].values
    
    movie_rec = []
    counter_rec = 0
    for _ in top_rate_indices:
        movie_rec_title = movie_not_watched[movie_not_watched['newMovieId'] == _]['title'].unique()
        if movie_rec_title.size != 0 and counter_rec <= 10:
            movie_rec.append(movie_rec_title[0])
            counter_rec += 1
    
    print("recommendation for user_id: ",user_id)
    print("="*20)
    print("top 3 movie by user: ")
    counter_user = 0
    for i in user_top_movie:
        counter_user += 1
        if counter_user <= 3:
            print(i)
            
    print("="*20)
    print("recommendation by system:")
    
    for i in movie_rec:
        print(i)
        
    return None


# In[142]:


user_id = throwback_df.newUserId.sample(1).iloc[0]
user_df_x = throwback_df[throwback_df['newUserId'] == user_id]
x = [user_df_x['newUserId'],user_df_x['newMovieId']]

output_recommendation(x)

