#content based recomendation system
import numpy as np
import pandas as pd
import warnings # with this module we can ignore any warning
warnings.filterwarnings('ignore')
#get movie lens dataset
columns_names = ["user_id","item_id","ratings","timestamp"]
df = pd.read_csv("C:/project movies/ml-100k/u.data",sep = "\t",names =columns_names) # we have used seperstos here because
#u.data is a tab sepreated file not csv file
print(df.head())
print(df.shape)
print(df["user_id"].nunique())#give no of unique user ids
print(df["item_id"].nunique())#give no of unique item (here item are "movies")
movie_title = pd.read_csv("C:/project movies/ml-100k/u.item",sep = "\|",header = None)#get thename of movies
movie_title = movie_title[[0,1]]
movie_title.columns = ["item_id","title"]
print(movie_title)
print(movie_title.shape)
print(movie_title.head())
df = pd.merge(df,movie_title,on = "item_id")#on take on which key are merging datasets
print(df.tail())
########expolatory data analysis#################
#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_style('white')
print( df.groupby('title').mean()['ratings'].sort_values(ascending = False))#find the average rating of any movie
#and sorting the  the values on the basis of ratings
##finding the no of person given the ratings######have given the ratings
print(df.groupby('title').count()['ratings'].sort_values(ascending = False))
###creating seperate dataframe for rating data of movives
rating = pd.DataFrame(df.groupby('title').mean()['ratings'])
print(rating.head())
rating['num of ratings']= pd.DataFrame(df.groupby('title').count()['ratings'])#addind no.of rating column in rating
print(rating.sort_values(by ='ratings',ascending = False))
#poting the histogram of "num of rating "column
plt.figure(figsize=(10,6))
plt.hist(rating['num of ratings'],bins =70)
plt.show()
#histogram of "rating"column
plt.hist(rating["ratings"],bins = 70)
plt.show()
#jointplot
#sns.jointplot(x='ratings' ,y='num of ratings' ,data = rating, alpha = 0.5)
plt.show()
#CREATING MOVIE RECOMENDATION SYSTEM
print(df.head())
moviematrix = df.pivot_table(index = "user_id", columns= "title",values = "ratings")
#in tgis mrix if a person of any user id has not rated any movie VALUE will be shown as NAN
print(moviematrix)
# TO CHECK HIGHEST WATCHED MOVIE
print(rating.sort_values('num of ratings',ascending = False).head())
#TO CHECK ANY PERTICULAR MOVIE RATING ONLY
print(moviematrix['Star Wars (1977)'])# here we are using STARWARS MOVIE
#corelation of any perticular movie with whole movie matrix
starwars_user_ratings = moviematrix['Star Wars (1977)']
similar_to_starwars = moviematrix.corrwith(starwars_user_ratings)
print(similar_to_starwars)  #telling how much other movies are similsr to starwars
##similar_to_starwars giving NAN value fothe move to whinch nobody has rated

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['correlation'])
corr_starwars.dropna( inplace= True )# it will not print NAN value
###NOTE :  if i use only dropna() it will not change the object value
### but if i will use dropna(inplace = True) it will change the vaue of object
print(corr_starwars)
print(corr_starwars.sort_values('correlation',ascending = False).head(10))
#doing fiteration on the basis of num of ratings
corr_starwars = corr_starwars.join(rating['num of ratings'])
print(corr_starwars.head())
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending = False))
#prection function
def predict_movies(movie_name):
    movie_user_ratings = moviematrix[movie_name]
    similar_to_movie = moviematrix.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['correlation'])
    corr_movie.dropna( inplace= True )
    corr_movie = corr_movie.join(rating['num of ratings'])
    prediction = corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending = False)
    return(prediction)
prediction = predict_movies("Star Wars (1977)")
print(prediction.head())


    











