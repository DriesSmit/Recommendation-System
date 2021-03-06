#https://beckernick.github.io/matrix-factorization-recommender/

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0

    #print(predictions_df)

    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    #print(sorted_user_predictions)
    #print(pd.DataFrame(sorted_user_predictions).reset_index())
    #print("Movie at 802: ", movies_df.values[802])
    #print(movies_df.values[0])

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').
        sort_values(['Rating'], ascending=False)
        )

    #print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    #print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='MovieID',
                                 right_on='MovieID').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :]
                           )
    #print(recommendations)
    return user_full, recommendations

#Load data and create normalized data
ratings_list = [i.strip().split("::") for i in open('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-1m/ratings.dat', 'r').readlines()]
#users_list = [i.strip().split("::") for i in open('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-1m/movies.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)



R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)

#Train

R = R_df.as_matrix()

user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

#Calculate the SVD
U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

#Recommend
already_rated, predictions = recommend_movies(preds_df, 837, movies_df, ratings_df, 10)

#print(already_rated.head(10))
print(predictions.head(10))