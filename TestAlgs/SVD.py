#https://beckernick.github.io/matrix-factorization-recommender/

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

'''def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
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
print(predictions.head(10))'''



#Test incremental SVD

#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=10, alpha=0.0002, beta=0.02):
    Q = Q.T
    curAlpha = 0.05
    for step in xrange(steps):

        if curAlpha > alpha:
            curAlpha *= 0.95

        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + curAlpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + curAlpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        print("Step: ", step)
        print(eR)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

if __name__ == "__main__":
    numpy.random.seed(0)
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)

    nR = numpy.dot(nP, nQ.T)
    print(nR)


    def trainIncrementalSVD(data, K=40, steps=10, alpha=0.0002, beta=0.02, alphaReg=True, Q=None, P=None):
        if Q is None:
            N = len(data)
            M = len(data[0])

            P = np.random.rand(N, K)
            Q = np.random.rand(M, K)

            Q = Q.T
        check = 0
        if alphaReg:
            curAlpha = 0.01
        else:
            curAlpha = alpha

        # Map non zeros
        print "Mapping actual values."
        valueMap = []
        for row in xrange(len(data)):
            for col in xrange(len(data[row])):
                if data[row][col] > 0:
                    valueMap.append((row, col))
        print "Actual values mapped."

        # loopTime = 0.0
        # otherTime = 0.0
        for step in xrange(steps):

            print "Step: ", step

            if curAlpha > alpha:
                curAlpha *= 0.90

            '''stepLen = step*len(data)
            if stepLen+row > check:
                #print("Start")
                check += steps*len(data) * 0.01

                mae = 0.0
                rse = 0.0
                count = 0
                skipRate = 500
                for i in xrange(0,len(data),skipRate):
                    for j in xrange(0,len(data[i]),skipRate):
                        if data[i][j] > 0:
                            mae += abs(data[i][j] - np.dot(P[i,:],Q[:,j]))  # Mean absolute error(MAE)
                            rse += pow(data[i][j] - np.dot(P[i,:],Q[:,j]), 2)  # Mean absolute error(MAE)
                            count += 1
                mae = mae / count if count>0 else None
                rse = rse / count if count>0 else None

                perc = round((step * len(data) + row) * 100.0 / (steps * len(data)), 2)

                print "Percentage completed: ", perc #, "%. Estimated mean absolute error: ", round(mae,3), ". Estimate root square error: ", round(rse, 3)
                if mae < 0.001:
                    break'''

            alphaBeta = curAlpha * beta
            # otherTime += time.time()-start
            # start = time.time()
            # print("Ready")
            for valLoc in valueMap:
                eij = data[row][col] - np.dot(P[valLoc[0], :], Q[:, valLoc[1]])
                eAlpha2 = curAlpha * 2 * eij
                for k in xrange(K):
                    P[valLoc[0]][k] += eAlpha2 * Q[k][valLoc[1]] - alphaBeta * P[row][k]
                    Q[k][valLoc[1]] += eAlpha2 * P[valLoc[0]][k] - alphaBeta * Q[k][valLoc[1]]
            # loopTime += time.time() - start

        # print "Looptime: ", loopTime, ". Other: ", otherTime

        nR = np.dot(P, Q)
        return nR