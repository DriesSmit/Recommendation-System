import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
import time
from scipy.sparse.linalg import svds

#Comments
#euclidean_similarity
#--------------------
# But works surprisingly well on the smaller datasets. This might be due to the almost consistent number of movies rated
# Not good if one person consistently rates more harsly

#pearson_similarity
#------------------
# This allows system finds stronger correlation even if one person rates is consistently harsher or kinder than the
# other.
# The system only considers commonly ranked items without a penalty for the amount of commonly ranked items.
# This means a person with only two movies rated can significantly influence another persons recommendations.
# The systems also scores perfect correlation if there is two or less commonly score items, which will have to be
# compensated for.

def doNothing(data):
    pass

def trainSVD(tableData):
    user_ratings_mean = np.mean(tableData, axis=1)
    R_demeaned = tableData - user_ratings_mean.reshape(-1, 1)

    # Calculate the SVD
    U, sigma, Vt = svds(R_demeaned, k=100)
    sigma = np.diag(sigma)

    model = []

    model.append(U)
    model.append(sigma)
    model.append(Vt)
    model.append(user_ratings_mean)

    return model

def train(tableData,algs=['SVD']):
    function_mappings = {
        'SVD': trainSVD,
    }

    trainTime = np.zeros(len(algs))
    models = []

    for i, curAlg in enumerate(algs):
        start = time.time()
        if curAlg=='SVD':
            print "Training ", curAlg,"..."
            models.append(function_mappings[curAlg](tableData))
        trainTime[i] += time.time() - start
    print "Training done."
    return trainTime,models

def euclidean_similarity(data,person1, person2):

	common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
	rankings = [(data[person1][itm], data[person2][itm]) for itm in common_ranked_items]
	distance = [pow(rank[0] - rank[1], 2) for rank in rankings]

	return 1 / (1 + sum(distance))

def pearson_similarity(data,person1, person2):

    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]

    n = len(common_ranked_items)

    s1 = sum([data[person1][item] for item in common_ranked_items])
    s2 = sum([data[person2][item] for item in common_ranked_items])

    ss1 = sum([pow(data[person1][item], 2) for item in common_ranked_items])
    ss2 = sum([pow(data[person2][item], 2) for item in common_ranked_items])

    ps = sum([data[person1][item] * data[person2][item] for item in common_ranked_items])
    num = n * ps - (s1 * s2)
    den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))

    return (num / den) if den != 0 else 0

def cosine_similarity(data,person1,person2):
    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]

    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx = sum([pow(data[person1][item], 2) for item in common_ranked_items])
    sumyy = sum([pow(data[person2][item], 2) for item in common_ranked_items])
    sumxy = sum([data[person1][item]*data[person2][item] for item in common_ranked_items])

    #print sumxx, " ",sumyy," ",sumxy

    den = math.sqrt(sumxx * sumyy)

    ans = sumxy / den if den != 0.0 else 0.0

    #print"Here:", ans
    return ans

def recommend(data,person, item,bound, similarity=pearson_similarity):
    scores = [(similarity(data,person, other), other) for other in data if other != person]
    scores.sort()
    scores.reverse()
    #print(scores)
    scores = scores[0:bound]

    #print(scores)

    recomms = (0.0,0.0)

    for sim, other in scores:
        ranked = data[other]

        if item in ranked:
            weight = sim * ranked[item]

            curSim, curW = recomms

            recomms = (curSim + sim, curW + weight)

    sim, weight = recomms

    recomms = weight / sim if sim!=0.0 else 2.5  # This is a extremely unlikely event of zero correlation found. Maybe
    # just add 0.0001 to sim to increase speed.
    return recomms

def general_popularity(data,person, item):

    meanScore = 0.0
    meanCount = 0

    for i in range(len(data)):

        curScore = data[i][item]

        if curScore != 0:
            meanScore += curScore
            meanCount += 1

    meanScore = meanScore/meanCount

    return meanScore

def randomItem():
    return random.random()

def SVD(tableData,model,userID,itemID):
    # Get and sort the user's predictions
    U = model[0]
    sigma = model[1]
    Vt = model[2]
    user_ratings_mean = model[3]

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    userRate = all_user_predicted_ratings[userID]

    #print "Real: ", tableData[userID]
    #print "Prediction: ", userRate

    #Maybe normalize the data
    return userRate[itemID]

