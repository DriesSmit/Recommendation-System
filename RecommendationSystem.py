import matplotlib.pyplot as plt
import numpy as np
import math
import random
import numpy as np
import time
from scipy.sparse.linalg import svds

#Comments
#Please refer to Report.odt for a more detailed explanation and result page
#of each algorithm

def trainFullSVD(data):
    # user-item_rating = average_item_rating + (average_user_rating)

    # Adding data to missing entries to better generalise

    # For initialisation the average between mean Columns and Rows was found to be the best values to use.
    # For a k=100 value and using the testHash set for u1
    # Row avg:      1.01
    # Column avg:   0.8573
    # Mix avg:      0.793
    #print("Copying table..")
    print("Copying table...")

    start = time.time()
    tableData = data.copy()

    print "Copied table in ", round(time.time()-start,2) ,"seconds. Calculating means in table..."
    start = time.time()
    meanRows = np.zeros(len(tableData))
    meanColumns = np.zeros(len(tableData[0]))
    rowCount = np.zeros(len(tableData))
    columnsCount = np.zeros(len(tableData[0]))


    #check = 0
    for i in range(len(tableData)):

        '''if i * len(tableData[0]) > check:
            check += len(tableData) * len(tableData[0]) * 0.01
            print "Percentage completed: ", round(i * 100.0 / (len(tableData)), 0), "%"'''

        for j in range(len(tableData[0])):
            if tableData[i][j] != 0.0:

                meanRows[i] += tableData[i][j]
                rowCount[i] += 1

                meanColumns[j] += tableData[i][j]
                columnsCount[j] += 1

                avgUserRating = tableData[i][j]


    #print "Now dividing the means ", time.time()-start, " seconds into mean calculation."
    for i in range(len(meanRows)):
        if rowCount[i] > 0:

            if rowCount[i]!=0:
                meanRows[i] = meanRows[i] / rowCount[i]
            else:
                meanRows[i] = 2.5 #<-- Set to the medium rating as a starting point

    for i in range(len(meanColumns)):
        if columnsCount[i] > 0:

            if columnsCount[i]!=0:
                meanColumns[i] = meanColumns[i] / columnsCount[i]
            else:
                meanColumns[i] = 0.0 #<-- Don't recommend movie if now ratings has been given yet
    avgUserRatings = sum(meanRows)/len(meanRows)
    print "Means calculated in ", round(time.time() - start, 2), "seconds. Artificially adding data..."
    start = time.time()
    for i in range(len(tableData)):
        '''if i * len(tableData[0]) > check:
                    check += len(tableData) * len(tableData[0]) * 0.01
                    print "Percentage completed: ", round(i * 100.0 / (len(tableData)), 0), "%"'''
        for j in range(len(tableData[0])):
            if tableData[i][j] == 0.0:
                tableData[i][j] = meanColumns[j] + meanRows[i]-avgUserRatings # Average rating for each movie adjusted
                # by how harsh the user rates compared to the average rating

    print "Added artificial data in  ", round(time.time() - start, 2), " seconds. De-meaning and calculating SVD..."

    # De-mean data
    #print("De-meaning data...")
    user_ratings_mean = np.mean(tableData, axis=1)
    R_demeaned = tableData - user_ratings_mean.reshape(-1, 1)

    # Calculate the SVD

    start = time.time()

    U, sigma, Vt = svds(R_demeaned, k=25) #k=40 means that 40 features of users will be considerd. Thus one user might be 80% user type 1 and 20% user type 2 and so forth.

    print "De-meaned and calculated SVD in ", time.time()-start, "seconds."
    start = time.time()
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    print "Output matrix calculated in ", time.time() - start, "seconds. The ratings are now calculated."

    return all_user_predicted_ratings

def trainIncrementalSVD(data, K=40, steps=10, alpha=0.0002, beta=0.02):

    R = np.array(data)
    N = len(R)
    M = len(R[0])

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    Q = Q.T
    check = 0
    curAlpha = 0.01
    for step in xrange(steps):
        if curAlpha > alpha:
            curAlpha *= 0.90
        #print(curAlpha)
        #print "Step: ", step

        for i in xrange(len(R)):

            if step*len(R)+i > check:
                check += steps*len(R) * 0.001
                print "Percentage completed: ", round((step*len(R)+i) * 100.0 / (steps*len(R)), 2), "%"
            alphaBeta = curAlpha * beta

            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    eAlpha2 = curAlpha * 2 * eij
                    for k in xrange(K):
                        P[i][k] += eAlpha2 * Q[k][j] - alphaBeta * P[i][k]
                        Q[k][j] += eAlpha2 * P[i][k] - alphaBeta * Q[k][j]

        #The code below is used to exit after a certain accuracy. This value does not really need to be calculated.
        '''e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break'''
    nR = np.dot(P, Q)
    return nR

def train(tableData,algs=['SVD']):
    function_mappings = {
        'SVDFull': trainFullSVD,
        'SVDInc': trainIncrementalSVD,
    }

    trainTime = np.zeros(len(algs))
    models = []
    models.append(None)
    models.append(None)

    for i, curAlg in enumerate(algs):
        start = time.time()
        if curAlg=='SVDFull':
            print "Training ", curAlg, "..."
            models[0] = function_mappings[curAlg](tableData)
        elif curAlg=='SVDInc':
            print "Training ", curAlg,"..."
            models[1] = function_mappings[curAlg](tableData,K=25,steps=5000)
        trainTime[i] += time.time() - start
    print "Training done."
    return trainTime, models

def euclidean_similarity(data,person1, person2):

	common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
	rankings = [(data[person1][itm], data[person2][itm]) for itm in common_ranked_items]
	distance = [pow(rank[0] - rank[1], 2) for rank in rankings]

	return 1 / (1 + sum(distance))

def pearson_similarity(data,person1, person2):

    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]

    n = len(common_ranked_items)

    s1 = sum([data[person1][item] for item in common_ranked_items])
    try:
        s2 = sum([data[person2][item] for item in common_ranked_items])
    except:
        print person2, " ", item, " ", data[person2][item]
        print common_ranked_items

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

    #print(person)

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

def general_popularity(data,movieMap, item):

    meanScore = 0.0
    meanCount = 0

    for i in range(len(data)):
        curScore = data[i][movieMap[item]]
        if curScore != 0:
            meanScore += curScore
            meanCount += 1

    meanScore = meanScore/meanCount if meanCount > 0 else 0

    return meanScore

def randomItem():
    return random.random()

def SVD(model,userMap,movieMap,userID,itemID):
    # Get and sort the user's predictions
    return model[userMap[userID]][movieMap[itemID]]