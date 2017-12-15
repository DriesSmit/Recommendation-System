import math
import random
import numpy as np
import time
from scipy.sparse.linalg import svds

#Comments
#Please refer to Report.odt for a more detailed explanation and result page
#of each algorithm

def trainFullSVD(data,demean=True):
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

    # Calculate the SVD

    start = time.time()

    if demean:
        user_ratings_mean = np.mean(tableData, axis=1)
        R_demeaned = tableData - user_ratings_mean.reshape(-1, 1)
        # Calculate the SVD
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    else:
        U, sigma, Vt = svds(tableData, k=25) #k=40 means that 40 features of users will be considerd. Thus one user might be 80% user type 1 and 20% user type 2 and so forth.
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    print "Calculated SVD in ", time.time()-start, "seconds."

    mae = 0.0
    rse = 0.0
    count = 0
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            if data[i][j] > 0:
                mae += abs(data[i][j] - all_user_predicted_ratings[i][j])  # Mean absolute error(MAE)
                rse += pow(data[i][j] - all_user_predicted_ratings[i][j],2)  # Mean absolute error(MAE)
                count += 1
    mae = mae / count
    rse = rse / count
    print "FullSVD: Mean absolute error: ", round(mae, 3),". Root square error: ",round(rse, 3)

    return all_user_predicted_ratings,U, sigma, Vt


def trainIncrementalSVD(data,valueMap, K=40, steps=10, alpha=0.0002, beta=0.02, alphaReg=True, Q=None, P=None):

    print "Len valueMap: ",len(valueMap)
    #print valueMap

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

    # loopTime = 0.0
    # otherTime = 0.0
    for step in xrange(steps):

        print "Step: ", step

        if curAlpha > alpha:
            curAlpha *= 0.90

        if step > check:
            #print("Start")
            check += steps * 0.01

            mae = 0.0
            rse = 0.0
            for valLoc in valueMap:
                mae += abs(data[valLoc[0]][valLoc[1]] - np.dot(P[valLoc[0],:],Q[:,valLoc[1]]))  # Mean absolute error(MAE)
                rse += pow(data[valLoc[0]][valLoc[1]] - np.dot(P[valLoc[0],:],Q[:,valLoc[1]]), 2)  # Mean absolute error(MAE)
            mae = mae / len(valueMap)
            rse = rse / len(valueMap)
            perc = round(step * 100.0 / steps, 2)

            print "Percentage completed: ", perc, "%. Mean absolute error: ", round(mae,3), ". Root square error: ", round(rse, 3)
            if mae < 0.001:
                break

        alphaBeta = curAlpha * beta
        # otherTime += time.time()-start
        # start = time.time()
        # print("Ready")
        for valLoc in valueMap:
            eij = data[valLoc[0]][valLoc[1]] - np.dot(P[valLoc[0], :], Q[:, valLoc[1]])
            eAlpha2 = curAlpha * 2 * eij
            for k in xrange(K):
                P[valLoc[0]][k] += eAlpha2 * Q[k][valLoc[1]] - alphaBeta * P[valLoc[0]][k]
                Q[k][valLoc[1]] += eAlpha2 * P[valLoc[0]][k] - alphaBeta * Q[k][valLoc[1]]
        # loopTime += time.time() - start

    # print "Looptime: ", loopTime, ". Other: ", otherTime

    nR = np.dot(P, Q)
    return nR

def trainFullIncSVD(data,valueMap, K=40, steps=10, alpha=0.0002, beta=0.02):#Combine FullInc to use both the previous functions. Save code
    ratingSVDFull,U, sigma, Q = trainFullSVD(data,demean=False)
    e = 0
    count = 0
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            if data[i][j] > 0:
                e += abs(data[i][j] - ratingSVDFull[i][j])  # Mean absolute error(MAE)
                count += 1
    mae = e / count
    print "Mean absolute error: ", round(mae, 3)

    P = np.dot(U,sigma)
    Q = np.array(Q)

    #print "Q shape: ", Q.shape
    #print "P shape: ",P.shape

    nR = trainIncrementalSVD(data,valueMap, K=K, steps=steps, alpha=alpha, beta=beta,alphaReg=False, Q=Q, P=P)

    mae = 0.0
    rse = 0.0
    count = 0
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            if data[i][j] > 0:
                mae += abs(data[i][j] - nR[i][j])  # Mean absolute error(MAE)
                rse += pow(data[i][j] - nR[i][j], 2)  # Mean absolute error(MAE)
                count += 1
    mae = mae / count
    rse = rse / count
    print "FullIncSVD: Mean absolute error: ", round(mae, 3), ". Root square error: ", round(rse, 3)
    return nR

def train(tableData,valueMap,algs=['SVD']):
    function_mappings = {
        'SVDFull': trainFullSVD,
        'SVDInc': trainIncrementalSVD,
        'SVDFullInc': trainFullIncSVD,
    }

    trainTime = np.zeros(len(algs))
    models = []
    models.append(None)
    models.append(None)
    models.append(None)

    for i, curAlg in enumerate(algs):
        #print(curAlg)
        start = time.time()
        if curAlg=='SVDFull':
            print "Training ", curAlg, "..."
            model,_,_,_ = function_mappings[curAlg](tableData)
            models[0] = model
        elif curAlg=='SVDInc':
            print "Training ", curAlg,"..."
            models[1] = function_mappings[curAlg](tableData,valueMap,K=25,steps=5)
        elif curAlg=='SVDFullInc':
            print "Training ", curAlg,"..."
            models[2] = function_mappings[curAlg](tableData,valueMap,K=25,steps=5,alpha=0.000001)
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