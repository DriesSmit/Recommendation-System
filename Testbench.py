import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
import time
from scipy.sparse.linalg import svds

#1.) Test on a bigger dataset
#2.) Take each algrithm to there seperate classes
#3.) Work on report with results

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



def createData():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    numUsers = 0
    numMovies = 0

    with open('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/u.user') as myfile:
        numUsers = sum(1 for line in myfile)

    with open('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/u.item') as myfile:
        numMovies = sum(1 for line in myfile)


    #Create training hash and normal table
    ratings = pd.read_csv('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/u1.base', sep='\t', names=r_cols,
                          encoding='latin-1')
    sortedRate = ratings.sort_values([('user_id')], ascending=True)
    values = sortedRate.values

    hashTrainData = {}

    idCount = values[0][0]

    tempHash = {}

    tableTrainData = np.zeros((numUsers, numMovies))#Initialize any value not known to 2.5

    #A mapping algorithm might be needed if there are missing user ID's or movie ID's e.g. 1,2,3,5,6

    meanRows = np.zeros(len(tableTrainData))
    meanColumns = np.zeros(len(tableTrainData[0]))
    rowCount = np.zeros(len(tableTrainData))
    columnsCount = np.zeros(len(tableTrainData[0]))

    for i in range(len(sortedRate)):
        tableTrainData[values[i][0]-1][values[i][1]-1] = values[i][2]

        meanRows[values[i][0]-1] += values[i][2]
        rowCount[values[i][0]-1] += 1

        meanColumns[values[i][1] - 1] += values[i][2]
        columnsCount[values[i][1] - 1] += 1

        if(values[i][0]==idCount):
            tempHash[values[i][1]] = values[i][2]
        else:
            hashTrainData[values[i-1][0]] = tempHash
            tempHash = {}
            tempHash[values[i][1]] = values[i][2]
            idCount = values[i][0]
    hashTrainData[values[i][0]] = tempHash

    for i in range(len(meanRows)):
        if rowCount[i]>0:
            meanRows[i] = meanRows[i]/rowCount[i]

    for i in range(len(meanColumns)):
        if columnsCount[i]>0:
            meanColumns[i] = meanColumns[i]/columnsCount[i]
    #print(meanRows, " ", meanColumns)
    #print("Columns len: ", len(meanColumns), " ", len(tableTrainData[0]))

    for i in range(len(tableTrainData)):
        for j in range(len(tableTrainData[0])):
            if tableTrainData[i][j]==0.0:
                value = (meanRows[i]+meanColumns[j])/2.0
                #print(value)
                tableTrainData[i][j] = value
    #For initialisation the average betwean mean Columns and Rows was found to be the best values to use.
    #For a k=100 value and using the testHash set for u1
    # Row avg:      1.01
    # Column avg:   0.8573
    # Mix avg:      0.793

    # Create test hash table
    ratings = pd.read_csv('/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/u1.test', sep='\t', names=r_cols,
                          encoding='latin-1')
    sortedRate = ratings.sort_values([('user_id')], ascending=True)
    values = sortedRate.values

    hashTestData = {}

    idCount = values[0][0]

    tempHash = {}

    # A mapping algorithm might be needed if there are missing user ID's or movie ID's e.g. 1,2,3,5,6
    for i in range(len(sortedRate)):
        if (values[i][0] == idCount):
            tempHash[values[i][1]] = values[i][2]
        else:
            hashTestData[values[i - 1][0]] = tempHash
            tempHash = {}
            tempHash[values[i][1]] = values[i][2]
            idCount = values[i][0]
    hashTestData[values[i][0]] = tempHash


    return hashTrainData,tableTrainData,hashTestData

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

def general_popularity(data,person1, person2):
    return 1.0

def randomItem():
    return random.random()

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

def  evaluate(data,models,testPerAlg,algs=['euclidean_similarity','pearson_similarity']):
    print("Testing algorithms:")

    function_mappings = {
        'euclidean_similarity': euclidean_similarity,
        'pearson_similarity'  : pearson_similarity,
        'general_popularity': general_popularity,
        'cosine_similarity': cosine_similarity,
        'randomItem': randomItem,
        'SVD': SVD,
    }
    #print("Value: ",data['Dries Smit']['MovieD'])
    results = np.zeros(len(algs))
    runTime = np.zeros(len(algs))
    for test in range(testPerAlg):

        print "Percentage completed: ", round(test*100.0/testPerAlg,2), "%"

        memberIndex = int((len(data) * random.random()))
        memberID = data.keys()[memberIndex]
        #print "Member: ", memberIndex, " ", memberID
        itemIndex = int((len(data[memberID]) * random.random()))
        itemID = data[memberID].keys()[itemIndex]
        #print "Item: ", itemIndex, " ", itemID

        #memberName = 'John Connor'
        #itemName = 'MovieB'

        for i,curAlg in enumerate(algs):
            start = time.time()
            limit = 150

            if curAlg=="euclidean_similarity" or curAlg=="pearson_similarity" or curAlg=="general_popularity" or curAlg=="cosine_similarity":
                if curAlg=="general_popularity":
                    limit = len(data)

                rec = recommend(data,memberID, itemID, limit, function_mappings[curAlg])
            elif curAlg=="SVD":
                rec = SVD(tableData,models[0], memberID-1, itemID-1)#toets deur om tableData in te vat en te kyk of dit decrease
            elif curAlg=="randomItem":
                rec = randomItem()
            else:
                print("Incorrect algorithm name entered.")
            runTime[i] += time.time()-start

            # I don't think mean squared error is needed yet. But it is a good practice to implement when actually
            # using the system.

            if curAlg=="euclidean_similarity":
                print("Alg rec: ",rec,". True ans: ",data[memberID][itemID])

            results[i] += abs(rec-data[memberID][itemID])
    results /= testPerAlg
    return results,runTime

hashData,tableData,testHashData = createData()

algs = ['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity','randomItem']#['euclidean_similarity','pearson_similarity','general_popularity','SVD']

trainTimes,models = train(tableData,algs=algs) #Train all the algorithms

result,runTime = evaluate(testHashData,models,100,algs=algs) #Test all the algorithms  #Maak testHasData

print "Results: ", result

x = np.arange(len(algs))
plt.figure("Score")
plt.bar(x, height= result)
plt.xticks(x+.4, algs)
plt.title("Error in ratings for different algorithms(0 is best)")
plt.xlabel("Algorithm used")
plt.ylabel("Average error in ratings")

plt.figure("Runtime")
plt.bar(x, height= runTime)
plt.xticks(x+.4, algs)
plt.title("Run time for different algorithms(0 is best)")
plt.xlabel("Algorithm used")
plt.ylabel("Run time in seconds")

plt.figure("Training time")
plt.bar(x, height= trainTimes)
plt.xticks(x+.4, algs)
plt.title("Training time for each algorithm")
plt.xlabel("Algorithm used")
plt.ylabel("Training time in seconds")
plt.show()