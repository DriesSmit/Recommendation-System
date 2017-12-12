import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import RecommendationSystem as rs
#1.) Test on a bigger dataset
#2.) Work on report with results
#3.) Try bigger machine leaning algorithm
def createOldData():
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
    for i in range(len(sortedRate)):
        tableTrainData[values[i][0]-1][values[i][1]-1] = values[i][2]

        if(values[i][0]==idCount):
            tempHash[values[i][1]] = values[i][2]
        else:
            hashTrainData[values[i-1][0]] = tempHash
            tempHash = {}
            tempHash[values[i][1]] = values[i][2]
            idCount = values[i][0]
    hashTrainData[values[i][0]] = tempHash

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

def createData(ratingsLoc,userLoc,movieLoc,seperator):
    print("Loading dataset...")
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    numUsers = 0
    numMovies = 0

    train_test_ratio = 0.9 # 0.8 = 80% training data and 20% test data

    # Count users and build map
    numUsers = 0
    userMap = {}
    with open(userLoc) as myfile:
        for aline in myfile.readlines():
            values = aline.split("::")
            userMap[int(values[0])] = numUsers
            numUsers += 1

    # Count movies and build map
    numMovies = 0
    movieMap = {}
    with open(movieLoc) as myfile:
        for aline in myfile.readlines():
            values = aline.split("::")
            movieMap[int(values[0])] = numMovies
            numMovies += 1


    #Create training hash and normal table
    ratings = pd.read_csv(ratingsLoc, sep=seperator, names=r_cols,
                          encoding='latin-1')
    sortedRate = ratings.sort_values([('user_id')], ascending=True)
    values = sortedRate.values

    hashTrainData = {}
    hashTestData = {}

    idCount = values[0][0]

    tempTrainHash = {}
    tempTestHash = {}
    #numMovies = 3952 #Hard hack
    tableTrainData = np.zeros((numUsers, numMovies))#Initialize any value not known to 2.5

    takeRate = int(1.0/(1.0-train_test_ratio))
    testCount = 0
    trainCount = 0
    #A mapping algorithm might be needed if there are missing user ID's or movie ID's e.g. 1,2,3,5,6
    for i in range(len(sortedRate)):
        tableTrainData[userMap[values[i][0]]][movieMap[values[i][1]]] = values[i][2]

        if(values[i][0]==idCount):
            if i%takeRate>0:
                tempTrainHash[values[i][1]] = values[i][2]
                trainCount +=1
            else:
                tempTestHash[values[i][1]] = values[i][2]
                testCount += 1
        else:
            hashTrainData[values[i-1][0]] = tempTrainHash
            hashTestData[values[i - 1][0]] = tempTestHash
            tempTrainHash = {}
            tempTestHash = {}
            if i%takeRate>0:
                tempTrainHash[values[i][1]] = values[i][2]
                trainCount += 1
            else:
                tempTestHash[values[i][1]] = values[i][2]
                testCount += 1

            idCount = values[i][0]
    hashTrainData[values[i][0]] = tempTrainHash
    tempTestHash[values[i][0]] = tempTestHash
    print "Number of users: ", numUsers, ". Number of movies: ",numMovies
    print "Number of ratings: ",len(sortedRate),". Number of training ratings: ",trainCount, ". Number of test ratings: ",testCount

    #print(hashTrainData)
    #print(hashTestData)

    return hashTrainData,tableTrainData,hashTestData,userMap,movieMap

def  evaluate(trainHashData,trainingTableData,testHasData,models,userMap,movieMap,testPerAlg,algs=['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity','randomItem']):
    print("Testing algorithms:")

    function_mappings = {
        'euclidean_similarity': rs.euclidean_similarity,
        'pearson_similarity'  : rs.pearson_similarity,
        'general_popularity': rs.general_popularity,
        'cosine_similarity': rs.cosine_similarity,
        'randomItem': rs.randomItem,
        'SVD': rs.SVD,
    }
    #print("Value: ",data['Dries Smit']['MovieD'])
    results = np.zeros(len(algs))
    runTime = np.zeros(len(algs))
    for test in range(testPerAlg):

        print "Percentage completed: ", round(test*100.0/testPerAlg,2), "%"

        memberIndex = int((len(testHasData) * random.random()))
        memberID = testHasData.keys()[memberIndex]
        #print "Member: ", memberIndex, " ", memberID
        itemIndex = int((len(testHasData[memberID]) * random.random()))
        itemID = testHasData[memberID].keys()[itemIndex]
        #print "Item: ", itemIndex, " ", itemID

        #memberName = 'John Connor'
        #itemName = 'MovieB'

        for i,curAlg in enumerate(algs):
            start = time.time()
            limit = 150
            #print(trainData)
            if curAlg=="euclidean_similarity" or curAlg=="pearson_similarity" or curAlg=="cosine_similarity":
                rec = rs.recommend(trainHashData,memberID, itemID, limit, function_mappings[curAlg])
            elif curAlg=="general_popularity":
                rec = rs.general_popularity(trainingTableData,movieMap, itemID-1)
            elif curAlg=="SVD":
                rec = rs.SVD(models[0],userMap,movieMap, memberID, itemID)#toets deur om tableData in te vat en te kyk of dit decrease
            elif curAlg=="randomItem":
                rec = rs.randomItem()
            else:
                print "Incorrect algorithm name entered: ",curAlg
            runTime[i] += time.time()-start

            # I don't think mean squared error is needed yet. But it is a good practice to implement when actually
            # using the system.

            '''if curAlg=="euclidean_similarity":
                print("Alg rec: ",rec,". True ans: ",data[memberID][itemID])'''

            results[i] += abs(rec-testHasData[memberID][itemID])
    results /= testPerAlg
    return results,runTime

# Database of 1m
dir = '/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-1m/'
ratingsFile = dir + 'ratings.dat'
userLocation = dir + 'users.dat'
movieLocation = dir + 'movies.dat'
sep = '::'

# Database of 100k
'''dir = '/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/'
ratingsFile = dir + 'u.data'
userLocation = dir + 'u.user'
movieLocation = dir + 'u.item'
sep = '\t'''

trainHashData, trainTableData,testHashData,userMap,movieMap = createData(ratingsFile,userLocation,movieLocation,seperator=sep)
#trainHashData, trainTableData,testHashData = createOldData()

algs = ['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity']#['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity','randomItem']
trainTimes,models = rs.train(trainTableData,algs=algs) #Train all the algorithms

result,runTime = evaluate(trainHashData,trainTableData,testHashData,models,userMap,movieMap,100,algs=algs) #Test all the algorithms

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