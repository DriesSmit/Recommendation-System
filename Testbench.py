import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import RecommendationSystem as rs
#1.) Test on hiGuru type of dataset.
#3.) Also test on link type of dataset.
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

def createData20ml(ratingsLoc,movieLoc):
    print("Loading dataset...")
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    numUsers = 0
    numMovies = 0

    train_test_ratio = 0.9 # 0.8 = 80% training data and 20% test data
    # Count movies and build map
    numMovies = 0
    movieMap = {}
    with open(movieLoc) as myfile:
        for i,aline in enumerate(myfile.readlines()):
            if i>0:
                values = aline.split(",")
                movieMap[int(values[0])] = numMovies
                numMovies += 1

    #Create training hash and normal table
    ratings = pd.read_csv(ratingsLoc, sep=',', names=r_cols,
                          encoding='latin-1')
    sortedRate = ratings.drop(ratings.index[[0]]).sort_values([('user_id')], ascending=True)
    values = sortedRate.values

    # Count users and build map
    userMap = {}

    hashTrainData = {}
    hashTestData = {}

    idCount = values[0][0]

    tempTrainHash = {}
    tempTestHash = {}
    takeRate = int(1.0 / (1.0 - train_test_ratio))
    testCount = 0
    trainCount = 0
    userMap[int(values[0][0])] = 0
    tableTrainData = np.zeros((138493, 27278), dtype=np.float16)
    valueMap = []

    nextCheck = 0

    for i in range(len(values)):
        #print(float(i)/float(len(values)))
        if float(i) > nextCheck:
            nextCheck += float(len(values))*0.01
            print "Percentage completed: ", round(i * 100.0 / len(values), 2), "%"

        if(values[i][0]==idCount):
            if i%takeRate>0:
                tempTrainHash[int(values[i][1])] = float(values[i][2])
                trainCount +=1
            else:
                tempTestHash[int(values[i][1])] = float(values[i][2])
                testCount += 1
        else:
            userMap[int(values[i][0])] = numUsers
            numUsers += 1
            hashTrainData[int(values[i-1][0])] = tempTrainHash
            hashTestData[int(values[i - 1][0])] = tempTestHash
            tempTrainHash = {}
            tempTestHash = {}
            if i%takeRate>0:
                tempTrainHash[int(values[i][1])] = float(values[i][2])
                trainCount += 1
            else:
                tempTestHash[int(values[i][1])] = float(values[i][2])
                testCount += 1

            idCount = values[i][0]
        if i % takeRate > 0:
            valueMap.append((userMap[int(values[i][0])],movieMap[int(values[i][1])]))
            tableTrainData[userMap[int(values[i][0])]][movieMap[int(values[i][1])]] = float(values[i][2])
    hashTrainData[int(values[i][0])] = tempTrainHash
    hashTestData[int(values[i][0])] = tempTestHash

    print "Number of users: ", numUsers, ". Number of movies: ",numMovies
    print "Number of ratings: ",len(sortedRate),". Number of training ratings: ",trainCount, ". Number of test ratings: ",testCount

    return hashTrainData,tableTrainData,hashTestData,userMap,movieMap,valueMap

def createData(ratingsLoc,userLoc,movieLoc,seperator,seperator2='::'):
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
            values = aline.split(seperator2)
            userMap[int(values[0])] = numUsers
            numUsers += 1

    # Count movies and build map
    numMovies = 0
    movieMap = {}
    with open(movieLoc) as myfile:
        for aline in myfile.readlines():
            values = aline.split(seperator2)
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
    tableTrainData = np.zeros((numUsers, numMovies))#Initialize any value not known to 2.5
    valueMap = []
    takeRate = int(1.0/(1.0-train_test_ratio))
    testCount = 0
    trainCount = 0
    #A mapping algorithm might be needed if there are missing user ID's or movie ID's e.g. 1,2,3,5,6
    for i in range(len(values)):
        if i % takeRate > 0:
            valueMap.append((userMap[int(values[i][0])],movieMap[int(values[i][1])]))
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

    return hashTrainData,tableTrainData,hashTestData,userMap,movieMap,valueMap

def  evaluate(trainHashData,trainTableData,testHasData,models,userMap,movieMap,testPerAlg=1000,algs=['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity','randomItem']):
    print("Testing algorithms:")

    function_mappings = {
        'euclidean_similarity': rs.euclidean_similarity,
        'pearson_similarity'  : rs.pearson_similarity,
        'general_popularity': rs.general_popularity,
        'cosine_similarity': rs.cosine_similarity,
        'randomItem': rs.randomItem,
    }

    results = np.zeros(len(algs))
    runTime = np.zeros(len(algs))
    check = 0
    for test in range(testPerAlg):
        if test >= check:
            check += testPerAlg * 0.01
            print "Percentage completed: ", round(test*100.0/testPerAlg,2), "%"

        memberIndex = int((len(testHasData) * random.random()))
        memberID = testHasData.keys()[memberIndex]
        while len(testHasData[memberID])==0:
            memberIndex = int((len(testHasData) * random.random()))
            memberID = testHasData.keys()[memberIndex]

        itemIndex = int((len(testHasData[memberID]) * random.random()))
        itemID = testHasData[memberID].keys()[itemIndex]

        for i,curAlg in enumerate(algs):
            start = time.time()
            limit = 150
            # It doesn't look like there is a case statement alternative in python. Further will be faster to store the
            # tests in an array and then run it separately for each algorithm, but this format looks better.
            if curAlg=="euclidean_similarity" or curAlg=="pearson_similarity" or curAlg=="cosine_similarity":   #K nearest neighbors algorithms
                rec = rs.recommend(trainHashData,memberID, itemID, limit, function_mappings[curAlg])
            elif curAlg=="general_popularity":
                rec = rs.general_popularity(trainTableData,movieMap, itemID)
            elif curAlg=="SVDFull":
                rec = rs.tableSVD(models[0],userMap,movieMap, memberID, itemID)
            elif curAlg=="SVDInc":
                rec = rs.incSVD(models[1],userMap,movieMap, memberID, itemID)
            elif curAlg=="SVDFullInc":
                rec = rs.incSVD(models[2],userMap,movieMap, memberID, itemID)
            elif curAlg=="randomItem":
                rec = rs.randomItem()
            else:
                print "Incorrect algorithm name entered: ", curAlg
            runTime[i] += time.time()-start
            '''if curAlg=="euclidean_similarity":
                print("Alg rec: ",rec,". True ans: ",data[memberID][itemID])'''

            results[i] += abs(rec-testHasData[memberID][itemID])     # Mean absolute error(MAE)
            #results[i] += pow(rec - testHasData[memberID][itemID],2)# Mean square error(RMS)
    results /= testPerAlg
    return results,runTime
# Database of 20m
dir = '/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-20m/'
ratingsFile = dir + 'ratings.csv'
userLocation = dir + 'users.csv'
movieLocation = dir + 'movies.csv'

# Database of 1m
'''dir = '/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-1m/'
ratingsFile = dir + 'ratings.dat'
userLocation = dir + 'users.dat'
movieLocation = dir + 'movies.dat'
sep2 = sep = '::'''

# Database of 100k
'''dir = '/home/dries/dev/RecommendationSystem/Data/MovieLens/ml-100k/'
ratingsFile = dir + 'u.data'
userLocation = dir + 'u.user'
movieLocation = dir + 'u.item'
sep = '\t'
sep2 = '|'''

trainHashData, trainTableData,testHashData,userMap,movieMap,valueMap = createData20ml(ratingsFile,movieLocation)
#trainHashData, trainTableData,testHashData,userMap,movieMap,valueMap = createData(ratingsFile,userLocation,movieLocation,seperator=sep,seperator2=sep2)

# ['pearson_similarity','SVDFull','SVDFullInc','SVDInc','general_popularity','euclidean_similarity','cosine_similarity','randomItem']
#algs = ['SVDFull','SVDFullInc','SVDInc']
algs = ['SVDInc','general_popularity']
trainTimes,models = rs.train(trainTableData,valueMap,algs=algs,iter=2) #Train all the algorithms

result,runTime = evaluate(trainHashData,trainTableData,testHashData,models,userMap,movieMap,testPerAlg=500,algs=algs) #Test all the algorithms
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