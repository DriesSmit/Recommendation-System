import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import time
import RecommendationSystem as rs
#1.) Create Testbench, RecommendationSystem
#1.) Test on a bigger dataset
#2.) Take each algrithm to there seperate classes
#3.) Work on report with results


def createData(size=1):
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

def  evaluate(data,models,testPerAlg,algs=['euclidean_similarity','pearson_similarity']):
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

            if curAlg=="euclidean_similarity" or curAlg=="cosine_similarity":
                rec = rs.recommend(data,memberID, itemID, limit, function_mappings[curAlg])
            elif curAlg=="pearson_similarity":
                rec = rs.general_popularity(tableData,memberID-1, itemID-1)
            elif curAlg=="SVD":
                rec = rs.SVD(tableData,models[0], memberID-1, itemID-1)#toets deur om tableData in te vat en te kyk of dit decrease
            elif curAlg=="randomItem":
                rec = rs.randomItem()
            else:
                print("Incorrect algorithm name entered.")
            runTime[i] += time.time()-start

            # I don't think mean squared error is needed yet. But it is a good practice to implement when actually
            # using the system.

            '''if curAlg=="euclidean_similarity":
                print("Alg rec: ",rec,". True ans: ",data[memberID][itemID])'''

            results[i] += abs(rec-data[memberID][itemID])
    results /= testPerAlg
    return results,runTime

hashData,tableData,testHashData = createData()

algs = ['pearson_similarity','SVD','general_popularity','euclidean_similarity','cosine_similarity','randomItem']#['euclidean_similarity','pearson_similarity','general_popularity','SVD']

trainTimes,models = rs.train(tableData,algs=algs) #Train all the algorithms

result,runTime = evaluate(testHashData,models,100,algs=algs) #Test all the algorithms

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