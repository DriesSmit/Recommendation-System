import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
from sklearn.cluster import KMeans

def createData():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                          encoding='latin-1')

    ratings = ratings.drop('unix_timestamp', 1)

    #ratings = ratings[:25]

    sortedRate = ratings.sort_values([('user_id')], ascending=True)

    print(sortedRate)


    #print(sortedRate)
    values = sortedRate.values

    #values = values[:,0:]
    #print(values)
    data = {}

    idCount = values[0][0]

    tempHash = {}

    for i in range(len(sortedRate)):
        #print(values[i][0])
        if(values[i][0]==idCount):
            tempHash[values[i][1]] = values[i][2]
        else:
            data[values[i-1][0]] = tempHash
            tempHash = {}
            tempHash[values[i][1]] = values[i][2]
            idCount = values[i][0]
    data[values[i][0]] = tempHash
    #print(data)



    '''data = {
        'Alan Perlis': {
            'MovieA': 3.0,
            'MovieB': 5.0,
            'MovieC': 4.0,
            'MovieD': 2.0
        },

        'Dries Smit': {
            'MovieA': 2.0,
            'MovieB': 4.0,
            'MovieC': 4.0,  # ,
            'MovieD': 2.0
        },

        'Matthew Mcconaughey': {
            'MovieA': 1.0,
            'MovieB': 3.0,
            'MovieC': 3.0,
            'MovieD': 1.0
        },

        'John Connor': {
            'MovieA': 3.0,
            'MovieB': 4.0,
            'MovieC': 3.0,
            'MovieD': 4.1,
        },
    }'''

    return data

# Not good if one person consistently rates more harsly
def euclidean_similarity(data,person1, person2):

	common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
	rankings = [(data[person1][itm], data[person2][itm]) for itm in common_ranked_items]
	distance = [pow(rank[0] - rank[1], 2) for rank in rankings]

	return 1 / (1 + sum(distance))

# This allows system finds stronger correlation even if one person rates is consistently harsher or kinder than the
# other.
# The system only considers commonly ranked items without a penalty for the amount of commonly ranked items.
# This means a person with only two movies rated can significantly influence another persons recommendations.
# The systems also scores perfect correlation if there is two or less commonly score items, which will have to be
# compensated for.
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

def average(data,person1, person2):

    return 1.0

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

def  evaluate(data,testPerAlg,algs=['euclidean_similarity','pearson_similarity']):
    function_mappings = {
        'euclidean_similarity': euclidean_similarity,
        'pearson_similarity'  : pearson_similarity,
        'average': average,
    }
    #print("Value: ",data['Dries Smit']['MovieD'])
    results = np.zeros(len(algs))

    for test in range(testPerAlg):

        print "Percentage completed: ", round(test*100.0/testPerAlg,2), "%"

        randMember = int((len(data) * random.random()))
        memberName = data.keys()[randMember]
        #print "Random member: ", memberName
        randItem = int((len(data[memberName]) * random.random()))
        itemName = data[memberName].keys()[randItem]
        #print "Random item: ", itemName

        #memberName = 'John Connor'
        #itemName = 'MovieB'


        #Maak average algorithm reg dat hy nie BOUNDED IS NIE!!!!!!!!!!!!!!!!!!!
        for i,curAlg in enumerate(algs):
            rec = recommend(data,memberName, itemName, 150, function_mappings[curAlg])
            # I don't think mean squared error is needed yet. But it is a good practice to implement when actually
            # using the system.
            results[i] += abs(rec-data[memberName][itemName])
    results /= testPerAlg
    return results

data = createData()

algs = ['euclidean_similarity','pearson_similarity','average']
result = evaluate(data,1000,algs=algs)

print "Results: ", result

x = np.arange(len(algs))
plt.bar(x, height= result)
plt.xticks(x+.4, algs)
plt.title("Error in ratings for different algorithms(0 is best)")
plt.xlabel("Algorithm used")
plt.ylabel("Average error in ratings")
plt.show()