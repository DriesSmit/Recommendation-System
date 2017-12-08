import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

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

def recommend(data,person, item,bound, similarity=pearson_similarity):
    scores = [(similarity(data,person, other), other) for other in data if other != person]

    scores.sort()
    scores.reverse()
    scores = scores[0:bound]

    print(scores)

    recomms = (0.0,0.0)

    for sim, other in scores:
        ranked = data[other]

        if item in ranked:
            weight = sim * ranked[item]

            curSim, curW = recomms

            recomms = (curSim + sim, curW + weight)

    sim, weight = recomms
    recomms = weight / sim
    return recomms

def  evaluate(data,testPerAlg,algs=['euclidean_similarity','pearson_similarity']):
    function_mappings = {
        'euclidean_similarity': euclidean_similarity,
        'pearson_similarity'  : pearson_similarity,
    }
    print("Value: ",data['Dries Smit']['MovieD'])
    results = {}

    for i in range(len(algs)):
        results[str(algs[i])] = 0.0
        for test in range(testPerAlg):
            rec = recommend(data,"Dries Smit", 'MovieD', 1, function_mappings[algs[i]])
            #I don't think mean squared error is needed yet
            results[str(algs[i])] += abs(rec-data['Dries Smit']['MovieD'])
        results[str(algs[i])] /= testPerAlg

    return results

data = {
    'Alan Perlis': {
        'MovieA': 3.0,
        'MovieB': 5.0,
        'MovieC': 4.0,
        'MovieD': 2.0
    },

    'Dries Smit': {
        'MovieA': 2.0,
        'MovieB': 4.0,
        'MovieC': 4.0,#,
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
        'MovieD': 4.0,
    },
}

algs = ['euclidean_similarity','pearson_similarity']
result = evaluate(data,1,algs=algs)

print "Results: ", result.values()
values = result.values()

x = np.arange(len(algs))
plt.bar(x, height= values)
plt.xticks(x+.4, algs)
plt.title("Error in ratings different algorithms(0 is best)")
plt.xlabel("Algorithm used")
plt.ylabel("Average error in ratings")
plt.show()
