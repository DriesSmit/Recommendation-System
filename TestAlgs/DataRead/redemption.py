#Database: ignite_production
#Username: python
#Password: pythonAdmin@#1
import mysql.connector
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def binary_search(array, target):
    lower = 0
    upper = len(array)
    while lower < upper:   # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:   # this two are the actual lines
                break        # you're looking for
            lower = x
        elif target < val:
            upper = x

areaLength = 50.0
cenLon = 18.4232200
cenLat = -33.9258400

map = Basemap(projection='robin', lat_0=0, lon_0=-100,
              resolution='l', area_thresh=1000.0)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral')
map.drawmapboundary()


conn = mysql.connector.connect (user='python', password='pythonAdmin@#1',
                               host='localhost',buffered=True)
cursor = conn.cursor()
command = ("use ignite_production;")
cursor.execute(command)


#Store places
command = ("select id, merchant_id, lng,lat from places ORDER BY id")
cursor.execute(command)

location = []
for (line) in cursor:
    location.append(line)

#Store redemptions
command = ("select place_id from redemptions")
cursor.execute(command)

redemptions = []
for (line) in cursor:
    redemptions.append(line)

#Store members
'''command = ("select id from members")
cursor.execute(command)

members = []
for (line) in cursor:
    members.append(line)

print "Number of members: ", len(members)

memPlaces = [[] for i in range(len(members))]'''

#Plot redemption locations
'''for i in range(len(location)):
    memPlaces[location[i][0]].append(location[i][1])

    if len(memPlaces[location[i][0]])==2:
        print(location[i][0])

print memPlaces[location[i][0]]'''

placeIDArr = [location[i][0] for i in range(len(location))]
#print "Place IDs: ", placeIDArr


redempPlace = 0
placeTrace = 0
placeLoc = 0
for i in range(len(redemptions)):

    if redemptions[i]!=None:

        placeID = redemptions[i][0]

        #print("PlaceID: ", placeID)
        #print("Location: ", placeIDArr)

        if placeID != None:

            redempPlace += 1

        placeIndex = binary_search(placeIDArr,placeID)

        #print "Result: ", placeIndex, ". placeID: ", placeID

        if placeIndex != None:

            placeTrace += 1

            if location[placeIndex][2]!=None and location[placeIndex][3]!=None:
                placeLoc += 1
                x, y = map(float(location[placeIndex][2]), float(location[placeIndex][3]))
                map.plot(x, y, 'bo', markersize=3)
print "Number of redemptions: ", len(redemptions)
print "Number of redemptions with placeIDs: ",redempPlace
print "Number of redemptions that the place exist in the database: ",placeTrace
print "Number of redemptions with places that has a location in the database:", placeLoc
plt.show()

