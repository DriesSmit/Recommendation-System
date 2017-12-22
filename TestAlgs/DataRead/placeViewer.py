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

for i in range(len(location)):

        if location[i][2]!=None and location[i][3]!=None:
            x, y = map(float(location[i][2]), float(location[i][3]))
            map.plot(x, y, 'bo', markersize=3)
plt.show()