from datetime import datetime
import numpy as np

START_TIME = datetime.now()

# one of the following lines is uncommented before execution
#s = np.float64(1)  #Runtime: 0:00:04.174606
#s = np.float32(1)  #Runtime: 0:00:04.271724
s = np.float16(1)  #Runtime: 0:00:04.293526
#s = 1.0             #Runtime: 0:00:01.717769

for i in range(10000000):
    s = (s + 8) * s % 2399232

print(s)
print 'Runtime:', (datetime.now() - START_TIME)

