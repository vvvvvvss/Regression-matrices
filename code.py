from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
y_true=[0,2,3,5,9]
y_pred=[0.1, 1.7, 3.9, 5, 7]
print('Mean Absolute error=', mean_absolute_error(y_true,y_pred))
print('Mean Squared error=', mean_squared_error(y_true,y_pred))
print('Root Mean Squared error=', np.sqrt( mean_squared_error(y_true,y_pred)))
