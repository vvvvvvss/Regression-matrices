from sklearn.metrics import mean_absolute_error
y_true=[0,2,3,5,9]
y_pred=[0.1, 1.7, 3.9, 5, 7]
print('Mean Absolut error=', mean_absolute_error(y_true,y_pred))
