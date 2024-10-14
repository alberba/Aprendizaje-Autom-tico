## RBF

Para el kernel rbf, hemos probado con los siguientes hiperparámetros:

```python

'rbf': {'kernel': ['rbf'], 'C': [0.01, 0.05, 0.1, 1], 'gamma': ['scale', 'auto'], 'tol': [1e-2, 1e-3, 1e-4], 'max_iter': [1000, -1]}

```

![alt text](image-1.png)

Parece ser que en el rbf, no superamos una precisión mayor al 0.659. De momento, la mejor combinación de hiperparámetros es: ***{'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 1000, 'tol': 0.01}***.

Tampoco importa igualmente el tipo de hog que apliquemos, ya que los resultados son los mismos.



## Linear

```python
'linear': {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10], 'tol': [1e-2, 1e-3, 1e-4], 'max_iter': [1000, -1]},
```

![hog 8 4 18](image-2.png)

```python
'linear': {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 100], 'tol': [1e-2, 1e-3, 1e-4], 'max_iter': [10000, -1]},
```

![hog 8 4 9](image-3.png)

```python	
'linear': {'kernel': ['linear'], 'C': [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 100], 'tol': [1, 0.1, 1e-2, 1e-3, 1e-4], 'max_iter': [5000, 10000, 20000, -1]},
```

![alt text](image-4.png)