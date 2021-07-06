# Gradient Descent Practice

In this notebook, we practice using the 


```python
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

data = load_diabetes()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

# Jupyter configuration
%config Completer.use_jedi = False
```

## 1. Set up a train test split


```python
# Your code here
```


```python
#__SOLUTION__
X = df.drop('target', axis = 1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
```

## 2. Initialize an SGDRegressor model


```python
# Your code here
```


```python
#__SOLUTION__
model = SGDRegressor(random_state=2021)
```

## 3. Fit the model


```python
# Your code here
```


```python
#__SOLUTION__
model.fit(X_train, y_train)
```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_stochastic_gradient.py:1228: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      ConvergenceWarning)





    SGDRegressor(random_state=2021)



## 4. Multiple choice question asking them to select what the default setting is for max_iter


```python
# Run this cell unchanged
from src.questions import *
question_4.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 5. Update the max_iter variable to 10000


```python
# Your code here
```


```python
#__SOLUTION__
model = SGDRegressor(max_iter=10000, random_state=2021)
model.fit(X_train, y_train)
```




    SGDRegressor(max_iter=10000, random_state=2021)



# Quick aside about finding the max iter programmatically


```python
# Run this cell unchanged
import warnings

# Loop over a range of numbers between 1000 10,000
for i in range(1000, 10000, 500):
    # Catch the ConvergenceWarning
    with warnings.catch_warnings():
        # If a warning is produced, throw an error instead
        warnings.filterwarnings('error')
        # Place the model fit inside a try except block to catch the error
        try:
            model = SGDRegressor(max_iter=i, random_state=2021)
            model.fit(X_train, y_train)
            # If the model fits without a ConvergenceWarning stop the for loop
            break
        except Warning:
            # If the model returns a ConvergenceWarning, move to the next iteration.
            continue
            
# Print the number of iterations that allowed the model to converge
print('Max iterations needed for convergence:', i)
```

    Max iterations needed for convergence: 6500



```python
#__SOLUTION__
# Run this cell unchanged
import warnings

# Loop over a range of numbers between 1000 10,000
for i in range(1000, 10000, 500):
    # Catch the ConvergenceWarning
    with warnings.catch_warnings():
        # If a warning is produced, throw an error instead
        warnings.filterwarnings('error')
        # Place the model fit inside a try except block to catch the error
        try:
            model = SGDRegressor(max_iter=i, random_state=2021)
            model.fit(X_train, y_train)
            # If the model fits without a ConvergenceWarning stop the for loop
            break
        except Warning:
            # If the model returns a ConvergenceWarning, move to the next iteration.
            continue
            
# Print the number of iterations that allowed the model to converge
print('Max iterations needed for convergence:', i)
```

## 6. What is the default setting for alpha? - Multi choice


```python
# Run this cell unchanged
question_6.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'dat…



```python
#__SOLUTION__
# Run this cell unchanged
question_6.display()
```

## 7. Update the alpha to .01 and set the max_iter to 1500


```python
# Your code here
```


```python
#__SOLUTION__
model = SGDRegressor(max_iter=1500, alpha=0.01, random_state=2021)
model.fit(X, y)
```




    SGDRegressor(alpha=0.01, max_iter=1500, random_state=2021)



## 8. The model converged - True or False


```python
# Run this cell unchanged
question_8.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'dat…



```python
#__SOLUTION__
# Run this cell unchanged
question_8.display()
```

## 9. Select the answer that best describes how alpha impacts a models fit


```python
# Run this cell unchanged
question_9.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'dat…



```python
#__SOLUTION__
# Run this cell unchanged
question_9.display()
```

# Linear Algebra Practice

## 10. When finding the dot product for two vectors, the length of the vectors must be the same.


```python
# Run this cell unchanged
question_10.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'dat…



```python
#__SOLUTION__
# Run this cell unchanged
question_10.display()
```

## 11. Please select the solution for the dot product of the following vectors.

$vector_1 = \begin{bmatrix} 10&13\\ \end{bmatrix}$

$vector_2= \begin{bmatrix} -4&82\\ \end{bmatrix}$



```python
# Run this cell unchanged
question_11.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'dat…



```python
#__SOLUTION__
# Run this cell unchanged
question_11.display()
```

## 12. How do you turn a list into a numpy array?


```python
# Run this cell unchanged
```

## 13. Which answer correctly represents the  dot product of the two matrices?



$\begin{bmatrix} 2&3\\ 1&3\\ \end{bmatrix} \cdot \begin{bmatrix} 2&3\\ 3&4\\ \end{bmatrix} = \begin{bmatrix} ?+?&?+?\\ ?+?&?+?\\ \end{bmatrix}$


```python
# Run this cell unchanged
```
