# Gradient Descent and Linear Algebra Review

In this notebook, we review:
* `learning rate` - a standard hyperparameter for gradient descent algorithms, and evaluate how learning rate impacts a model's fit.
* Introductory elements of linear algebra.
    * Vector compatibility 
    * 


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

    Max iterations needed for convergence: 6500


## 6. What is the default setting for alpha? - Multi choice


```python
# Run this cell unchanged
question_6.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…



```python
#__SOLUTION__
# Run this cell unchanged
question_6.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


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


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…



```python
#__SOLUTION__
# Run this cell unchanged
question_8.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 9. Select the answer that best describes how alpha impacts a model's fit


```python
# Run this cell unchanged
question_9.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…



```python
#__SOLUTION__
# Run this cell unchanged
question_9.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


# Linear Algebra Practice

## 10. When finding the dot product for two vectors, the length of the vectors must be the same.


```python
# Run this cell unchanged
question_10.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…



```python
#__SOLUTION__
# Run this cell unchanged
question_10.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 11. Please select the solution for the dot product of the following vectors.

$vector_1 = \begin{bmatrix} 10&13\\ \end{bmatrix}$

$vector_2= \begin{bmatrix} -4&82\\ \end{bmatrix}$



```python
# Run this cell unchanged
question_11.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…



```python
#__SOLUTION__
# Run this cell unchanged
question_11.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 12. How do you turn a list into a numpy array?


```python
# Run this cell unchanged

question_12.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 13. Please find the dot product of the following vectors


```python
vector_1 = [
               [ 0.80559827,  0.29916789,  0.39630405,  0.92797795, -0.13808099],
               [ 1.7249222 ,  1.59418491,  1.95963002,  0.64988373, -0.08225951],
               [-0.50472891,  0.74287965,  1.8927091 ,  0.33783705,  0.94361808],
               [ 0.99034854, -1.0526394 , -0.33825968, -0.40148036,  1.81821604],
               [-0.7298026 , -0.88302624,  0.49319177, -0.02758864,  0.33430167],
               [ 0.85938167, -0.71149948, -1.8434118 ,  0.89097775,  0.53842254]
                                                                                    ]


vector_2 = [
              [ 0.13288805],
              [-2.50839814],
              [-0.90620828],
              [ 0.09841538],
              [ 1.86783262],
              [ 1.98903307]
                               ]
```


```python
# Your code here
```


```python
#__SOLUTION__
vector_1 = np.array(vector_1)
vector_2 = np.array(vector_2)

np.dot(vector_1.T, vector_2)
```




    array([[-3.31869275],
           [-7.80043543],
           [-9.35675419],
           [-0.13185929],
           [ 1.207176  ]])


