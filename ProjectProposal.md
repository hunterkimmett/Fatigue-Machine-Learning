# Project proposal for Using Machine Learning to Predict Steel Fatigue Life
Author: Hunter Kimmett

## 1. Why: Question/Topic being investigated 2pts

I would like to further expand on the ML methods learned in class and apply them to data that I am familiar with. The data being explored is Fatigue Test data from multiple steel alloys. This data presents a unique challenge because it can have many features, such as heat treatment (which can have multiple dimensions of time and temperature), composition (% of several elements) and size of the tested material. In the dataset I have found for steel tests there are 26 features.

At my previous job in the aerospace industry, fatigue tests of materials were critical to Research and Development. These tests determined if the materials would be used in aircraft parts, as fatigue is a big threat to the reliability of the parts. Fatigue testing is a very expensive process and a single test can cost upwards of $3000, and to fully characterize a material 50+ tests can be required. Linear regression is used to predict the fatigue life of materials with test data, and I would like to explore the possibility of other predictive methods being used to make a characterization and make the process more efficient.

## 2. How: Plan of attack 2pts

My main reference for this project is the paper "Exploration of data science techniques to predict fatigue strength of steel from composition and processing parameters" as seen here:

https://link.springer.com/article/10.1186/2193-9772-3-8

I will be using previous labs as a template to implement ML methods explored in the paper. First I will visualize some of the features, then I will modify them using scaling and possibly dimensionality reduction. I will then use the usual lab steps of seperating datasets into training and testing. I believe I will train 3 to 5 models on the data and use cross-validation to verify their accuracy. In the paper some models use default hyper parameters, others use some described in the paper. If I am unsatisfied with the initial scores of my models I will use a grid search to tune them. Once the models are tuned they will be used to predict the test set of data and then the results will be compared and visualized.

The goal will be to have the models perform comparably to those in the paper.

## 3. What: Dataset, models, framework, components 2pts

- Dataset from the paper: https://link.springer.com/article/10.1186/2193-9772-3-8
- Lab/Assignment methods
- Models from paper if available in Scikit-learn:
- LinearRegression
- PaceRegression
- ANN
- M5ModelTree
- MPR

Models are not final and may be changed. If model is not available in Scikit-Learn alternate ways to apply model will be found.
