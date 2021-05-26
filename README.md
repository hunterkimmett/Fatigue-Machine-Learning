# Final Project - Machine Learning Applied to Steel Fatigue Data
Author: Hunter Kimmett

## Introduction

This project will involve visualizing and applying ML models to Fatigue data from various steel alloys. Fatigue life expectancy of alloys is usually estimated using Linear Regression, and for this project we will attempt to use a different model to achieve better results.

We will be attempting to match results from the paper "Exploration of data science techniques to predict fatigue strength of steel from composition and processing parameters" as seen here:

https://link.springer.com/article/10.1186/2193-9772-3-8


## Contents

- fatigue_dataset.xlsx: steel fatigue data
- ProjectNotebook.ipynb: code for analysis
- reference_images: folder of reference images for report


## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- sklearn


## How to Run

Ensure Notebook and dataset are in the same folder and that all images are in reference_images folder, then run Notebook as any Jupyter Notebook.

## 0. Function definitions

There are 2 key functions:

- get_cross_val_scores: performes cross validation and returns training and test scores
- get_scores: performs model prediction and prints scores, returns predicitions

## 1. Load data

Data is loaded into pandas dataframe, feature and target matrices. 25 features, 437 data points.


## 2. Inspect the data 

### 2.1 Variations in comparable feature types

In the paper there are 3 categories of features within this dataset:
- Chemical composition - %C, %Si, %Mn, %P, %S, %Ni, %Cr, %Cu, %Mo (all in wt. %)
- Heat treatment conditions - temperature, time and other process conditions for normalizing, through-hardening, carburizing-quenching and tempering processes
- Upstream processing details - ingot size, reduction ratio, non-metallic inclusions

These features are shown defined below, as their abbreviations are the column headers for our feature matrix. We can also look at a boxplot comparing variances in these comparable features:

![Feature table](reference_images/feature_table.png)


#### Feature Boxplots

![Boxplot](reference_images/bp_chem.png)

From this plot we can see that the elements with the most variance are Ni and Cr. Ni, Mn Si and C have the most outlier points, while P, S, Cu and Mo are the most consistent. It is good to see consistent low data for Phosphorus and Sulfur, as these elements can significantly weaken steel.


![Boxplot](reference_images/bp_ht_time.png)

![Boxplot](reference_images/bp_ht_temp.png)

![Boxplot](reference_images/bp_ht_rate.png)

For heat treatments we can see that in some cases no heat treatment wsa performed at all, showing with the median CT, Ct, DT, Dt, and QmT values of zero with some outliers. Values in other categories seem to coalesce around the same vlues or range of values, showing moderate consisitency with the heat treatments used on these samples.


![Boxplot](reference_images/bp_proc_a.png)

![Boxplot](reference_images/bp_proc_r.png)


### 2.2 Correlation heatmap of features 

To understand if pairs of features are potentially related, contain similar information, pair-wise cross-correlation can be calculated.

Let's take a look at the correlation heatmap of our features. Since one heatmap cannot contain all this info, let's just look at the correlation between the weight% of all the alloying materials:


![Heatmap](reference_images/heatmap.png)

Looks like fairly low correlation across the board, except Cr with Mo. There are also a few more minor correlations across the board but it remains quite low in most cases.

### 2.3 Histogram of target vector

Let's look at the distribution of the fatigue life of all our samples.

![Histogram](reference_images/histogram.png)

Looks like we have a high distribution of samples with a fatigue life of 500 * 10^7 cycles.


## 3. Create training and test sets

Using scikit-learn `train_test_split()` with parameters `random_state=37`, `test_size=0.2`, split `X` and `y` into training and test sets.


## 4. Compare models using cross-validation

### 4.1 Model Selection

We will be using the following 3 models, similar to ones from the paper. Unfortunately many of the models from the paper are not available to use in scikit-learn, so we will be making due with models that closely match the better performing models from the paper. These are the models:

#### Linear Regression

Linear Regression was chosen because it is the standard fatigue life prediction model in use. This model performed well in the paper, boasting high R and R^2 scores. This also makes for a good base case regressor as no hyperparameter tuning is required.

#### Decision Tree Regressor

Decision Tree Regressor was chosen because it will perform most similarly to the Reduced Error Pruning Tree (REPTree) model from the paper. This model had among the highest R and R^2 scores in the paper and performed better than the Linear Regression model they used. It will be interesting to see if the scikit-learn model will be analogous to the model from the paper.


#### MLP Regressor

MLP Regressor was chosen because it will perform most similarly to the Artificial Neural Network (ANN) model from the paper. This model will likely be the most difficult to use, as I do not have much experience with Neural Networks. In the paper it performed better than Linear Regression and worse than REPTree, however this model from sickit-learn is definitely not the same as the one used in the paper. It will be intriguing to see how it performs in comparison, however expecting it to perform as well as Linear Regression may be asking too much.


### 4.2 Creating List of Models

Create a list containing a `LinearRegression()`, `DecisionTreeRegressor()` and `MLPRegressor()` objects.

Iterate this list, compute the negative root mean-squared error using the `get_cross_val_scores()` function, and print the training and validation scores with **2 decimal places**. Use 7-fold cross-validation.


![Results](reference_images/model_list_results.png)

Linear regression seems to remain the best option using default parameters and is in line with results from the paper, but let's see if we can use a grid search with the DecisionTreeRegressor to find optimal hyperparameters and perhaps outperform LinearRegression.


## 5. Hyperparameter tuning using grid search 

### 5.1 Grid search for DecisionTreeRegressor

Perform grid search using `GridSearchCV` for the `DecisionTreeRegressor()`.

Grid search to use 7-fold cross-validation, and `r2` as the scoring function.

Using the following hyperparameters and values:
    
- `"criterion": ["mse", "mae"]`
- `"min_samples_split": [5, 10, 20, 40]`
- `"max_depth": [2, 6, 8, 10]`
- `"min_samples_leaf": [1, 5, 10, 20]`
- `"max_leaf_nodes": [5, 20, 100, 200]`


Best parameters: {'criterion': 'mae', 'max_depth': 8, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 20}

Best cross-validation score: 0.97

## 6. Retrain best model

Retraining the best estimator from the grid search and a new Linear Regression Model to compare.


## 7. Evaluate best model on training and test data
### 7.1 Calculating R-squared, Mean absolute error, and Root mean-squared error

For the retrained best estimator and linear regression, prints the scores for training and test sets.

![Results](reference_images/grid_results.png)

### 7.2 Predicted vs actual fatigue life plots

![Scatterplot](reference_images/result_plot_dr.png)

![Scatterplot](reference_images/result_plot_lr.png)


### 7.3 Residual plots

![Residual](reference_images/residual_plot_dr.png)

![Residual](reference_images/residual_plot_lr.png)


## 8. Analysis

Upon looking at the data from section 7.1, it looks like the Linear Regression model performs the best over the other models tried in this project. This is further backed up looking at 7.2 and 7.3, where the predicted data for Linear Regression is clustered more tightly around its actual value compared to the Decision Tree Regressor. 


The Decision Tree Regressor came very close to achieving the same test scores as the Linear Regression model, however it still could not perform as well. The difference between this model's performance and the one in the paper could be due to different tuning, or the model from the paper may have been different than scikit-learn's model.


It is interesting to note that when actual fatigue life exceeded around 800 cycles we can see a decrease in performance for the Linear Regression model. This could be due to the model being less flexible to outlying points, but we would need more data to come to any conclusions of its performance with these values. 


It is also interesting that the Linear Regression model from scikit-learn outperformed the paper's model, however this is likely due to the data selection for the test being easier for the model to get right.


## 9. Conclusion

While it is unfortunate that my models could not quite match the paper, there are some good conclusions that can be drawn:
- Decision Tree Regressor models can perform nearly as well as Linear Regression
- Linear Regression is a very effective way to measure fatigue life and is good as an industry standard
- I will need to look into models outside of scikit-learn to further my ML learning experience


## 10. Reflection

My final project and my initial proposal had several deviations:

#### Models Selected

Initially I wanted to try and use 5 different models that outperformed linear regression in the paper, however upon reviewing the limitations of scikit-learn I had to reduce the scope of my project, and even use different models altogether. Linear Regression stayed the same as from the paper, while Decision Tree Regressor and MLP Regressor served as analogues for Reduced Error Pruning Tree and Artificial Neural Network models respectively. I could not find a good enough analogue for the M5 Model Tree, and Multivariate Polynomial Regression was slightly too complex to fit well with my cross-validation method.

#### Scaling & Dimensionality Reduction

Unfortunately due to time constraints I was unable to implement Scaling. This was a method not used by the paper so it would have been very interesting to see the data produced. Dimensionality reduction was also a possibility but upon further research I determined that it would not be necessary for this project.


#### Further Reflection

Overall, I think for the most part the project went according to the proposal, but I am disappointed I couldn't outperform Linear Regression. This is a very tough task with R^2 scores so high as they are. If I were to repeat this project I might have used a different dataset (maybe a different material) as it would be more realistic to compare this to industry standards. A small sample size of 30-50 data points may change the effectiveness of the models used for this project.

