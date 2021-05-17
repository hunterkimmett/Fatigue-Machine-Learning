# Machine learning project

Total 24pts.

Tasks:
1. Write a project proposal in `ProjectProposal.md`
2. Complete the project. In `ProjectReadme.md` include:
  - Background on your code files
  - How to run your code, guide to install any additional packages
  - Results, interpretation and reflection

## 1. Proposal (6pts)
1. Why: Question/Topic being investigated 2pts
2. How: Plan of attack 2pts
3. What: Dataset, models, framework, components 2pts

### Example 1
1. Why: Question being investigated 2pt  

I would like to understand better how the decision tree algorithm works.

2. How: Plan of attack 2pt

In *insert reference here* a guide to implement decision trees from scratch is available. I will download the code, run with the tutorial dataset (very small) and then apply it to a real dataset (see below) and compare to scikit-learn. Furthermore, I will try and implement early stopping by introducting `max_depth` parameter. This is not part of the tutorial code and an extension.

3. What: Dataset, models, framework, components 2pts

- UCI dataset *insert url here* (13 features, 300 samples)
- Code from *insert code github url*
- Scikit-learn DecisionTreeClassifier

### Example 2
1. Why: Question being investigated 2pt

I would like to see what it takes to participate in a Kaggle machine learning competition.

2. How: Plan of attack 2pt

I have created an account on Kaggle and they recommend starting with the Titanic classification problem. I will follow these guidlines to prepare a notebook and submit it. I will use the Lab3 notebook as a template.

If time premits, a second competition will be selected and prepared.

3. What: Dataset, models, framework, components 2pts

- Kaggle competition url *insert url*
- Scikit-learn classifiers: LogisticRegression, RandomForest, GradientBoosting, SVC
- Possibility to explore XGBoost library as an alternate classifier.

### Example 3
1. Why: Question being investigated 2pt

I am interested in learning more about putting a model into production.

2. How: Plan of attack 2pt

Initial research showed, that mlflow *insert url* is a framework that allows for training and deploying machine learning models. On their website *insert url here* there are numerous tutorials. I am planning to follow the following:
- Tutorial 1 *add description*
- Tutorial 2 *add description*

Subsequently, I will adapt the tutorial code to create a website that allows entering Iris flower sepal and petal measurements, and the classifier displays the predicted type of Iris flower. To demonstrate this pipeline, only one classifier will be trained.

3. What: Dataset, models, framework, components 2pts

- mlflow *url here* with submodules *module 1* *module 2*
- mlflow to serve the model as a RESTapi
- Scikit-learn Iris dataset
- Scikit-learn classifiers: LogisticRegression
- Flask framework to setup up webserver.

The Flask server will provide the front-end for the website allowing the user to enter Iris measurements. Upon submission of the measurements, Flask will call the mlflow RESTapi to obtain the prediction results. Flask then displayes the results as: predicted class, probability and a sample image of the Iris class.



## 2. Final report (18pts)
1. Code 6pts
  - Runs 2pts
  - Complete 2pts
  - Organized 2pts
2. Results 6pts
  - Data summarized/visualized 3pts
  - Model selection organized and explained 3pts
3. Interpretation 3pts
  - As described in the proposal, was the question answered/topic investigated and how?
4. Reflection 3pts
  - Where there deviations from proposal? Explain why (or why not).


## 3. Project ideas

### Gaining more confidence
Use lab2 or lab3 as a template and select a different dataset. Run through the steps that make sense for the data, add new steps if necessary, and show your solution.

### Becoming competitive
Select a kaggle.com competition and try to put together a solution for the problem.

### Implementing from scratch
Browse Data Sceince from Scratch for an algorithm of interest. Or use any other blog that guides through an implementation from scratch for your favourite algorithm. Demonstrate your working code on a different dataset than the guid/book used.

### Machine learning history
Write about the history of machine learning and algorithms. Here it would be important to provide a new perspective. For example, can the sequence of algorithms reported be correlated with popularity of these algorithms?

### Investigating a machine learning library or framework
If you find an interesting library or framework, you would follow the intoroduction tutorial and try to adapt it to new data.

### Machine learning theory
Write about the mathematics used in one of your favourite algorithms. Here it would be important to connect any equation to code. The goal would be to make theory more accessible to others.

### Machine learning productizing
Describe a problem that would need a frontend or app to bring a machine learnign model to users. Design the system, select frameworks, maybe build a small prototype.