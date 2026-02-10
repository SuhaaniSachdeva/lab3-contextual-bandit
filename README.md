This repository contains my implementation for Lab 3 of the Reinforcement Learning course. The goal of the assignment is to build a contextual bandit based news recommendation system by first classifying users into contexts and then learning optimal recommendations for each context.

The work is implemented entirely in a single Jupyter notebook:
lab3_results_U20230075.ipynb

Data files used:

train_users.csv

test_users.csv

news_articles.csv

The assignment is divided into two main parts.

User Classification (Section 5.2):
User features are preprocessed by handling missing values and encoding categorical variables using one-hot encoding. A Decision Tree classifier is trained to classify users into three user contexts. The model is evaluated using a validation split and cross-validation. Validation accuracy and classification reports are printed in the notebook.

Contextual Bandits (Section 5.3):
Each user context is treated as a separate bandit problem with four arms corresponding to news categories. Rewards are generated using the provided rlcmab_sampler. The following algorithms are implemented and compared:

Epsilon-Greedy

Upper Confidence Bound (UCB)

Softmax

Each algorithm is run for a fixed number of time steps and evaluated using average reward over time. Hyperparameter comparisons and cumulative reward plots are included.

Recommendation Engine (Section 5.4):
A recommendation function is implemented that first predicts a user’s context using the trained classifier and then selects the best news category using the learned bandit policy. Both single-user and batch recommendations are demonstrated. The distribution of recommended categories and predicted user contexts is also analyzed and visualized.

Results:
Among the evaluated algorithms, UCB achieves the highest final average reward, followed closely by Softmax. Epsilon-Greedy performs well for small epsilon values but converges more slowly.

Requirements:
Python 3.x
NumPy
Pandas
Matplotlib
Scikit-learn
rlcmab_sampler
