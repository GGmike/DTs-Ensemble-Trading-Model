from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from config import config
from main import X_train, y_train

def uniqueDT(columns, max_combination, num_nodes):
    base_learners = []
    used_combinations = []
    all_combinations = list(combinations(columns, num_nodes))
    selected_combinations = all_combinations[:min(max_combination, len(all_combinations))]
    for training_columns in selected_combinations:
        training_columns = list(training_columns)        
        X_sample = X_train[training_columns]
        dt = DecisionTreeClassifier(max_depth=config['max_depth'])
        dt.fit(X_sample, y_train)
        
        base_learners.append(dt)
        used_combinations.append(training_columns)
    return base_learners, used_combinations