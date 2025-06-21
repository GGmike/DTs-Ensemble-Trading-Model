from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from config import config

def uniqueDT(columns, max_combination, num_nodes, X_train, y_train):
    print(f"Creating unique Decision Trees with {num_nodes} nodes and max {max_combination} combinations.")
    print("Type of max_combination:", type(max_combination))
    print("Type of num_nodes:", type(num_nodes))
    print("Type of columns:", type(columns))
    print(f"Columns used: {len(columns)}")
    # print("Columns:", columns)
    base_learners = []
    used_combinations = []
    all_combinations = list(combinations(columns, int(num_nodes)))
    print(f"Total combinations: {len(all_combinations)}")
    selected_combinations = all_combinations[:min(int(max_combination), len(all_combinations))]
    for training_columns in selected_combinations:
        training_columns = list(training_columns)        
        X_sample = X_train[training_columns]
        dt = DecisionTreeClassifier(max_depth=config['max_depth'])
        dt.fit(X_sample, y_train)
        
        base_learners.append(dt)
        used_combinations.append(training_columns)
    return base_learners, used_combinations