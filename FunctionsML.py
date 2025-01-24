import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, root_mean_squared_log_error
import math
import matplotlib.pyplot as plt
import seaborn as sns

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_scores(search,X,y):
    """
    Description:
    Get 3 different scores 
    Parameters:
        search: GridSearchCV Transformer,
        X: features of train/test
        y: target feature train/test
    Returns:
        dictionary with scores
    """
    
    y_pred = search.predict(X)
    score = {'neg_root_mean_squared_log_error':round(root_mean_squared_log_error(y, y_pred),3)*-1,
                 'R2':round(r2_score(y, y_pred),3),
                 'RSME':round(root_mean_squared_error(y, y_pred),3)
                }
    return score

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def final_scores(scores,al_score,name):
    """
    Description:
    Add new score to the df scores
    Parameters:
        scores: df with stored scores
        al_score: dictionary, with scores to add
        name: str, what name the index of the df gets
    Returns:
        scores, df with scores
    """
    # write df from dictionary with name
    df = pd.DataFrame(al_score,index=[name])
    # avoid double entries by checking if name is already an index
    if not name in scores.index: 
        # concatenate df to scores only if not empty
        scores = pd.concat([scores, df], axis=0) if not scores.empty else df
    return scores


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
def show_best(search,many):
    """
    Description:
    function to show best results in a dataframe from 
    the search.cv_results_ attribute
    Parameters:
        search: GridSearchCV Transformer
        many: int, how many lines should be shown
        name: str, what name the index of the df gets
    Returns:
        dictionary with parameters, rank and mean test scores
    """
    # Extract results of cv_results_
    results_df = pd.DataFrame(search.cv_results_)
    # Filter to relevant columns
    relevant_cols_mask = (
                        (results_df.columns.str.contains('param_')) | 
                        (results_df.columns.isin(['rank_test_score', 'mean_test_score']))
    )
    # Sort and filter
    relevant_results = results_df.sort_values(by='rank_test_score').loc[:,relevant_cols_mask].copy()
    # Create shorter names for easier to view data frame
    short_names = {col_name:" ".join(col_name.split("__")[-2:]
                                    ) for col_name in relevant_results.columns}
    relevant_results.rename(short_names, axis=1, inplace=True)
    
    #relevant_results.head(15)
    return relevant_results.loc[relevant_results['rank_test_score']<many]

