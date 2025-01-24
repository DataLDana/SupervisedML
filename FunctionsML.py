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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Due to time constraints this plot was generated with chatgpt 
def param_plot(df):
    
    # Define helper functions
    def is_numeric_series(series):
        return series.apply(lambda x: isinstance(x, (int, float)) or pd.isna(x)).all()
    
    def is_categorical_series(series):
        return series.dtype == 'object' or series.apply(lambda x: isinstance(x, str)).any()
    
    # Ensure you have the correct filter for your column names
    param_columns = [col for col in df.columns if 'param_' in col]
    if not param_columns:
        raise ValueError("No columns with 'param_' found in the DataFrame.")
    
    # Assign unique colors and markers to each sample
    unique_samples = df.index  # Assuming each row is a sample
    color_palette = sns.color_palette("husl", len(unique_samples))
    markers = ['o', 's', 'D', '^']  # Extend if needed, 'v', '<', '>', 'P', '*', 'X', 'h', '+', 'x'
    sample_colors = {sample: color_palette[i % len(color_palette)] for i, sample in enumerate(unique_samples)}
    sample_markers = {sample: markers[i % len(markers)] for i, sample in enumerate(unique_samples)}
    
    # Prepare subplot grid
    n_cols = 2
    n_rows = math.ceil(len(param_columns) / n_cols)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9, 4 * n_rows))
    axes = axes.flatten()
    
    for ax, param in zip(axes, param_columns):
        if is_numeric_series(df[param]):
            for sample in unique_samples:
                ax.scatter(df.loc[sample, param], df.loc[sample, 'mean_test_score'], 
                           color=sample_colors[sample], 
                           marker=sample_markers[sample],
                           label=f'Sample {sample}', alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Test Score')
        elif is_categorical_series(df[param]):
            for sample in unique_samples:
                sns.stripplot(x=[df.loc[sample, param]], 
                              y=[df.loc[sample, 'mean_test_score']], 
                              ax=ax, 
                              jitter=False, 
                              color=sample_colors[sample],
                              marker=sample_markers[sample],
                              alpha=0.7, 
                              size=10)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Test Score')
        else:
            ax.set_visible(False)
    
    # Add legend to the first subplot only (for clarity)
    handles = [plt.Line2D([0], [0], color=color, marker=marker, linestyle='', label=f'Sample {sample}') 
               for sample, color, marker in zip(unique_samples, color_palette, markers)]
    axes[0].legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Turn off unused subplots
    for ax in axes[len(param_columns):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()
