import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    description="Take the path to a regression csv file, perform multiple regressions and save the results for each regression in seperate html files"
)

parser.add_argument(
    "--path", type=str, help="path to the input CSV file containing the pre-processed data"
)


parser.add_argument(
    "--threshold",
    type=int,
    default=1e2,
    help="minimum impresion count (used to filter out outliers)",
)



def create_regressor_columns_string(columns):
  # Join all independent variables to define the formula used by the model
  regressor_columns = list(filter(lambda x: x != 'impression_count',columns))
  regressor_columns_string = "+".join(regressor_columns)
  return regressor_columns_string

def perform_regression(regression_df, significant_variables):
    # Define formula without cross terms
    regressor_columns_string = create_regressor_columns_string(significant_variables)
    # Fit model
    mod = smf.ols(formula=f'impression_count ~ {regressor_columns_string}', data=regression_df)
    res = mod.fit()
    return res

def keep_significant_var(p_values):
    # Filter p-values < 0.05
    p_values = p_values[p_values < 0.05]

    # Get significant variables
    significant_variables = list(p_values.index)
    # Remove intercept if significant
    try:
        significant_variables.remove('Intercept')
    except:
        pass

    return significant_variables

def regression(path, threshold=1e2):

    # Load data
    regression_df_pd = pd.read_csv(path)
    regression_df_pd = regression_df_pd.drop('tweet_text', axis=1)
    
    #======================#
    #   FIRST REGRESSION   #
    #======================#
    significant_variables = list(regression_df_pd.columns)
    res = perform_regression(regression_df_pd, significant_variables)
    p_values = res.pvalues
    nb_variables_before = len(p_values)
    
    #======================#
    #   SECOND REGRESSION  #
    #======================#
    # Filter out data if impression_count < threshold
    regression_df_pd = regression_df_pd[regression_df_pd['impression_count'] >= threshold].copy()

    # Perform new regression with significant variables
    res_threshold = perform_regression(regression_df_pd, significant_variables)

    #======================#
    #   THIRD REGRESSION   #
    #======================#
    # Apply log transformation to independent variables
    for var in regression_df_pd.columns:
            regression_df_pd[var] = regression_df_pd[var].apply(lambda x: np.log(1+x))

    # Perform new regression with significant variables
    res_log = perform_regression(regression_df_pd, significant_variables)
    p_values = res_log.pvalues

    #======================#
    #   FINAL RESULTS      #
    #======================#
    # Filter p-values < 0.05
    significant_variables = keep_significant_var(p_values)
    nb_variables_after = len(significant_variables)

    # Perform new regression with significant variables
    res_final = perform_regression(regression_df_pd, significant_variables)

    print(f'Number of discarded variables: {nb_variables_before - nb_variables_after}')
    print(f'Significant variables ({len(significant_variables)}): {significant_variables}')

    #======================#
    #   SAVE AS HTML       #
    #======================#
    # Save first res as html
    res_html = res.summary().as_html()
    name = path.split('/')[-1].split('.')[0][:-3]
    path_root = os.path.join(os.getcwd(), "data", "regression", "html_regression")
    path = os.path.join(path_root, name+".html")
    with open(path, 'w') as f:
        f.write(res_html)

    # Save second res as html
    res_thresh_html = res_threshold.summary().as_html()
    path_opti = os.path.join(path_root, name+"_threshold.html")
    with open(path_opti, 'w') as f:
        f.write(res_thresh_html)

    # Save third res as html
    res_log_html = res_log.summary().as_html()
    path_log = os.path.join(path_root, name+"_log.html")
    with open(path_log, 'w') as f:
        f.write(res_log_html)

    # Save final res as html
    res_final_html = res_final.summary().as_html()
    path_final = os.path.join(path_root, name+"_final.html")
    with open(path_final, 'w') as f:
        f.write(res_final_html)

    return res, res_threshold, res_log, res_final


if __name__ == "__main__":
    args = parser.parse_args()
    regression(args.path, args.threshold)