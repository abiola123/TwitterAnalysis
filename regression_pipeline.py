import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

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

parser.add_argument(
    "--name",
    type=str,
    help="name of the dataset used to save the coefficients plot",
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

# Assign significant levels stars to each variable name
def assign_stars(row):
    p_value = float(row['P>|t|'])
    name = row[0].replace('_', ' ').title()
    if p_value <= 0.01:
        return name + ' ***'
    elif p_value <= 0.05:
        return name + ' **'
    elif p_value <= 0.1:
        return name + ' *'
    else:
        return name
    
# Define function to output plot of the model coefficients
def coefplot(results, name):
    ### PREPARE DATA FOR PLOTTING
    # Create dataframe of results summary 
    coef_df = pd.DataFrame(results.summary().tables[1].data)
    
    # Add column names and drop the extra row with column labels
    coef_df.columns = coef_df.iloc[0]
    columns = coef_df.iloc[:,0]
    coef_df=coef_df.drop(0)

    # Rename column 0 and append * ** or *** for significance levels
    # coef_df["index"] = coef_df.apply(lambda x: assign_stars(x), axis=1)

    # Set index to variable names
    coef_df = coef_df.set_index(coef_df.apply(lambda x: assign_stars(x), axis=1))
    coef_df = coef_df.drop(coef_df.columns[0], axis=1)

    # Change datatype from object to float
    coef_df = coef_df.astype(float)

    # Get errors; (coef - lower bound of conf interval)
    errors = coef_df['coef'] - coef_df['[0.025']
    coef_df['errors'] = errors

    # Sort values by coef ascending
    coef_df = coef_df.sort_values(by=['coef'])

    ### PLOT COEFFICIENTS
    variables = list(coef_df.index.values)
    coef_df['variables'] = variables

    # Define figure, axes, and plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Error bars for 95% confidence interval
    # Can increase capsize to add whiskers
    coef_df.plot(x='variables', y='coef', kind='bar', ax=ax, color='none', fontsize=15, ecolor='steelblue', capsize=0, yerr='errors', legend=False)
    
    # Coefficients
    ax.scatter(x=np.arange(coef_df.shape[0]), marker='o', s=80, y=coef_df['coef'], color='steelblue')
    
    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
    
    # Set title & labels
    #plt.title('Coefficients of Features - 95% Confidence Intervals',fontsize=20)
    ax.set_ylabel('Coefficients',fontsize=15)
    ax.set_xlabel('',fontsize=15)

    # Rotate y ticks and move to the right side
    ax.yaxis.tick_right()
    plt.yticks(rotation=90, fontsize=15)
    
    plt.savefig(name+'_coefficient_plot.png', bbox_inches='tight', dpi=300)
    plt.show()

    return list(columns[2:])

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
    _, _, _, res = regression(args.path, args.threshold)
    coefplot(res, args.name)