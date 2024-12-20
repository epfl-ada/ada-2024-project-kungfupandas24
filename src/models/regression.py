import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(X):
    print("------------------------------------------------------------------------------------------------")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    if all(vif_data["VIF"] < 5):
        print("All VIF values are below 5. Multicollinearity is not a concern.")
    elif any(vif_data["VIF"] >= 10):
        print("Some VIF values are above 10. There is significant multicollinearity.")
    else:
        print("Some VIF values are between 5 and 10. Multicollinearity may be moderate.")
        
    print("------------------------------------------------------------------------------------------------")

    return vif_data

def prepare_data(df, indep_vars, dep_var):
    X = df[indep_vars]
    y = df[dep_var]
    
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, columns=indep_vars, index=X_train.index)
    X_scaled_wconst = sm.add_constant(X_scaled)
    y = y_train
    
    return X_scaled_wconst, y, X_test, y_test, scaler

def run_ols_regression(X, y):
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model

def run_regression_for_genre(X_sclaed_wconst_train, y_train):
    
    model = run_ols_regression(X_scaled_wconst_train, y_train)
    
    return model

def compute_rmse(X_test, y_test, scaler, model, df, dep_var):
    X_test_scaled = scaler.fit_transform(X_test)

    X_test_scaled_wconst = sm.add_constant(X_test_scaled)
    
    y_pred = model.predict(X_test_scaled_wconst)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print(f"\n------------------------------------------------------------------------------------------------")
    print(f"The root mean squared error of the model on the test dataset is: {rmse:.2f}.")
    print(f"This represents a relative error of: {rmse / (df[dep_var].max() - df[dep_var].min()):.2%}.")
    print(f"------------------------------------------------------------------------------------------------\n")
    

def run_regression(df, indep_vars, dep_var, VIF=True, genre=None):
    # Filter by genre for genre specific regression
    if genre is not None:
        print(f"Running regression for genre: {genre.replace('Main_genre_', '').capitalize()}")
        df = df[df[genre] == 1]
    
    # Execute train/test split and scale data
    X_scaled_wconst_train, y_train, X_test, y_test, scaler = prepare_data(df, indep_vars, dep_var)

    if VIF:
        # Calculate and display VIF values
        vif_data = calculate_vif(X_scaled_wconst_train.iloc[:, 1:]) 
    
    # Run regression
    model = run_ols_regression(X_scaled_wconst_train, y_train)

    # Computing root mean squared error
    compute_rmse(X_test, y_test, scaler, model, df, dep_var)
