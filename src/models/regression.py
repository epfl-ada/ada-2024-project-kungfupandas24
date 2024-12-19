import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def prepare_data(df, indep_vars, dep_var):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[indep_vars])
    X_scaled = pd.DataFrame(X_scaled, columns=indep_vars, index=df.index)
    X_scaled_wconst = sm.add_constant(X_scaled)
    y = df[dep_var]
    return X_scaled_wconst, y

def run_ols_regression(X, y):
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model

def run_regression_for_genre(df, genre, indep_vars, dep_var):
    df_genre = df[df[genre] == 1]
    X_scaled_wconst, y = prepare_data(df_genre, indep_vars, dep_var)
    print(f"\n\nRegression Results for {genre.replace('Main_genre_', '').capitalize()}:\n")
    model = run_ols_regression(X_scaled_wconst, y)
    vif_data = calculate_vif(X_scaled_wconst.iloc[:, 1:])
    print("\nVIF values:\n", vif_data)
