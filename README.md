import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
# ========= FIX BLUR =========
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
# For Colab (extra clarity)
%config InlineBackend.figure_format = 'retina'
# ========= LOAD DATA =========
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
# ========= PREPROCESS =========
df = df.drop(columns=['Order','PID'], errors='ignore')
df.fillna(df.median(numeric_only=True), inplace=True)
# FIGURE 1: EDA (UPGRADED)
fig, axs = plt.subplots(2,3, figsize=(18,12))
# Histogram + KDE
sns.histplot(df['SalePrice'], kde=True, ax=axs[0,0])
axs[0,0].set_title("SalePrice Distribution")
# Log Histogram
sns.histplot(np.log(df['SalePrice']), kde=True, ax=axs[0,1])
axs[0,1].set_title("Log SalePrice")
# Boxplot
sns.boxplot(y=df['SalePrice'], ax=axs[0,2])
axs[0,2].set_title("Boxplot")
# Heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, cmap='coolwarm', ax=axs[1,0])
axs[1,0].set_title("Correlation Heatmap")
# Scatter
axs[1,1].scatter(df['Gr Liv Area'], df['SalePrice'], alpha=0.6)
axs[1,1].set_title("Living Area vs Price")
# Quality vs Price
axs[1,2].scatter(df['Overall Qual'], df['SalePrice'], alpha=0.6)
axs[1,2].set_title("Quality vs Price")
plt.tight_layout()
plt.show()
# EXTRA: PAIRPLOT (TOP FEATURES)
cols = ['SalePrice','Gr Liv Area','Overall Qual','Total Bsmt SF']
cols = [c for c in cols if c in df.columns]
sns.pairplot(df[cols])
plt.show()
# ENCODE
df = pd.get_dummies(df, drop_first=True)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# MODEL 1
m1 = LinearRegression().fit(X_train, y_train)
pred1 = m1.predict(X_test)
# MODEL 2 (SELECTED FEATURES)
cols = ['Overall Qual','Gr Liv Area','Total Bsmt SF','Garage Area']
cols = [c for c in cols if c in X.columns]
m2 = LinearRegression().fit(X_train[cols], y_train)
pred2 = m2.predict(X_test[cols])
# MODEL 3 (LOG TRANSFORM)
Xt_train = X_train.copy()
Xt_test = X_test.copy()
if 'Gr Liv Area' in X.columns:
 Xt_train['log_GrLiv'] = np.log1p(X_train['Gr Liv Area'])
 Xt_test['log_GrLiv'] = np.log1p(X_test['Gr Liv Area'])
y_log = np.log(y_train)
m3 = LinearRegression().fit(Xt_train, y_log)
pred3 = np.exp(m3.predict(Xt_test))
# FIGURE 2: MODEL COMPARISON
fig, axs = plt.subplots(1,3, figsize=(18,5))
for ax, pred, title in zip(axs,[pred1,pred2,pred3],
 ["Model 1","Model 2","Model 3"]):
 ax.scatter(y_test, pred, alpha=0.6)
 ax.plot([y_test.min(), y_test.max()],
 [y_test.min(), y_test.max()], linestyle='dashed')
 ax.set_title(title)
 ax.set_xlabel("Actual")
 ax.set_ylabel("Predicted")
plt.tight_layout()
plt.show()
# FIGURE 3: OUTLIER ANALYSIS
z = np.abs(stats.zscore(y_train))
mask = z < 3
X_no = X_train[mask]
y_no = y_train[mask]
m_out = LinearRegression().fit(X_no, y_no)
pred_out = m_out.predict(X_test)
plt.figure(figsize=(8,5))
plt.scatter(y_test, pred_out, alpha=0.6)
plt.title("Outlier Removed Model")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# FIGURE 4: STRATIFIED MODEL
df_strat = df.copy()
if 'Overall Qual' in df_strat.columns:
 df_strat['Group'] = pd.cut(df_strat['Overall Qual'],
 bins=[0,4,6,7,10],
 labels=['Low','Mid','High','Premium'])
 group_rmse = {}
for g in df_strat['Group'].dropna().unique():
  sub = df_strat[df_strat['Group']==g]
  if len(sub)>50:
    Xg = sub.drop(['SalePrice','Group'], axis=1)
    yg = sub['SalePrice']
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(Xg,yg,test_size=0.2)
    model = LinearRegression().fit(Xg_train,yg_train)
    pred = model.predict(Xg_test)
    rmse = np.sqrt(mean_squared_error(yg_test,pred))
    group_rmse[g] = rmse
plt.figure(figsize=(8,5))
plt.bar(group_rmse.keys(), group_rmse.values())
plt.title("RMSE by Quality Group")
plt.show()
# FIGURE 5: ERROR ANALYSIS
errors = (pred3 - y_test) / y_test * 100
plt.figure(figsize=(8,5))
sns.histplot(errors, kde=True)
plt.title("Prediction Error (%)")
plt.show()
# FIGURE 6: RESIDUAL ANALYSIS
residuals = y_test - pred3
# Residual plot
plt.figure(figsize=(8,5))
plt.scatter(pred3, residuals, alpha=0.6)
plt.axhline(0, linestyle='dashed')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
# Q-Q plot
plt.figure(figsize=(6,6))
stats.probplot(residuals, plot=plt)
plt.title("Q-Q Plot")
plt.show()
# EXTRA: FEATURE IMPORTANCE
importance = pd.Series(m1.coef_, index=X.columns)
top_features = importance.abs().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
top_features.sort_values().plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()
# FINAL METRICS
def evaluate(y_true, y_pred, name):
 r2 = r2_score(y_true, y_pred)
 rmse = np.sqrt(mean_squared_error(y_true, y_pred))
 print(f"{name} -> R2:", r2, " RMSE:", rmse)
print("\nFINAL RESULTS:")
evaluate(y_test, pred1, "Model 1")
evaluate(y_test, pred2, "Model 2")
evaluate(y_test, pred3, "Model 3") 
