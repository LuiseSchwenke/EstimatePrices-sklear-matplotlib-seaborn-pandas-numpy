import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:,.2f}'.format
df = pd.read_csv('boston.csv', index_col=0)

### GOAL OF THE PROJECT ###
# building a model that can provide a price estimate based on a home's characteristics like:
# The number of rooms,
# How rich or poor the area is,...

# Therefore I run Multivariable Regression, data will be  trained and tested,
# evaluated by model's coefficients and residuals and log transformed


##################
# Explanation of the boston.csv data:

# Attribute Information (in order):
#    1. CRIM     per capita crime rate by town
#    2. ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#    3. INDUS    proportion of non-retail business acres per town
#    4. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#    5. NOX      nitric oxides concentration (parts per 10 million)
#    6. RM       average number of rooms per dwelling
#    7. AGE      proportion of owner-occupied units built prior to 1940
#   8. DIS      weighted distances to five Boston employment centres
#    9. RAD      index of accessibility to radial highways
#    10. TAX      full-value property-tax rate per $10,000
#    11. PTRATIO  pupil-teacher ratio by town
#    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#    13. LSTAT    % lower status of the population
#    14. PRICE     Median value of owner-occupied homes in $1000's

# Creator: Harrison, D. and Rubinfeld, D.L.
# This is a copy of UCI ML housing dataset. This dataset was taken from the StatLib
# library which is maintained at Carnegie Mellon University.
# original Research Paper: https://deepblue.lib.umich.edu/bitstream/handle/2027.42/22636/0000186.pdf?sequence=1&isAllowed=y
################

# investigate data
df.head()
df.duplicated().values.any()
df.isna().values.any()
df.describe()

# every attribute can be visualized like this:
# here: df["PRICE"]: The home price in thousands.
sns.displot(df["PRICE"], kde=True, aspect=2, bins=50, ec='black', color='#7b1fa2')
plt.title(f'1970s Home Values in Boston. Average: $ {(1000 * df.PRICE.mean()):.6}')
plt.xlabel('Price in .000s')
plt.ylabel('Nr. of Homes')
plt.show()

# checking how many homes are away from the Charles river (df["CHAS"]) versus next to it:
access_to_river = df["CHAS"].value_counts()

fig = px.bar(x=["Yes", "No"], y=access_to_river.values, color=access_to_river.values,
             title="Number of Houses located next to Charles River or not")
fig.update_layout(xaxis_title='Yes or No',
                  yaxis_title='Number of Homes',
                  coloraxis_showscale=False)
fig.show()

# getting an overview of possible relations between all attributes:
sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})

# to investigate specific relations closer:
# here: Distance from employment (df["DIS"]) with Nitric Oxide Pollution (df["NOx"])
plt.figure(figsize=(10, 5), dpi=200)
sns.jointplot(df, x="DIS", y="NOX", kind="reg", height=8, ratio=2, marginal_ticks=True, color='green')
plt.title('(Distance from employment compared with NOX (Nitric Oxide Pollution')
plt.xlabel('Distance from employment')
plt.ylabel('Nitric Oxide Pollution')
plt.show()
# In the plot, it gets more clear, that the pollution goes down with farther distance from deployment.

# test-train the model with sklearn:
# i decided to split the training and testing data into 80/20
from sklearn.model_selection import train_test_split

target = df["PRICE"]
features = df.drop('PRICE', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)

# check the train/test percentages:
test_pct = 100 * len(X_test) / len(features)
train_pct = 100 * X_train.shape[0] / features.shape[0]

# checking variance of the regression with r-squared:
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
reg_squared = linear_reg.score(X_train, y_train)
# result: 0.75, so around 75% of the variability in the target variable (df["PRICE"]) is explained
# by the regression model

# It further makes sense to investigate the regression coefficients:
regr_coef = pd.DataFrame(data=linear_reg.coef_, index=X_train.columns, columns=['Coefficient'])
# out of interest:
# Extra price (in $) for having an extra room
premium = regr_coef.loc['RM'].values[0] * 1000
# here it would be 3108.45$

# checking residuals (difference between the model's predictions and the original values
# inside y_train of the regression:
predicted_values = regr_coef.predict(X_train)
residuals = (y_train - predicted_values)

# plotting the residuals against the predicted prices:
plt.scatter(x=predicted_values, y=residuals, alpha=0.4, c="red")
plt.title('Predicted vs real Residuals in .000s $')
plt.xlabel('predicted house prices in .000s $')
plt.ylabel('Residuals')
plt.show()
# residuals should look random, to see that the model has no systematic bias
# investing skew and mean of residuals and plotting it:
res_mean = round(residuals.mean(), 2)
res_skew = round(residuals.skew(), 2)
sns.displot(residuals, kde=True, aspect=2, bins=50)
plt.title(f'Residuals Skew: ({res_skew}), Mean: ({res_mean})')
plt.xlabel('Distance from mean')
plt.ylabel('Probability density')
plt.show()
# unfortunate, the skew is quite high with a value of 1.46
# to try not having to change to a different model than linear, I calculate the log of df["PRICE"],
# to make it fit better with the linear regression model (and visualize the log prices):
price = df["PRICE"]
price_skew = round(price.skew(), 2)

logs = np.log(price)
skew_logpr = round(logs.skew(), 2)

sns.displot(logs, kde=True, aspect=2, bins=50)
plt.title(f'Log prices with skew {skew_logpr}')
plt.xlabel('Prices log')
plt.ylabel('Nr. of Homes')
plt.show()
# the skew now is by -0.33, which is definitely closer to 0 than 1.46
# this can also be visualized by plotting the actual prices against the log prices:
plt.figure(dpi=150)
plt.scatter(df.PRICE, np.log(df.PRICE))

plt.title('The Original Price compared to the Log Price')
plt.ylabel('Log Price')
plt.xlabel('Actual $ Price in 000s')
plt.show()

# test-train the model again, this time with the log prices:
log_target = logs
log_features = df.drop('PRICE', axis=1)
log_X_train, log_X_test, log_y_train, log_y_test = train_test_split(log_features, log_target, test_size=0.2,
                                                                    random_state=10)
# log_train_pct = 100*len(log_X_train)/len(log_features)
# log_test_pct = 100*len(log_X_test)/len(log_features)

log_linear_reg = LinearRegression()
log_linear_reg.fit(log_X_train, log_y_train)
log_reg_squared = log_linear_reg.score(log_X_train, log_y_train)
# the r-squared of the regression with log prices is now by 0.79 so 79%. A higher r-squared CAN be good

log_regr_coef = pd.DataFrame(data=log_linear_reg.coef_, index=log_X_train.columns, columns=['Coef_log'])
# looking at the coefficients, e.g. being close to the river results in higher property prices
# because CHAS has a coefficient greater than zero. Therefore, property prices are higher
# next to the river. On the other hand a negative value for the students-teacher ration (df["PTRATIO"]
# indicates fewer students on one teacher, which is normally a sign for higher education level

log_predicted_values = log_linear_reg.predict(log_X_train)
log_residuals = (log_y_train - log_predicted_values)

# for visualization, the regression of the original price vd predicted price and
# log price vs predicted price as well as residuals vs predicted prices and residuals vs
# predicted log prices ca be plotted like this example:

plt.scatter(x=log_y_train, y=log_predicted_values, alpha=0.4, c="navy")
plt.plot(log_y_train, log_y_train, color="cyan")
plt.title(f'Original vs Predicted Log Prices: $y _i$ vs $\hat y_i$ (R-Squared {log_reg_squared:.2})', fontsize=17)
plt.xlabel('Log house prices in .000s $')
plt.ylabel('predicted house prices in .000s $')
plt.show()

# calculation the mean and skew for the log prices:
log_resid_mean = round(log_residuals.mean(), 2)
log_resid_skew = round(log_residuals.skew(), 2)
# plotted:
sns.displot(log_residuals, kde=True, color='navy')
plt.title(f"Log price model: Residual {log_resid_mean}, Skew: {log_resid_skew}")

sns.displot(residuals, kde=True, color='indigo')
plt.title(f"Original price model: Residual {res_mean}, Skew: {res_skew}")

# comparing the values:
# skew of log price data now is by 0.09, which is much closer to 0 than 1.49 of the original price data

# reg_squared = linear_reg.score(X_test, y_test)
log_reg_squared = log_linear_reg.score(log_X_test, log_y_test)
# Original Price Model Test Data r-squared: 0.67
# Log Price Model Test Data r-squared: 0.74
# the values are lower than for the training data but still high, so the model can be usefull

# for interest, we can now e.g. see the predicted price for a average home (means it has the mean
# values of all the attributes:
all_features = df.drop('PRICE', axis=1)
all_features_mean = all_features.mean().values
all_attributes = pd.DataFrame(data=all_features_mean.reshape(1, len(all_features.columns)),
                              columns=all_features.columns)
log_price_pred = log_linear_reg.predict(all_attributes)[0]
# in $ with numpy.exp():
dollar_price_pred = np.exp(log_linear_reg.predict(all_attributes)[0]) * 1000

# now it is possible to value a property by set attribute values like this:
next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = df.NOX.quantile(q=0.75)  # high
amount_of_poverty = df.LSTAT.quantile(q=0.25)  # low

all_attributes["LSTAT"] = amount_of_poverty
all_attributes["NOX"] = pollution
all_attributes["RM"] = nr_rooms
all_attributes["DIS"] = distance_to_town
all_attributes["PTRATIO"] = students_per_classroom
if next_to_river:
    all_attributes["CHAS"] = 1
else:
    all_attributes["CHAS"] = 0

defined_price_pred = log_linear_reg.predict(all_attributes)[0]
defined_dollar_pred = np.exp(log_linear_reg.predict(all_attributes)[0]) * 1000
# the result would in this case be $ 25792.02
