# %%
# Importing required libraries 

import math
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, f1_score

from joblib import dump
from joblib import load

sns.set()

# %%
jet_data = pd.read_csv(
    r"C:\Users\PMLS\Desktop\fyp\Predictive Maintanance\Predictive Maintenance (NASA TurboFan Engine)\train_FD001.txt", sep=" ", header=None)
jet_rul = pd.read_csv(
    r"C:\Users\PMLS\Desktop\fyp\Predictive Maintanance\Predictive Maintenance (NASA TurboFan Engine)\RUL_FD001.txt", sep=" ", header=None)
test_data = pd.read_csv(
    r"C:\Users\PMLS\Desktop\fyp\Predictive Maintanance\Predictive Maintenance (NASA TurboFan Engine)\test_FD001.txt", sep="\s+", header=None)
jet_data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21","sensor22","sensor23"]
test_data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21"]

jet_data.drop(['sensor22', 'sensor23'], axis=1, inplace=True)

jet_rul.columns = ['cycles', 'id']
jet_rul['id'] = jet_data['id'].unique()
jet_rul.set_index('id', inplace=True)

jet_id_and_rul = jet_data.groupby(['id'])[["id" ,"cycle"]].max()
jet_id_and_rul.set_index('id', inplace=True)



# %%
#adding RUL column to the data
def RUL_calculator(df, df_max_cycles):
    max_cycle = df_max_cycles["cycle"]
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"] 
    result_frame.drop(['max_cycle'], axis=1, inplace=True)
    return result_frame

jet_data = RUL_calculator(jet_data, jet_id_and_rul)

# %%
jet_data.describe()

# %%
jet_data.head()

# %%
# Visualize total number of cycles by each engine

jet_id_and_rul = jet_data.groupby(['id'])[["id" ,"cycle"]].max()

f, ax = plt.subplots(figsize=(10, 15))
sns.set_color_codes("pastel")
sns.barplot(x="cycle", y="id", data=jet_id_and_rul, label="Total Cycles", color="blue", orient = 'h', dodge=False)
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylim=(0, 100), ylabel="",xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)
ax.tick_params(labelsize=11)
ax.tick_params(length=0, axis='x')
ax.set_ylabel("Engine Number", fontsize=11)
ax.set_xlabel("Number of Cycles", fontsize=11)
plt.tight_layout()
plt.show()

# %%
# Determining mean number of cycles and histogram of number of cycle

plt.subplots(figsize=(12, 6))
sns.histplot(jet_id_and_rul["cycle"], kde = True, color='red');
print("Mean number of cycles after which jet engine fails is "+ str(math.floor(jet_id_and_rul["cycle"].mean())))

# %%
#Histogram representation of each sensor data

sns.set()
fig = plt.figure(figsize = [15,10])
cols = jet_data.columns
cnt = 1
for col in cols :
    plt.subplot(8,4,cnt)
    sns.histplot(jet_data[col],color='purple')
    cnt+=1
plt.tight_layout()
plt.show() 

# %%
#Creating a heatmap to compare with RUL

plt.figure(figsize=(13,8))
cmap = sns.diverging_palette(500, 10, as_cmap=True)
sns.heatmap(jet_data.corr(), cmap =cmap, center=0, annot=False, square=True);

# %%


# %%
#New Dataframe with relevant parameters

jet_relevant_data = jet_data.drop(["cycle", "op1", "op2", "op3", "sensor1", "sensor5", "sensor6", "sensor10", "sensor16", "sensor18", "sensor19", "sensor14", "sensor13", "sensor12", "sensor11"], axis=1)

# %%
#Updated heatmap with relevant parameters

plt.figure(figsize=(15, 12))
cmap = sns.diverging_palette(500, 10, as_cmap=True)
sns.heatmap(jet_relevant_data.corr(), cmap =cmap, center=0, annot=True, square=True);

# %%
def plot_sensor(sensor_name,X):
    plt.figure(figsize=(13,5))
    for i in X['id'].unique():
        if (i % 10 == 0):  # only plot every engine
            plt.plot('RUL', sensor_name, 
                     data=X[X['id']==i].rolling(8).mean())
            plt.axvline(30, color='red', linestyle='dashed', linewidth=2)
    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()

# %%
for sensor in jet_relevant_data.drop(['id', 'RUL'], axis=1).columns:
    plot_sensor(sensor, jet_relevant_data)

# %%
# Based on the above data variation, we removed sensor 9
jet_relevant_data.drop('sensor9', axis=1, inplace=True)

# %%
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(jet_relevant_data.drop(['id', 'RUL'], axis=1))
scaled_features = pd.DataFrame(scaled_features, columns=jet_relevant_data.drop(['id', 'RUL'], axis=1).columns)

# %%
scaled_features['id'] = jet_relevant_data['id']
scaled_features['RUL'] = jet_relevant_data['RUL']

# %%
scaled_features.head()

# %%
data = scaled_features.copy()


# %%
cycle=30
data['label'] = data['RUL'].apply(lambda x: 1 if x <= cycle else 0)

# %%
y = data['label']
X = data.drop(['RUL', 'id', 'label'], axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3)

print('X_train shape : ',X_train.shape)
print('X_test shape : ',X_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)

# %%
classifier = RandomForestClassifier(random_state=90, oob_score = False)

# %%
# Define the parameter Grid
params = {
 'max_depth': [18, 20, 22],
 'max_features': ['auto', 'sqrt'],
 'min_samples_split': [22, 25],
 'min_samples_leaf': [12, 10, 8],
 'n_estimators': [20, 30, 40]
}
# Initialize the Grid Search with accuracy metrics 
grid_search = GridSearchCV(estimator=classifier,
                                  param_grid=params,
                                  cv = 5,
                                  scoring="f1")
# Fitting 5 Folds for each of 108 candidates, total 540 fits
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# Let's check the score
grid_search.best_score_

# %%
grid_search.best_params_

# %%
pred = grid_search.predict(X_test)

# %%
dump(grid_search.best_estimator_, 'random_forest_model.joblib')

# %%
acc_score = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print('Acc Score: {:.2%}'.format(acc_score))
print('Roc Auc Score: {:.2%}'.format(roc_auc))
print('Precision Score: {:.2%}'.format(precision))
print('Recall Score: {:.2%}'.format(recall))
print('F1 Score: {:.2%}'.format(f1))

# %%
plt.figure(figsize=(15,8))

cm = confusion_matrix(y_test, pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot()
plt.grid(False)
plt.show()


# %%
best_model = grid_search.best_estimator_

import matplotlib.colors as mcolors

# Assuming best_model, feature_importance, sorted_idx, and X are defined as in your code

# Normalize the feature importance scores
feature_importance = best_model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[-50:]  # Select the top 50 important features

# Define custom light to darker color gradient
light_color = '#C0C0C0'  # Light color (light grey)
dark_color = '#4B0082'   # Dark color (indigo)

# Generate a custom colormap transitioning from light to dark
colors = [light_color, dark_color]
num_colors = len(sorted_idx)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, num_colors)

# Plotting the horizontal bar chart with custom gradient colors
plt.figure(figsize=(10, 12))

# Calculate positions for bars
positions = np.arange(len(sorted_idx))

# Plot the horizontal bar chart with specified custom colormap
bars = plt.barh(positions, feature_importance[sorted_idx], align='center', color=custom_cmap(np.linspace(0, 1, num_colors)))

plt.yticks(positions, X.columns[sorted_idx])
plt.xlabel('Score')
plt.title('Feature Importance')

# Customize grid appearance
plt.grid(color='gray', linestyle='dashdot', linewidth=0.5)  # Customize grid lines

# Display the plot
plt.show()

# %%
dump(grid_search.best_estimator_, 'random_forest_model.pkl')

# %%
loaded_model = load('random_forest_model.pkl')

# %%
pred = loaded_model.predict(X_test)

acc_score = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print('Acc Score: {:.2%}'.format(acc_score))
print('Roc Auc Score: {:.2%}'.format(roc_auc))
print('Precision Score: {:.2%}'.format(precision))
print('Recall Score: {:.2%}'.format(recall))
print('F1 Score: {:.2%}'.format(f1))

# %%
# Use the trained model to predict labels for the test set
predicted_labels = best_model.predict(X_test)

# Create a dataframe with engine ID and its predicted class
result_df = pd.DataFrame({
    'Engine_ID': data.loc[X_test.index, 'id'],
    'Predicted_Class': predicted_labels,
    'RUL': data.loc[X_test.index, 'RUL']
})

# Display the resulting dataframe
result_df


# %%
failing_engines_df = result_df[result_df['Predicted_Class'] == 1]

# Display the filtered DataFrame
print(failing_engines_df)

# Export the data to a text file
#failing_engines.to_csv('failing_engines.txt', sep=',', index=False)

#print("Data of engines predicted as 1 has been exported to 'failing_engines_data.txt'.")

# %%
# Filter engines that are predicted as 1
failing_engines = result_df[result_df['Predicted_Class'] == 1]

# Merge with the original data to get the complete information of the failing engines
failing_engines_data = jet_data[jet_data['id'].isin(failing_engines['Engine_ID'])]

# Export the data to a text file
failing_engines_data.to_csv('failing_engines_data.txt', sep=',', index=False)

print("Data of engines predicted as 1 has been exported to 'failing_engines_data.txt'.")



# %%
result_df['Predicted_Class'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Predicted Classes')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# %%
# Use the trained model to predict labels for the test set
predicted_labels = loaded_model.predict(X_test)

# Create a DataFrame with engine ID and its predicted class
result_df = pd.DataFrame({'Engine_ID': data.loc[X_test.index, 'id'], 'Predicted_Class': predicted_labels})

# Add Remaining RUL to the DataFrame for non-failing engines
result_df['Remaining_RUL'] = data.loc[X_test.index, 'RUL']

# Filter non-failing engines and display the resulting DataFrame with sorted Engine IDs
result_df_sorted = result_df[result_df['Predicted_Class'] == 0].sort_values(by='Engine_ID')

result_df_sorted['Correct_Prediction'] = result_df_sorted['Remaining_RUL'] == 0

print(result_df_sorted)


# %%
# Filter non-failing engines
non_failing_engines = result_df_sorted[result_df_sorted['Predicted_Class'] == 0]

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting Predicted Class
ax1.bar(non_failing_engines['Engine_ID'], non_failing_engines['Predicted_Class'], color='blue', label='Predicted Class')
ax1.set_xlabel('Engine ID')
ax1.set_ylabel('Predicted Class', color='blue')
ax1.tick_params('y', colors='blue')

# Creating a second y-axis for Remaining RUL
ax2 = ax1.twinx()
ax2.plot(non_failing_engines['Engine_ID'], non_failing_engines['Remaining_RUL'], color='red', label='Remaining RUL')
ax2.set_ylabel('Remaining RUL', color='red')
ax2.tick_params('y', colors='red')

plt.title('Predicted Class and Remaining RUL for Non-Failing Engines')
plt.show()

# %%
# Use the trained model to predict labels for the test set
predicted_labels = loaded_model.predict(X_test)

# Create a dataframe with engine ID, predicted class, and RUL
result_df = pd.DataFrame({
    'Engine_ID': data.loc[X_test.index, 'id'],
    'Predicted_Class': predicted_labels,
    'RUL': data.loc[X_test.index, 'RUL']
})

# Calculate the maximum cycle for each engine from the input data
max_cycles = jet_data.groupby('id')['cycle'].max().reset_index()
max_cycles.columns = ['Engine_ID', 'Max_Cycle']

# Merge max cycles with the result DataFrame
result_df = result_df.merge(max_cycles, on='Engine_ID', how='left')

# Filter the DataFrame to only include engines predicted as class 1
failing_engines_df = result_df[result_df['Predicted_Class'] == 1]

# Display the filtered DataFrame
print(failing_engines_df)

# Export the filtered data to a text file
failing_engines_df.to_csv('failing_engines_with_rul_and_max_cycle.txt', sep='\t', index=False)

print("Data of engines predicted as class 1 with RUL and max cycle has been exported to 'failing_engines_with_rul_and_max_cycle.txt'.")


# %%
max_cycle = jet_id_and_rul["cycle"]

print(max_cycle)
max_cycle.to_csv('failing_engines_with_rul_and_max_cycle.txt', sep=' ', index=False)

print("Data of engines predicted as class 1 with RUL and max cycle has been exported to 'failing_engines_with_rul_and_max_cycle.txt'.")

# %%
average_max_cycle = max_cycles['Max_Cycle'].mean()

# Print the average max cycle
print(f"The average of max_cycle is: {average_max_cycle:.2f}")

# %%