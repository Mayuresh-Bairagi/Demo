#!/usr/bin/env python
# coding: utf-8

# ### If you have a CSV or Excel file, load the data using the following code. If you don’t have a file, create the data manually, then save it as a CSV or Excel file and load it.

# In[ ]:


import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv('file.csv', encoding='utf-8')

# Read a excel file into a DataFrame
df = pd.read_excel('file.xlsx', sheet_name='Sheet1', encoding='utf-8')


# ## Or make Pandas dataframe with the help dict

# In[ ]:


import pandas as pd
import random

random_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [random.randint(20, 40) for _ in range(4)],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(random_dict)

print(df)


# ## Overview

# df.head()
# df.info()
# df.describe()
# df.columns
# df.dtypes

# #### When handling missing data, first check the amount of missing values. If the missing data is minimal, replace it with the mean. Otherwise, remove the rows using df.dropna(inplace=True).

# In[ ]:


# Detect
df.isnull().sum()

# Drop
df.dropna(inplace=True)

# Fill
df.fillna(0, inplace=True)
df['col'].fillna(df['col'].mean(), inplace=True)


# In[ ]:


#Duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)


# ## Label Encoding (Convert categories to numbers)

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])   # category_encoded is the new column with encoded values


# ## One-Hot Encoding (Create binary columns for each category)

# In[ ]:


df = pd.get_dummies(df, columns=['category'], prefix='category', drop_first=False) 


# ####  Separate the data into X and y, where X contains the independent variables (features), and y contains the dependent variable (target or predicted value).

# In[ ]:


X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual column name
y = df['target_column']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43,
)     


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ### Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)

y_pred = poly_model.predict(X_test)

print(f"Polynomial Regression (degree={degree}):")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ### Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

print("Ridge Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ### Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

print("Lasso Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression")
print(classification_report(y_test, y_pred))


# ### K-Nearest Neighbors (KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

num_neighbors = 3  # Set the number of neighbors for KNN
model = KNeighborsClassifier(n_neighbors=num_neighbors)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("KNN Classifier")
print(classification_report(y_test, y_pred))


# ### Support Vector Machine (SVM)

# In[ ]:


from sklearn.svm import SVC

model = SVC(kernel='rbf')  # or 'linear'
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("SVM Classifier")
print(classification_report(y_test, y_pred))


# ### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Decision Tree Classifier")
print(classification_report(y_test, y_pred))


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Classifier")
print(classification_report(y_test, y_pred))


# ###  Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Naive Bayes Classifier")
print(classification_report(y_test, y_pred))


# ### Confusion Matrix for only classification models

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ## Clustering 
# ###  Common Setup for Clustering

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

X = df.copy() # Assume df contains only features (no target column)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ## To decide the optimal number of clusters (n), we use the elbow method and the silhouette score

# #### Elbow Method: 

# In[ ]:


wcss = []
K_range = range(2, 11)  # testing clusters from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ is WCSS

plt.figure(figsize=(8,4))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.title('Elbow Method For Optimal k')
plt.show()


# #### Silhouette Scores

# In[ ]:


k =10
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    sil_scores.append(score)

plt.figure(figsize=(8,4))
plt.plot(K_range, sil_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()


# ### K-Means Clustering

# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

print("K-Means Silhouette Score:", silhouette_score(X_scaled, labels_kmeans))


# ###  DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

print("DBSCAN Silhouette Score:", silhouette_score(X_scaled, labels_dbscan))

