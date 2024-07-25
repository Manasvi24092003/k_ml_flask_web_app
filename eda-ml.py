#!/usr/bin/env python
# coding: utf-8

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from kneed import KneeLocator
from sqlalchemy import create_engine

# 1. Load Data
airstat = pd.read_csv(r"C:\Users\manas\Downloads\Data Set\Data Set (5)\AirTraffic_Passenger_Statistics.csv")

# 2. Save Data to MySQL
user = 'root'
pw = '2170'
db = 'airstat'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
airstat.to_sql('airstat_tbl', con=engine, if_exists='replace', chunksize=1000, index=False)

# 3. Exploratory Data Analysis (EDA)
df = airstat.copy()
df.describe()

# Automated EDA
import sweetviz
my_report = sweetviz.analyze([df, "df"])
my_report.show_html('Report.html')

# 4. Handling Missing Data
df.dropna(inplace=True)

# 5. Data Preprocessing Pipeline
numeric_features = df.select_dtypes(exclude=['object']).columns
num_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])
processed = num_pipeline.fit(df[numeric_features])
joblib.dump(processed, 'processed1')
airstat_clean = pd.DataFrame(processed.transform(df[numeric_features]), columns=numeric_features)

# 6. Clustering Model Building
TWSS = [KMeans(n_clusters=i).fit(airstat_clean).inertia_ for i in range(2, 9)]
plt.plot(range(2, 9), TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.show()

# Determine the optimal number of clusters
kl = KneeLocator(range(2, 9), TWSS, curve='convex')
k = kl.elbow
print(f"Optimal number of clusters: {k}")

# Train the KMeans model
bestmodel = KMeans(n_clusters=k)
bestmodel.fit(airstat_clean)
print(f"Silhouette Score: {metrics.silhouette_score(airstat_clean, bestmodel.labels_)}")

# Save the trained model
pickle.dump(bestmodel, open('Clust_airstat.pkl', 'wb'))

# Concatenate the cluster labels with the original data
df['cluster_id'] = bestmodel.labels_

# Save the final dataframe to a CSV file
df.to_csv('KMeans_Airstat.csv', encoding='utf-8', index=False)
