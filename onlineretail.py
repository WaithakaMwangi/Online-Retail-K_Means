import pandas as pd

#load data
data=pd.read_csv("OnlineRetail.csv", encoding = 'ISO-8859-1', header=0)

# trying to understand the data
print (data.head(10))
# the data is a sales data for a business

# check the amount of data in the dataset
print (data.shape)

# the column and the datatype of the columns
print(data.info())

# get the mean, mode, count etc
print(data.describe())

# check if there are null values and sum them
print (data.isnull().sum())

# calculate the missing values in DataFrame and truncated into 2 decimal places
df_null=round(100*(data.isnull().sum())/len(data),2)
df_null
print(df_null)

# after checking if there are null values we drop the rows that have missing values
data = data.dropna()
print (data.shape)

# changing the datatype of CustomerID as business understanding
data['CustomerID'] = data['CustomerID'].astype(str)
print (data.info())

# Data Preparation
# New attribute : Monetary
data['Amount'] = data['Quantity']*data['UnitPrice']
print(data.head())

# checking which customer makes the most sales
data_monitoring=data.groupby('CustomerID')['Amount'].sum()
print(data_monitoring.head())

'''customer_most_purchase = data_monitoring.idxmax()
print("This is the customer who made the most purchases",customer_most_purchase)
amount_bought = data_monitoring.max()
print("This is the amount bought by the customer",amount_bought)
'''

#  which product made the most sales
'''product_monitoring = data.groupby('Description')['Amount'].sum().sort_values(ascending=False)
print(product_monitoring)'''
# find the product description (index) of the most sold product
'''product_most_sold = product_monitoring.idxmax()
print("This is the most sold product",product_most_sold)'''
# Find the amount sold for the most popular product
'''amount_sold = product_monitoring.max()
print("This is the amount sold for most popular",amount_sold)'''

#  which product was sold the most
product_monitoring = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(product_monitoring)

# find the product description (index) of the most sold product
'''product_most_sold = product_monitoring.idxmax()
print("This is the most sold product in terms of quantity",product_most_sold)'''
# Find the amount of product sold for the most popular product
'''amount_sold = product_monitoring.max()
print("This is the amount sold for most popular",amount_sold)'''

# what country is most product sold
'''country_monitoring = data.groupby('Country')['Amount'].sum()
print(country_monitoring)'''
# find the country where most products were sold(the index)
'''country_most_popular = country_monitoring.idxmax()
print("This is country most product are sold",country_most_popular)'''
# Find the amount sold for that country
'''region_amount_sold = country_monitoring.max()
print("This is the amount sold for that country",region_amount_sold)'''

# what country is most product sold
country_monitoring = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
country_monitoring = country_monitoring.reset_index()
country_monitoring.columns = ['Country','Quantity']
print(country_monitoring)

# find the country where most products were sold(the index)
'''country_most_popular = country_monitoring.idxmax()
print("This is country most product are sold",country_most_popular)'''
#Find the amount sold for that country
'''region_amount_sold = country_monitoring.max()
print("This is the amount sold for that country",region_amount_sold)
'''

data_monitoring = data_monitoring.reset_index()
print(data_monitoring.head())

#The idxmax function applied returns the index.
# The max function applied returns the maximum value.

# frequency
data_frequency = (data.groupby('Description')["InvoiceNo"].count().sort_values
                  (ascending=False))
print(data_frequency)

# frequently sold
frequent_monitoring = data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
frequent_monitoring = frequent_monitoring.reset_index()
frequent_monitoring.columns = ['Description','InvoiceNo']
print(frequent_monitoring)

# new attribute : Recency
# convert the date/time to proper datatype

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],format = '%m/%d/%Y %H:%M')

# compute the maximum date to know the last transaction date

max_date = max(data['InvoiceDate'])
print(max_date)

# compute the minimum date to know the last transaction date

min_date = min(data['InvoiceDate'])
print(min_date)

# days where the sales took place

days = max_date - min_date
print(days)

from datetime import timedelta
new_min_date = max_date - timedelta(days=30)
print(new_min_date)

# filter the data for the last month

last_month_data = data[data['InvoiceDate'] >= new_min_date]
# after getting the new_min_date

# sales made in those 30 days
last_30_days_sales = data[(data['InvoiceDate'] >= new_min_date) & (data['InvoiceDate'] <= max_date)]['Amount'].sum()
print(last_30_days_sales)

# calculate the total number of sale over the last one month

total_sales = data[(data['InvoiceDate'] >= new_min_date) & (data['InvoiceDate'] <= max_date)]['Amount'].count()
print(total_sales)


# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
import seaborn as sns


# Scale the data
scaler = StandardScaler()
data_monitoring_scaled = scaler.fit_transform(data_monitoring[['Amount']])

# Define the range for the number of clusters
K = range(2, 10)

# Calculate Silhouette Scores for the range of K and store the KMeans models
silhouette_scores = []
models = {}
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_monitoring_scaled)
    score = silhouette_score(data_monitoring_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    models[k] = kmeans
print("\nScores: ", silhouette_scores)


# Plotting the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
# 'bo-' means blue color, circle markers, solid line
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal values of K')
plt.show()

# Optimal K based on the highest silhouette score
# Optimal value of K is where the score appears to be the highest
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print('The optimal value of K is', {optimal_k})

# Visualizing the clusters for the optimal k (Which is two in our case)
kmeans_optimal = models[optimal_k]
data_monitoring['Cluster'] = kmeans_optimal.labels_

plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    cluster_data = data_monitoring[data_monitoring['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Amount'], label=f'Cluster {cluster}')

plt.xlabel('Customer ID Index')
plt.ylabel('Monetary Value')
plt.title(f'Customer Clusters for k = {optimal_k}')
plt.legend()
plt.show()

fits = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(data_monitoring_scaled)
    
    fits.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(fits)


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=40)
    kmeans.fit(data_monitoring_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(data_monitoring_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(data_monitoring_scaled)



