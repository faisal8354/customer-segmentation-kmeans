
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(7)
n_customers = 200
customers = pd.DataFrame({
    'CustomerID': np.arange(1, n_customers + 1),
    'Recency': np.random.randint(1, 365, n_customers),
    'Frequency': np.random.poisson(12, n_customers),
    'Monetary': np.random.gamma(2, 150, n_customers)
})

customers.to_csv('data/customers.csv', index=False)
df = pd.read_csv('data/customers.csv')

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=3, random_state=7)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

def cluster_summary(df):
    return df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})

print("\n=== Cluster summary ===")
print(cluster_summary(df))

plt.figure(figsize=(8,5))
colors = ['#007acc', '#ff6600', '#33cc33']
for cluster in df['Cluster'].unique():
    temp = df[df['Cluster'] == cluster]
    plt.scatter(temp['Recency'], temp['Monetary'],
                label=f'Cluster {cluster}',
                alpha=0.6, s=60, c=colors[cluster])
plt.xlabel('Recency (days)')
plt.ylabel('Monetary ($)')
plt.title('Customer Segments')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
