# Customer Segmentation with K-Means

Small Python project to try out clustering on synthetic ecommerce customer data. Mainly built to practice pandas, sklearn and quick visual plots.

## What it does
- Creates sample data for 200 customers with:
  - Recency: days since last purchase
  - Frequency: number of orders
  - Monetary: total amount spent
- Scales these RFM features and runs K-Means clustering (3 segments).
- Prints average RFM stats by cluster and shows a scatter plot of Recency vs Monetary colored by segment.

## How to run
Install the requirements first:
pip install -r requirements.txt

Then simply run:
python customer_segmentation.py

It will print a summary of clusters and open a scatter plot.

## Note
Mostly done to explore clustering ideas on customer data — could easily swap in real ecommerce transactions later. Good practice for scaling, unsupervised learning, and quick matplotlib plots.
