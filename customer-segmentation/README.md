
# Customer Segmentation with K-Means

Small Python project to try out clustering on synthetic ecommerce customer data.  
Mainly built to practice pandas, sklearn and quick visual plots.

## What it does
- Creates sample data with Recency (days since last order), Frequency (orders), and Monetary (total spent).
- Scales the data and runs K-Means clustering with 3 segments.
- Prints average RFM stats by cluster and plots Recency vs Monetary colored by segment.

## How to run
Install requirements first:

    pip install -r requirements.txt

Then simply run:

    python customer_segmentation.py

It'll print cluster summary stats and show a scatter plot.

## Note
Mostly done to play around with clustering ideas — could easily swap to real ecommerce or transaction data later.
