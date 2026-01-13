import pandas as pd
import os

def load_data(data_path='.'):
    """
    Load user_product_interactions.csv and products.csv.
    """
    
    # Load Ratings (Interactions)
    ratings_file = os.path.join(data_path, 'user_product_interactions.csv')
    ratings = pd.read_csv(ratings_file)
    
    # Load Products
    products_file = os.path.join(data_path, 'products.csv')
    products = pd.read_csv(products_file)
    
    return ratings, products

def get_user_item_matrix(ratings):
    """
    Create the user-item interaction matrix.
    Row: User
    Col: Product
    Value: Rating
    """
    return ratings.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
