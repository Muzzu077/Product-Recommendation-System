import pandas as pd

class RuleBasedRecommender:
    """
    Simulates a non-personalized recommendation engine.
    """
    
    def __init__(self, ratings_df, items_df):
        """
        ratings_df: DataFrame containing user-product interactions (user_id, product_id, rating)
        items_df: DataFrame containing product details (product_id, product_name, category)
        """
        self.ratings = ratings_df
        self.items = items_df
        
    def get_top_popular_products(self, n=10):
        """
        Recommends products based on the number of ratings (popularity).
        """
        # Count number of ratings for each product
        popularity_counts = self.ratings.groupby('product_id').size().reset_index(name='interaction_count')
        
        # Sort by count descending
        top_products = popularity_counts.sort_values(by='interaction_count', ascending=False).head(n)
        
        # Merge with item details to get titles
        top_products = pd.merge(top_products, self.items, on='product_id')
        
        return top_products[['product_id', 'product_name', 'category', 'interaction_count']]

    def get_top_rated_products(self, n=10, min_interactions=50):
        """
        Recommends products based on average rating.
        Filters out products with few interactions to avoid noise.
        """
        # Calculate average rating and count
        product_stats = self.ratings.groupby('product_id').agg(
            avg_rating=('rating', 'mean'),
            count=('rating', 'count')
        ).reset_index()
        
        # Filter by minimum interactions
        qualified_products = product_stats[product_stats['count'] >= min_interactions]
        
        # Sort by average rating descending
        top_rated = qualified_products.sort_values(by='avg_rating', ascending=False).head(n)
        
        # Merge with item details
        top_rated = pd.merge(top_rated, self.items, on='product_id')
        
        return top_rated[['product_id', 'product_name', 'category', 'avg_rating', 'count']]
        
    def get_recommendations(self, method='popular', n=10):
        if method == 'popular':
            return self.get_top_popular_products(n)
        elif method == 'top_rated':
            return self.get_top_rated_products(n)
        else:
            raise ValueError("Unknown method. Choose 'popular' or 'top_rated'.")
