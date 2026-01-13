import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data.loader import get_user_item_matrix

class CollaborativeRecommender:
    """
    Implements Item-Based Collaborative Filtering.
    """
    
    def __init__(self, ratings_df, items_df):
        self.ratings = ratings_df
        self.items = items_df
        self.user_item_matrix = get_user_item_matrix(self.ratings)
        self.item_similarity_df = None
        self.train_model()

    def train_model(self):
        """
        Compute the cosine similarity between items (products).
        """
        # Calculate cosine similarity between items (columns of the matrix)
        # The matrix is User x Product. We want similarity between Products.
        # Transpose to get Product x User, then compute similarity.
        # Or simply use the columns directly if the function supports it. 
        # sklearn's cosine_similarity computes the L2-normalized dot product of vectors.
        # If we input (n_samples_X, n_features), it returns (n_samples_X, n_samples_X).
        # We want (n_products, n_products). So we transpose the user-item matrix.
        product_user_matrix = self.user_item_matrix.T
        similarity_matrix = cosine_similarity(product_user_matrix)
        
        self.item_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=product_user_matrix.index,
            columns=product_user_matrix.index
        )
        
    def recommend(self, user_id, n=10):
        """
        Recommend products for a user based on their past interactions.
        Logic:
        1. Get products the user has liked/interacted with.
        2. Find similar products to those.
        3. Weight by original rating (optional) or just sum similarity scores.
        """
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=['product_id', 'product_name', 'category', 'score'])
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        # Filter to products they actually rated > 0
        user_rated_products = user_ratings[user_ratings > 0]
        
        if user_rated_products.empty:
             return pd.DataFrame(columns=['product_id', 'product_name', 'category', 'score'])
        
        # Calculate scores for candidate products
        # Score = Sum (Similarity(item_i, item_candidate) * Rating(item_i)) / Sum (Similarity) ?
        # Simplified: Sum (Similarity * Rating)
        
        # We'll create a Series to hold the aggregate scores
        scores = pd.Series(dtype='float64')
        
        for product_id, rating in user_rated_products.items():
            # Get similarity scores for this product against all others
            similar_scores = self.item_similarity_df[product_id]
            
            # Weighted by the user's rating
            weighted_scores = similar_scores * rating
            
            # specialized logic: avoid recommending products the user has already seen?
            # Standard CF usually recommends new items.
            
            scores = scores.add(weighted_scores, fill_value=0)
            
        # Remove products already rated by the user
        scores = scores.drop(user_rated_products.index, errors='ignore')
        
        # Get top n
        top_product_ids = scores.sort_values(ascending=False).head(n)
        
        # Format output
        recommendations = pd.DataFrame({'product_id': top_product_ids.index, 'score': top_product_ids.values})
        recommendations = pd.merge(recommendations, self.items, on='product_id')
        
        return recommendations[['product_id', 'product_name', 'category', 'score']]
