import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

class DeepLearningRecommender:
    """
    Implements a Neural Collaborative Filtering (NCF) style recommender.
    Uses Embeddings for Users and Products.
    """
    
    def __init__(self, ratings_df, items_df):
        self.ratings = ratings_df
        self.items = items_df
        
        # User and Item encoding
        self.user_ids = ratings_df['user_id'].unique()
        self.product_ids = ratings_df['product_id'].unique()
        
        # Map IDs to continuous indices (0 to N-1)
        self.user2idx = {o: i for i, o in enumerate(self.user_ids)}
        self.product2idx = {o: i for i, o in enumerate(self.product_ids)}
        
        self.idx2user = {i: o for o, i in self.user2idx.items()}
        self.idx2product = {i: o for o, i in self.product2idx.items()}
        
        self.num_users = len(self.user_ids)
        self.num_products = len(self.product_ids)
        
        self.model = self._build_model()
        
    def _build_model(self, embedding_size=50):
        # inputs
        user_input = Input(shape=(1,), name='user_input')
        product_input = Input(shape=(1,), name='product_input')
        
        # Embeddings
        user_embedding = Embedding(input_dim=self.num_users, output_dim=embedding_size, name='user_embedding')(user_input)
        product_embedding = Embedding(input_dim=self.num_products, output_dim=embedding_size, name='product_embedding')(product_input)
        
        # Flatten
        user_vec = Flatten()(user_embedding)
        product_vec = Flatten()(product_embedding)
        
        # Concatenate or Dot Product
        # We'll use Concatenation + MLP for "Deep Learning" approach
        concat = Concatenate()([user_vec, product_vec])
        
        # Dense Layers
        x = Dense(128, activation='relu')(concat)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Output
        output = Dense(1, activation='linear')(x) # Predicting Rating
        
        model = Model(inputs=[user_input, product_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
        
    def train(self, epochs=5, batch_size=64):
        # Prepare data
        user_indices = self.ratings['user_id'].map(self.user2idx).values
        product_indices = self.ratings['product_id'].map(self.product2idx).values
        y = self.ratings['rating'].values
        
        self.model.fit(
            [user_indices, product_indices],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
    def recommend(self, user_id, n=10):
        if user_id not in self.user2idx:
            # New user (Cold start) -> fallback to rule-based or empty
             return pd.DataFrame(columns=['product_id', 'product_name', 'category', 'score']) # Helper handle upstream or return popular
        
        user_idx = self.user2idx[user_id]
        
        # Candidate generation: Predict for ALL products
        # Logic: Predict score for every product for this user, sort descending.
        
        all_product_indices = np.array(range(self.num_products))
        user_indices = np.array([user_idx] * self.num_products)
        
        predictions = self.model.predict([user_indices, all_product_indices], batch_size=256, verbose=0)
        predictions = predictions.flatten()
        
        # Create DataFrame
        results = pd.DataFrame({
            'product_idx': all_product_indices,
            'score': predictions
        })
        
        # Sort
        results = results.sort_values(by='score', ascending=False)
        
        # Map back to original IDs
        results['product_id'] = results['product_idx'].map(self.idx2product)
        
        # Filter out products already rated?
        # Typically yes, but for "Top Recommendations" often we leave them or filter. 
        # Let's filter out known interactions to suggest NEW things.
        known_products = self.ratings[self.ratings['user_id'] == user_id]['product_id']
        results = results[~results['product_id'].isin(known_products)]
        
        top_n = results.head(n)
        top_n = pd.merge(top_n, self.items, on='product_id')
        
        return top_n[['product_id', 'product_name', 'category', 'score']]
