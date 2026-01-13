import sys
import os

print("Verifying imports...")
try:
    from src.data.loader import load_data
    from src.recommenders.rule_based import RuleBasedRecommender
    from src.recommenders.collaborative import CollaborativeRecommender
    from src.recommenders.deep_learning import DeepLearningRecommender
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Verifying Data Loader...")
try:
    if os.path.exists('user_product_interactions.csv') and os.path.exists('products.csv'):
        ratings, items = load_data('.')
        print(f"Data loaded. Ratings shape: {ratings.shape}, Products shape: {items.shape}")
        
        print("Verifying Models Initialization...")
        rb = RuleBasedRecommender(ratings, items)
        print("Rule Based initialized.")
        
        cf = CollaborativeRecommender(ratings, items)
        print("Collaborative Filtering initialized.")
        
        dl = DeepLearningRecommender(ratings, items)
        print("Deep Learning initialized.")
        
        print("Verification Complete. System is ready.")
    else:
        print("Dataset files not found in current directory. Skipping logic verification.")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
