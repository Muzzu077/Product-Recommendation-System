import pandas as pd
import random
import os

def transform_dataset():
    print("Transforming dataset...")
    
    # 1. Load Original Data
    # u.data: user_id | item_id | rating | timestamp
    ratings = pd.read_csv(
        "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # u.item: item_id | title | ... (genres)
    # We only care about ID to map it.
    items = pd.read_csv(
        "u.item",
        sep="|",
        encoding="latin-1",
        names=["item_id", "title"] + list(range(22)),
        usecols=[0, 1]
    )

    # 2. Rename Columns (Movie -> Product)
    ratings.rename(columns={"item_id": "product_id"}, inplace=True)
    items.rename(columns={"item_id": "product_id"}, inplace=True)

    # 3. Remove Movie Titles (Keep only product_id initially)
    # The user wants to remove movie titles completely.
    products = items[["product_id"]].copy()

    # 4. Generate Synthetic Product Names
    products["product_name"] = products["product_id"].apply(lambda x: f"Product #{x}")

    # 5. Add Product Categories
    categories = [
        "Electronics",
        "Fashion",
        "Home Appliances",
        "Books",
        "Sports",
        "Beauty",
        "Toys"
    ]
    
    # Fix random seed for reproducibility
    random.seed(42)
    
    # Assign category based on ID simple hash to be deterministic
    products["category"] = products["product_id"].apply(
        lambda x: categories[x % len(categories)]
    )

    # 6. Save Final Dataset
    products.to_csv("products.csv", index=False)
    ratings.to_csv("user_product_interactions.csv", index=False)

    print("Transformation Complete.")
    print("Created 'products.csv' and 'user_product_interactions.csv'")
    print(products.head())
    print(ratings.head())

if __name__ == "__main__":
    if os.path.exists("u.data") and os.path.exists("u.item"):
        transform_dataset()
    else:
        print("Error: u.data or u.item not found in current directory.")
