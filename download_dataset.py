import kagglehub
import os
import shutil

def download_and_setup():
    print("Downloading dataset...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("saurav9786/amazon-product-reviews")
        print("Path to dataset files:", path)
        
        # List files in the downloaded path
        files = os.listdir(path)
        print("Files found:", files)
        
        # We expect a CSV file, likely 'ratings_Electronics.csv' or similar.
        # Let's try to identify it and copy it to our project dir
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            source_file = os.path.join(path, csv_files[0])
            destination = os.path.join(os.getcwd(), 'amazon_reviews.csv')
            shutil.copy(source_file, destination)
            print(f"Copied {source_file} to {destination}")
        else:
            print("No CSV file found in downloaded dataset.")
            
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    download_and_setup()
