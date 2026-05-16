import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_and_clean_data(path="makeup.csv"):
    df = pd.read_csv(path)

    # Clean price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Clean brand names and categories
    df["brand"] = df["brand"].str.lower().str.strip()
    df["category"] = df["category"].fillna("unknown_category").str.lower().str.strip()
    
    # Feature Engineering: Description length (Proxy for brand investment)
    df["desc_length"] = df["description"].fillna("").astype(str).apply(len)

    return df

# MODULE 03: Similarity, Dimensionality Reduction, and Cleaning
def feature_engineering(df):
    """Creates price tiers and encodes categorical variables."""
    # Create Price Tiers (Target Variable)
    def categorize_tier(price):
        if price < 15:
            return "Drugstore"
        elif price <= 25:
            return "Mid-range"
        else:
            return "Luxury"
            
    df["price_tier"] = df["price"].apply(categorize_tier)
    
    # Encode categorical features for modeling
    le_brand = LabelEncoder()
    le_category = LabelEncoder()
    
    df["brand_encoded"] = le_brand.fit_transform(df["brand"])
    df["category_encoded"] = le_category.fit_transform(df["category"])
    
    return df, le_brand, le_category

# MODULE 06 & 07: Supervised Machine Learning & Evaluation
def train_and_evaluate_model(df):
    """Trains a classifier to predict price tier and evaluates it."""
    # Define features (X) and target (y)
    X = df[["brand_encoded", "category_encoded", "desc_length"]]
    y = df["price_tier"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the Supervised ML Model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # MODULE 07: Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance Plot
    importances = model.feature_importances_
    features = ["Brand", "Category", "Description Length"]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.title("Feature Importance in Predicting Price Tier")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("final_figure3_feature_importance.png", dpi=300)
    plt.close()
    
    return model

def plot_brand_prices(df):
    top_brands = df["brand"].value_counts().head(10).index
    df_top = df[df["brand"].isin(top_brands)]

    brand_order = df_top.groupby("brand")["price"].median().sort_values().index
    data = [df_top[df_top["brand"] == b]["price"] for b in brand_order]

    plt.style.use("default")
    plt.figure(figsize=(11, 5))
    box = plt.boxplot(data, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe2ff")

    plt.title("Price Distribution by Brand", fontsize=14)
    plt.xlabel("Brand")
    plt.ylabel("Price (USD)")
    plt.xticks(range(1, len(brand_order)+1), brand_order, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("final_figure1_brand_price.png", dpi=300)
    plt.close()

def plot_category_prices(df):
    df_cat = df[df["category"] != "unknown_category"]

    cat_order = df_cat.groupby("category")["price"].median().sort_values().index
    data = [df_cat[df_cat["category"] == c]["price"] for c in cat_order]

    plt.style.use("default")
    plt.figure(figsize=(11, 5))
    box = plt.boxplot(data, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#d4edda")

    plt.title("Price Distribution by Category", fontsize=14)
    plt.xlabel("Category")
    plt.ylabel("Price (USD)")
    plt.xticks(range(1, len(cat_order)+1), cat_order, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("final_figure2_category_price.png", dpi=300)
    plt.close()

def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    
    print("Generating EDA plots...")
    plot_brand_prices(df)
    plot_category_prices(df)
    
    print("Applying Module 03 (Feature Engineering)...")
    df_engineered, _, _ = feature_engineering(df)
    
    print("Applying Module 06 & 07 (ML Training and Evaluation)...")
    train_and_evaluate_model(df_engineered)
    
    print("Done. Figures saved.")

if __name__ == "__main__":
    main()
