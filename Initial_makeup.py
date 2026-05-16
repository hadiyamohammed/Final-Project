import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="makeup.csv"):
    df = pd.read_csv(path)

    # clean price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # clean brand names
    df["brand"] = df["brand"].str.lower().str.strip()

    return df


def plot_brand_prices(df):
    top_brands = df["brand"].value_counts().head(10).index
    df_top = df[df["brand"].isin(top_brands)]

    # sort brands by median price
    brand_order = df_top.groupby("brand")["price"].median().sort_values().index
    data = [df_top[df_top["brand"] == b]["price"] for b in brand_order]

    plt.style.use("default")  # clean white background
    plt.figure(figsize=(11, 5))

    box = plt.boxplot(data, patch_artist=True)

    # soft color
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe2ff")

    plt.title("Price Distribution by Brand", fontsize=14)
    plt.xlabel("Brand")
    plt.ylabel("Price (USD)")
    plt.xticks(range(1, len(brand_order)+1), brand_order, rotation=45)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("initial_figure1_brand_price.png", dpi=300)
    plt.close()

def plot_category_prices(df):
    df_cat = df.dropna(subset=["category"])

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
    plt.savefig("initial_figure2_category_price.png", dpi=300)
    plt.close()



def main():
    df = load_data()

    plot_brand_prices(df)
    plot_category_prices(df)

    print("Done. Figures saved.")


if __name__ == "__main__":
    main()
