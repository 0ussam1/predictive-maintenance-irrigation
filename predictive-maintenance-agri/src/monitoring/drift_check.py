import pandas as pd

def main():
    ref = pd.read_csv("data/processed/train_features.csv")
    current = pd.read_csv("data/processed/test_features.csv")

    ref_mean = ref.mean(numeric_only=True)
    current_mean = current.mean(numeric_only=True)

    drift = (current_mean - ref_mean).abs()

    print("=== Drift simple (diffrence des moyennes) ===")
    print(drift.sort_values(ascending=False))

if __name__ == "__main__":
    main()