import pandas as pd
from sklearn.model_selection import train_test_split

required_columns = ["text", "label", "language"]

eng_df = pd.read_csv("final_cleaned_english_hate_speech.csv")
spa_df = pd.read_csv("final_cleaned_spanish_hate_speech.csv")

eng_df["language"] = "en"
spa_df["language"] = "es"

def check_and_fix_columns(df, dataset_name):
    missing_columns = [col for col in required_columns if col not in df.columns]

    for col in missing_columns:
        df[col] = "unknown" if col == "language" else ""

    return df

eng_df = check_and_fix_columns(eng_df, "English Dataset")
spa_df = check_and_fix_columns(spa_df, "Spanish Dataset")

eng_train, eng_test = train_test_split(eng_df, test_size=0.2, random_state=42, stratify=eng_df['label'])
spa_train, spa_test = train_test_split(spa_df, test_size=0.2, random_state=42, stratify=spa_df['label'])

train_df = pd.concat([eng_train, spa_train], ignore_index=True)
test_df = pd.concat([eng_test, spa_test], ignore_index=True)

train_df = check_and_fix_columns(train_df, "Merged Train Dataset")
test_df = check_and_fix_columns(test_df, "Merged Test Dataset")

eng_train.to_csv("english_train.csv", index=False)
eng_test.to_csv("english_test.csv", index=False)
spa_train.to_csv("spanish_train.csv", index=False)
spa_test.to_csv("spanish_test.csv", index=False)
train_df.to_csv("merged_train.csv", index=False)
test_df.to_csv("merged_test.csv", index=False)

print("\nSplitting and validation completed. Files saved:")
print("- english_train.csv")
print("- english_test.csv")
print("- spanish_train.csv")
print("- spanish_test.csv")
print("- merged_train.csv")
print("- merged_test.csv")
