import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

# very unreadable, remove plot.show() to see plots.
# Load the dataset
df = pd.read_csv("data/data.csv")

# Display basic information
print("Dataset Head:")
print(df.head(), "\n")
print("Basic Information:")
print(df.info(), "\n")
print("Dataset Summary Statistics:")
print(df.describe(include="all"), "\n")

# Check for missing values
print("Missing Values:")
print(df.isnull().sum(), "\n")

# Check unique labels
df["label_list"] = df["label"].apply(
    lambda x: x.split(",")
)  # Assuming multilabels are comma-separated
all_labels = [label for labels in df["label_list"] for label in labels]
label_counts = Counter(all_labels)

print("Unique Labels and Their Counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Distribution of labels
plt.figure(figsize=(10, 6))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.title("Distribution of Labels")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.xticks(rotation=45)
# plt.show()

# Number of labels per sentence
df["num_labels"] = df["label_list"].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df["num_labels"], bins=10, kde=False)
plt.title("Distribution of Number of Labels per Sentence")
plt.xlabel("Number of Labels")
plt.ylabel("Count")
# plt.show()

# Sentence length analysis
df["sentence_length"] = df["sentence"].apply(lambda x: len(x.split()))
print("Average Sentence length: " + str(df["sentence_length"].mean()))
plt.figure(figsize=(8, 5))
sns.histplot(df["sentence_length"], bins=20, kde=True)
plt.title("Distribution of Sentence Lengths")
plt.xlabel("Sentence Length")
plt.ylabel("Count")
# plt.show()
# Z - score
df["z_score"] = stats.zscore(df["sentence_length"])
task_no_outliers = df[df["z_score"].abs() <= 3]
df_no_outliers = pd.concat(
    [df[df["label"].apply(lambda x: "Task" not in x)], task_no_outliers]
)
plt.figure(figsize=(8, 5))
sns.histplot(task_no_outliers["sentence_length"], bins=20, kde=True)
plt.title("Distribution of Sentence Lengths without Outliers")
plt.xlabel("Sentence Length")
plt.ylabel("Count")
plt.show()

# Check for duplicate sentences
duplicate_sentences = df[df.duplicated(subset=["sentence"], keep=False)]
print(f"Number of Duplicate Sentences: {len(duplicate_sentences)}")
if len(duplicate_sentences) > 0:
    print("Duplicate Sentences:")
    print(duplicate_sentences, "\n")

# Correlation between labels (co-occurrence matrix)
co_occurrence = pd.DataFrame(0, index=label_counts.keys(), columns=label_counts.keys())
for labels in df["label_list"]:
    for label1 in labels:
        for label2 in labels:
            co_occurrence[label1][label2] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence, annot=False, cmap="Blues")
plt.title("Label Co-occurrence Matrix")
# plt.show()
# Create a dataframe where each row corresponds to a single label
exploded_df = df.explode("label_list")
# Plot the distribution of sentence lengths for each label
plt.figure(figsize=(10, 6))
sns.boxplot(x="label_list", y="sentence_length", data=exploded_df, palette="Set3")
plt.title("Distribution of Sentence Lengths by Label")
plt.xlabel("Labels")
plt.ylabel("Sentence Length")
plt.xticks(rotation=45)
# plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=exploded_df,
    x="sentence_length",
    hue="label_list",
    fill=True,
    common_norm=False,
    palette="Set2",
)
plt.title("Kernel Density Plot of Sentence Lengths by Label")
plt.xlabel("Sentence Length")
plt.ylabel("Density")


# plt.show()
# Filter and remove outliers for specific labels
def remove_outliers_for_label(df, label):
    # Filter sentences for the specified label
    label_df = df[df["label_list"] == label]
    # Calculate Z-scores for sentence lengths
    label_df["z_score"] = stats.zscore(label_df["sentence_length"])
    # Remove outliers (Z-score > 3 or < -3)
    label_no_outliers = label_df[label_df["z_score"].abs() <= 3]
    return label_no_outliers


a_df = exploded_df[exploded_df["label_list"] == "Action"]
s_no_outliers = remove_outliers_for_label(exploded_df, "Situation")
r_no_outliers = remove_outliers_for_label(exploded_df, "Result")
t_no_outliers = remove_outliers_for_label(exploded_df, "Task")
df_no_outliers = pd.concat([s_no_outliers, r_no_outliers, t_no_outliers, a_df])
# Averge sentence length NEW
df_no_outliers["sentence_length"] = df_no_outliers["sentence"].apply(
    lambda x: len(x.split())
)
print("Average Sentence length: " + str(df_no_outliers["sentence_length"].mean()))
# # Filter sentences that have the 'Task' label
#
# task_df = exploded_df[exploded_df["label_list"] == "Task"]
#
# # Calculate Z-scores for sentence lengths in the 'Task' subset
# task_df["z_score"] = stats.zscore(task_df["sentence_length"])
#
# # Remove outliers from 'Task' sentences (Z-score > 3)
# task_no_outliers = task_df[task_df["z_score"].abs() <= 3]
#
# # Combine the 'Task' sentences without outliers with other non-'Task' sentences
# df_no_outliers = pd.concat(
#     [exploded_df[exploded_df["label_list"] != "Task"], task_no_outliers]
# )

# 1. Plot the Distribution of Sentence Lengths by Label (With Outliers)
plt.figure(figsize=(12, 6))
sns.boxplot(x="label_list", y="sentence_length", data=exploded_df, palette="Set3")
plt.title("Distribution of Sentence Lengths by Label (With Outliers)")
plt.xlabel("Labels")
plt.ylabel("Sentence Length")
plt.xticks(rotation=45)
# plt.show()

# 2. Plot the Distribution of Sentence Lengths by Label (Without Outliers for Task Only)
plt.figure(figsize=(12, 6))
sns.boxplot(x="label_list", y="sentence_length", data=df_no_outliers, palette="Set3")
plt.title("Distribution of Sentence Lengths by Label (Without Outliers for Task Only)")
plt.xlabel("Labels")
plt.ylabel("Sentence Length")
plt.xticks(rotation=45)
# plt.show()
# Plot the counter of sentences by label
plt.figure(figsize=(10, 6))
sns.countplot(x="label_list", data=df_no_outliers, palette="Set3")
plt.title("Number of Sentences by Label (Without Outliers)")
plt.xlabel("Labels")
plt.ylabel("Number of Sentences")
plt.xticks(rotation=45)
# plt.show()
print(df_no_outliers["label_list"].value_counts())

# Optional: Kernel Density Plot (Smoothed Distributions) for Sentence Lengths by Label (With Outliers)
plt.figure(figsize=(12, 6))
sns.kdeplot(
    data=exploded_df,
    x="sentence_length",
    hue="label_list",
    fill=True,
    common_norm=False,
    palette="Set2",
)
plt.title("Kernel Density Plot of Sentence Lengths by Label (With Outliers)")
plt.xlabel("Sentence Length")
plt.ylabel("Density")
# plt.show()

# Optional: Kernel Density Plot (Smoothed Distributions) for Sentence Lengths by Label (Without Outliers for Task Only)
plt.figure(figsize=(12, 6))
sns.kdeplot(
    data=df_no_outliers,
    x="sentence_length",
    hue="label_list",
    fill=True,
    common_norm=False,
    palette="Set2",
)
plt.title(
    "Kernel Density Plot of Sentence Lengths by Label (Without Outliers for Task Only)"
)
plt.xlabel("Sentence Length")
plt.ylabel("Density")
# plt.show()
# export exploded_df and df_no_outliers to csv files
# exploded_df.to_csv("exploded_df.csv", index=False)
# df_no_outliers.to_csv("df_no_outliers.csv", index=False)
exported_df_no_outliers = df_no_outliers[["sentence", "label"]]
exported_df_no_outliers.to_csv("data/exported_df_no_outliers.csv", index=False)
