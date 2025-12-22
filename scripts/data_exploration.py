# genimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Display settings
pd.set_option("display.max_colwidth", 200)
sns.set_style("whitegrid")


df = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Arabic_Depression_15.000_Tweets_Annotated.xlsx")

# First look
df.head()

# Shape
print("Dataset shape:", df.shape)

# Column names & types
df.info()

# Check missing values
df.isnull().sum()

df['label'].value_counts()

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.xlabel("Label (0 = Non-depressed, 1 = Depressed)")
plt.ylabel("Number of Tweets")
plt.show()

df['text_length'] = df['tweet'].astype(str).apply(len)
df['word_count'] = df['tweet'].astype(str).apply(lambda x: len(x.split()))

df[['text_length', 'word_count']].describe()

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
plt.title("Tweet Length Distribution")

plt.subplot(1,2,2)
sns.histplot(data=df, x='word_count', hue='label', bins=50, kde=True)
plt.title("Word Count Distribution")

plt.tight_layout()
plt.show()

from collections import Counter

all_words = " ".join(df['tweet'].astype(str)).split()
word_freq = Counter(all_words)

word_freq.most_common(20)

depressed_words = " ".join(df[df['label'] == 1]['tweet'].astype(str)).split()
normal_words = " ".join(df[df['label'] == 0]['tweet'].astype(str)).split()

Counter(depressed_words).most_common(20)

Counter(normal_words).most_common(20)

# Arabic letters percentage
df['arabic_ratio'] = df['tweet'].apply(
    lambda x: sum('\u0600' <= c <= '\u06FF' for c in str(x)) / max(len(str(x)), 1)
)

df['arabic_ratio'].describe()

sns.boxplot(x='label', y='arabic_ratio', data=df)
plt.title("Arabic Character Ratio by Class")
plt.show()

print("Duplicate tweets:", df.duplicated(subset='tweet').sum())
df = df.drop_duplicates(subset='tweet')

print("Depressed Tweets Samples:\n")
df[df['label'] == 1]['tweet'].sample(5, random_state=42)
print("Non-Depressed Tweets Samples:\n")
df[df['label'] == 0]['tweet'].sample(5, random_state=42)

df.to_csv("arabic_depression_tweets_eda.csv", index=False)

erer un json file under data contenant les resultatas de lexploration et charger les graphes de l'exploration
