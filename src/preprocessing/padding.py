def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# 1️ Split data
x_train, x_test, y_train, y_test = train_test_split(
    df["text"],    # features
    df["label"],   # correct label column
    test_size=0.2,
    random_state=42
)
# x_train=merged["text"].where(merged["split"] == "train"))
# x_test=merged.where(merged["split"] == "test").drop(["source","split"], axis=1)
# y_train=merged.where(merged["split"] == "train").drop(["source","split"], axis=1)


# 2️ Tokenize
final_list_train, encoded_train, final_list_test, encoded_test, vocab =  tokenize_arabic(
    x_train, y_train, x_test, y_test
)

# 3️ Pad sequences
x_train_pad = padding_(final_list_train, seq_len=50)
x_test_pad  = padding_(final_list_test, seq_len=50)

# 4️ Labels
y_train = encoded_train
y_test  = encoded_test

# Print the first 5 sequences
print("First 5 padded sequences:")
for i in range(5):
    print(x_train_pad[i])

# Print the shape of the padded training set
print("x_train_pad shape:", x_train_pad.shape)
