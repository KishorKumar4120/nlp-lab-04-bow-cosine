import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Sample Input Text
corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

# 1. Instantiate CountVectorizer(stop_words='english') and fit-transform the corpus
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# 2. Extract the vocabulary list
vocab = vectorizer.get_feature_names_out()

# 3. Convert the transformed sparse matrix into a Pandas DataFrame
df_bow = pd.DataFrame(X.toarray(), columns=vocab)
print(df_bow)