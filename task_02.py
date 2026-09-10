from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample Input Documents & Query
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]
query = ["machine learning algorithms for data"]

# 1. Fit CountVectorizer on documents
vectorizer = CountVectorizer()
vectorizer.fit(documents)

# 2. Transform both documents and query into numerical vector arrays
doc_vectors = vectorizer.transform(documents)
query_vector = vectorizer.transform(query)

# 3. Compute pairwise cosine similarity 
similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()

# 4. Display ranked documents from highest score to lowest score
ranked_results = sorted(zip(similarity_scores, documents), reverse=True)

print("Search Query:", query[0])
print("-" * 50)
for score, doc in ranked_results:
    print(f"Score: {score:.4f} | Document: {doc}")