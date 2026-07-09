from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

embeddings = model.encode(input_texts, normalize_embeddings=True)

# similarity scores
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())