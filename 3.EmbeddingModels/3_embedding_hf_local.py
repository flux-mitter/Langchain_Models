# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # text = "The capital of India is New Delhi."
# documents = [
#     "Delhi is the capital of India.",
#     "Kolkata is the capital of West Bengal.",
#     "Paris is the capital of France.",
# ]

# vector = embeddings.embed_query(documents)
# print(str(vector)  )


from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France.",
]

vectors = embeddings.embed_documents(documents)
print(str(vectors[0])  )

# for document, vector in zip(documents, vectors):
#     print(f"\nDocument: {document}")
#     print(f"Vector dimensions: {len(vector)}")
#     print(f"First 5 values: {vector[:5]}")

