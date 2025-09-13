import faiss
import pickle
from google import genai
import numpy as np
import time
client = genai.Client(api_key=API_KEY)

# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    docs = f.read().split("\n")


res = client.models.embed_content(
    model="text-embedding-004",
    contents=docs   # pass a list of texts
)
embeddings = [e.values for e in res.embeddings]
dim = len(embeddings[0])

# Store in FAISS
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save index + docs
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
faiss.write_index(index, "index.faiss")

print("Index built and saved ✅")
