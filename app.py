import faiss
import pickle
import numpy as np
from google import genai

client = genai.Client(api_key="AIzaSyB-DocK8hcBqqPlrd4y-WCd-ooG0n0oYb8")


with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
index = faiss.read_index("index.faiss")

def embed_text(text):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(res.embeddings[0].values, dtype="float32")

def retrieve_context(query, top_k=3):
    q_vec = embed_text(query).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    return [docs[i] for i in I[0]]

def ask_about_me(query):
    context = "\n".join(retrieve_context(query))
    prompt = f"""
    You are my personal assistant. Answer only using the context below:

    {context}

    Question: {query}
    
    if he asks questions which does not comes under the context then say Adhithya didn't mention this details
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    while True:
        q = input("Ask about me: ")
        if q.lower() == "q":
            break
        print("Answer:", ask_about_me(q))
