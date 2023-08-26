from fastapi import FastAPI
from sentence_transformers import SentenceTransformer,util
from pydantic import BaseModel
import torch
import json
import pickle

app = FastAPI()

class Query(BaseModel):
    question: str

embeddings = pickle.load(open('./processed_data/issue_embeddings.pkl', 'rb'))
resolution = pickle.load(open('./processed_data/resolution.pkl', 'rb'))
issue = pickle.load(open('./processed_data/issues.pkl', 'rb'))
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/query",status_code=200)
async def query(q: Query):
    query_embedding = model.encode(q.question,convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding,embeddings)
    top_results = torch.topk(cosine_scores,k=1)

    if top_results[0][0].item() >= 0.4:
        response = {"Resolution": resolution[top_results[1][0]],"Similar": issue[top_results[1][0]]}
    else:
        response = {"Resolution": "I do not have a resolution"}
    json_response = json.dumps(response)
    return json_response