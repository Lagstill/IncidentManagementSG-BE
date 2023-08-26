import pandas as pd
import numpy as np
 
from sentence_transformers import SentenceTransformer, util
import pickle

data = pd.read_csv('./data/data.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(data):
    embeddings = model.encode(data, convert_to_tensor=True)
    return embeddings

issues = data['Issue '].tolist()
resolution = data['Resolution'].tolist()

issue_embeddings = get_embeddings(issues)

pickle.dump(issue_embeddings, open('./processed_data/issue_embeddings.pkl', 'wb'))
pickle.dump(resolution, open('./processed_data/resolution.pkl', 'wb'))
pickle.dump(issues, open('./processed_data/issues.pkl', 'wb'))


