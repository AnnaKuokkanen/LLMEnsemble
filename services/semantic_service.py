from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_similarity_scores(prev_queries, curr_query):
  # append the current query and compute results
  prev_queries.append(curr_query)

  embeddings = model.encode(prev_queries)
  similarities = model.similarity(embeddings, embeddings)

  # return last column which corresponds to the new query's similarity to all previous queries
  last_column = similarities[:, -1].tolist()

  return last_column

def get_embedding(sentence):
  return model.encode(sentence)