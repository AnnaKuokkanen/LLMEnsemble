from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def find_k_nearest_neighbors(query, corpus, k):
  # encode the textual query and corpus
  query_embeddings = get_embedding(query)
  corpus_embeddings = get_embedding(corpus)
  
  return semantic_search(query_embeddings, corpus_embeddings, top_k=k)

def get_similarity_scores(prev_queries, curr_query):
  # append the current query and compute results
  prev_queries.append(curr_query)

  embeddings = model.encode(prev_queries)
  similarities = model.similarity(embeddings, embeddings)

  # return last column which corresponds to the new query's similarity to all previous queries
  last_column = similarities[:, -1].tolist()

  return last_column

def get_sentence_similarity(sentence_1, sentence_2):
  embedding_1 = model.encode(sentence_1)
  embedding_2 = model.encode(sentence_2)

  similarity = model.similarity(embedding_1, embedding_2)

  return similarity.item()

def get_embedding(text):
  return model.encode(text)

def get_text(embedding):
  return model.decode(embedding)