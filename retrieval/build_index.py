# retrieval/build_index.py
import faiss, numpy as np
emb = np.load("embeddings.npy").astype('float32')
print("emb shape:", emb.shape)
d = emb.shape[1]
# use inner product on normalized vectors -> equivalent to cosine
index = faiss.IndexFlatIP(d)
# if many vectors, consider IndexIVFFlat + training
index.add(emb)
faiss.write_index(index, "idx.faiss")
print("Index built. n=", index.ntotal)
