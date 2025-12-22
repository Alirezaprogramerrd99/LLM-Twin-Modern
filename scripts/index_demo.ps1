$body = @(
  @{ id = "docA"; text = "Transformers use self-attention for context. Chunking improves retrieval." },
  @{ id = "docB"; text = "Qdrant is a vector database for embeddings and similarity search." },
  @{ id = "docC"; text = "Sentence Transformers produce dense semantic vectors." }
) | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/index" -Method Post -Body $body -ContentType "application/json"