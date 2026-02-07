import chromadb

client = chromadb.HttpClient(host="127.0.0.1", port=8000)
print("Collections found:", client.list_collections())
