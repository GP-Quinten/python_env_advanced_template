import os
from dotenv import load_dotenv
import weaviate
import json

# Load environment variables from .env file
load_dotenv()

# Read environment variables
http_host = os.getenv("WEAVIATE__HTTP_HOST")
http_port = int(os.getenv("WEAVIATE__HTTP_PORT"))
grpc_host = os.getenv("WEAVIATE__GRPC_HOST")
grpc_port = int(os.getenv("WEAVIATE__GRPC_PORT"))
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Connect to Weaviate using both HTTP and gRPC (connect_to_custom requires grpc args)
client = weaviate.connect_to_custom(
    http_host=http_host,
    http_port=http_port,
    http_secure=False,  # Set to True if your instance uses HTTPS
    grpc_host=grpc_host,
    grpc_port=grpc_port,
    grpc_secure=False,  # Set to True if your gRPC endpoint uses TLS
    # No additional parameters provided
    headers={
        "X-Mistral-Api-Key": mistral_api_key
    }

)

# Check connection
if client.is_ready():
    # Get meta information
    meta_info = client.get_meta()
    print(json.dumps(meta_info, indent=2))
    # Print enabled modules
    print("\nEnabled modules:")
    for module, details in meta_info.get("modules", {}).items():
        print(f"- {module}: {details.get('name', '')}")
    # Print default vectorizer if available
    default_vectorizer = meta_info.get("defaultVectorizerModule")
    if default_vectorizer:
        print(f"\nDefault vectorizer module: {default_vectorizer}")
    else:
        print("\nDefault vectorizer module not specified in meta info.")
else:
    print("Weaviate connection failed.")

# Import helper types if available in the installed weaviate client
from weaviate.classes.config import Property, DataType, Configure

# print all collections
print("\nExisting collections:")
for col in client.collections.list_all():
    print(f"- {col}")

print("\n--- Creating 'Article' collection ---")

# Check if 'Article' collection exists and delete if so
if "Article" in [col for col in client.collections.list_all()]:
    print("'Article' collection already exists. Deleting it first...")
    client.collections.delete("Article")
    print("Existing 'Article' collection deleted.")
# 2. Create a collection
client.collections.create(
    name="Article",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="body", data_type=DataType.TEXT),
    ],
    vector_config=Configure.Vectors.text2vec_mistral(),  # Requires OpenAI API key in headers if using OpenAI
)
print("Collection 'Article' created.")

# 3. Insert a couple of example objects
articles = client.collections.get("Article")
objects_to_insert = [
    {"title": "Weaviate Introduction", "body": "Weaviate is an open-source vector database."},
    {"title": "Vector Search", "body": "Semantic search uses vector embeddings for retrieval."}
]
print("\n--- Inserting objects ---")
print(json.dumps(objects_to_insert, indent=2))
articles.data.insert_many(objects_to_insert)
print("Objects inserted.")

# 4. Run a simple semantic search
search_query = "What is Weaviate?"
print(f"\n--- Running semantic search for: '{search_query}' ---")
response = articles.query.near_text(
    query=search_query,
    limit=2
)
print("Search response objects:")
for obj in getattr(response, "objects", []):
    # Some client versions return different shapes; attempt to print properties safely
    if hasattr(obj, "properties"):
        print(json.dumps(obj.properties, indent=2))
    else:
        print(json.dumps(obj, indent=2))

# 5. Cleanup: delete the collection
print("\n--- Deleting 'Article' collection ---")
client.collections.delete("Article")
print("Collection 'Article' deleted.")


client.close()