import os
import json
import ollama
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import math

load_dotenv()

pc = Pinecone(api_key="pcsk_4cH8kP_SEu6dcG2q2iPPFW8HwG1famWWy5BYNccqVdK7kXLezc6YAFfW8GTdsLZS96nRH8")

# Ollama Models and Their Dimensions
PRODUCT_EMBED_MODEL = "nomic-embed-text"
PRODUCT_DIM = 768  # nomic-embed-text output size
STORE_EMBED_MODEL = "nomic-embed-text"
STORE_DIM = 768    # using the same model (can change, match dimension!)

# === Ollama Embedding Utility === #

def generate_ollama_embedding(text, model):
    try:
        response = ollama.embeddings(
            model=model,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating Ollama embedding: {e}")
        return None

# === 1. PRODUCT CATALOG INDEX === #

CATALOG_INDEX_NAME = "catalog"

def create_or_get_catalog_index(index_name=CATALOG_INDEX_NAME, dimension=PRODUCT_DIM):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
    return pc.Index(index_name), index_name

catalog_index, catalog_index_name = create_or_get_catalog_index()

def upsert_products(index, index_name, products, namespace):
    vectors = []
    for product in products:
        product_text = f"{product['name']} {product.get('color', 'Not specified')} {' '.join(product.get('keyFeatures', []))}"
        embedding = generate_ollama_embedding(product_text, PRODUCT_EMBED_MODEL)
        if embedding is None:
            print(f"Failed to generate embedding for product: {product['name']}")
            continue
        simplified_metadata = {
            "text": product_text,
            "name": product['name'],
            "price": str(product.get('price', 'Not specified')),
            "color": product.get('color', 'Not specified') if product.get('color') is not None else 'Color not specified',
            "keyFeatures": ', '.join(product.get('keyFeatures', []))
        }
        vectors.append({
            "id": product['productId'],
            "values": embedding,
            "metadata": {k: v for k, v in simplified_metadata.items() if v is not None}
        })
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted {len(vectors)} products into index {index_name} under namespace {namespace}")
    else:
        print("No vectors to upsert.")

# === 2. STORE DESCRIPTION INDEX === #

STORES_INDEX_NAME = "stores-list"

def create_or_get_stores_index(index_name=STORES_INDEX_NAME, dimension=STORE_DIM):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
    return pc.Index(index_name), index_name

stores_index, stores_index_name = create_or_get_stores_index()

def upsert_store_descriptions(index, index_name, stores_data, batch_size=100):
    total_stores = len(stores_data)
    num_batches = math.ceil(total_stores / batch_size)

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_stores)
        batch_stores = stores_data[start_idx:end_idx]

        vectors = []
        for store_info in batch_stores:
            store_name = store_info['store']
            store_description = store_info.get('description', '')
            embedding = generate_ollama_embedding(store_description, STORE_EMBED_MODEL)
            if embedding is None:
                print(f"Failed to generate embedding for store: {store_name}")
                continue
            metadata = {
                "store_name": store_name,
                "description": store_description
            }
            vectors.append({
                "id": f"{store_name}",
                "values": embedding,
                "metadata": metadata
            })

        if vectors:
            index.upsert(vectors=vectors)
            print(f"Upserted batch {batch_num + 1}/{num_batches} with {len(vectors)} store descriptions into index {index_name}")

    print(f"Completed upserting all {total_stores} store descriptions")

# === MAIN EXECUTION === #

# Products upsert routine (catalog.json)
if os.path.exists('catalog.json'):
    with open('catalog.json', 'r') as f:
        data = json.load(f)
        stores_data = data['stores']

    print(f"Total stores found in catalog: {len(stores_data)}")
    for store_info in stores_data:
        store_name = store_info['store']
        products = store_info['products']
        namespace = store_name.lower().replace(' ', '_')
        print(f"Processing store: {store_name} with {len(products)} products")
        upsert_products(catalog_index, catalog_index_name, products, namespace)
else:
    print("catalog.json not found. Skipping product embeddings.")

# Stores list upsert routine (stores_list.json)
if os.path.exists('stores_list.json'):
    with open('stores_list.json', 'r') as f:
        data = json.load(f)
        stores_list_data = data['stores']

    print(f"Total stores found in store list: {len(stores_list_data)}")
    upsert_store_descriptions(stores_index, stores_index_name, stores_list_data, batch_size=2)
else:
    print("stores_list.json not found. Skipping store description embeddings.")