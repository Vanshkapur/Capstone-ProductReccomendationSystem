import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
import ollama
from typing import Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# ─── Load configuration ──────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT   = os.getenv("PINECONE_ENVIRONMENT")
STORE_INDEX_NAME       = os.getenv("STORE_INDEX_NAME",   "stores-list")
PRODUCT_INDEX_NAME     = os.getenv("PRODUCT_INDEX_NAME", "catalog")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL",       "llama2")

# Global variables for indexes
pc: Optional[Pinecone] = None
index_stores = None
index_products = None

# ─── Initialize Pinecone client & indexes ───────────────────────────────────
def initialize_pinecone():
    global pc, index_stores, index_products
    
    print(f"Connecting to Pinecone with API key: {PINECONE_API_KEY[:10]}..." if PINECONE_API_KEY else "No API key found")

    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return False

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client created successfully")
        
        # Check if indexes exist
        existing_indexes = pc.list_indexes().names()
        print(f"Available indexes: {existing_indexes}")
        
        if STORE_INDEX_NAME in existing_indexes:
            index_stores = pc.Index(STORE_INDEX_NAME)
            print(f"✅ Connected to store index: {STORE_INDEX_NAME}")
        else:
            print(f"❌ Store index '{STORE_INDEX_NAME}' not found")
            
        if PRODUCT_INDEX_NAME in existing_indexes:
            index_products = pc.Index(PRODUCT_INDEX_NAME)
            print(f"✅ Connected to product index: {PRODUCT_INDEX_NAME}")
        else:
            print(f"❌ Product index '{PRODUCT_INDEX_NAME}' not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return False

# Initialize on startup
initialize_pinecone()

# ─── FastAPI app & schema ───────────────────────────────────────────────────
app = FastAPI(
    title="Store & Product Recommender",
    description="Query Pinecone & Ollama to get personalized store & product picks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",    # Alternative Vite port
        "http://127.0.0.1:5173",
        "http://localhost:4173",    # Vite preview port
        "http://127.0.0.1:4173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    profile: str
    top_k_stores: int = 5
    top_k_products: int = 3

class RecommendResponse(BaseModel):
    stores_prompt: str
    product_prompts: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    pinecone_connected: bool
    ollama_connected: bool
    store_index_available: bool
    product_index_available: bool

# ─── Helper functions ───────────────────────────────────────────────────────
def generate_embedding(text: str) -> list[float]:
    """Return a 768‑dim embedding via Ollama."""
    try:
        print(f"Generating embedding for: {text[:50]}...")
        
        # Check if Ollama is available
        try:
            resp = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        except Exception as ollama_error:
            print(f"Ollama connection error: {ollama_error}")
            raise HTTPException(
                status_code=503, 
                detail=f"Ollama service unavailable. Make sure Ollama is running and model '{OLLAMA_EMBEDDING_MODEL}' is available."
            )
        
        embedding = resp.get("embedding")
        if not embedding:
            raise HTTPException(status_code=500, detail="No embedding returned from Ollama")
            
        print(f"✅ Embedding generated, length: {len(embedding)}")
        return embedding
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def extract_store_numbers(store_prompt: str) -> list[str]:
    """Extract store numbers from LLM response."""
    # Try multiple patterns to extract store numbers
    patterns = [
        r"(\d+)\)",  # "1) Store Name"
        r"^(\d+)\.",  # "1. Store Name"
        r"Store\s+(\d+)",  # "Store 1"
        r"#(\d+)",  # "#1 Store"
    ]
    
    numbers = []
    for pattern in patterns:
        found = re.findall(pattern, store_prompt, re.MULTILINE)
        if found:
            numbers.extend(found)
            break
    
    # Remove duplicates while preserving order
    unique_numbers = list(dict.fromkeys(numbers))
    print(f"Extracted store numbers: {unique_numbers}")
    return unique_numbers

def get_store_recommendations(profile: str, top_k: int = 5) -> tuple[str, dict[str,str]]:
    """Get store recommendations based on user profile."""
    try:
        if index_stores is None:
            raise HTTPException(status_code=503, detail="Store index not available. Please check Pinecone connection.")
            
        print(f"Getting store recommendations for profile: {profile[:50]}...")
        emb = generate_embedding(profile)
        
        print("Querying store index...")
        resp = index_stores.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = resp.get('matches', [])
        print(f"Store query response: {len(matches)} matches")
        
        if not matches:
            raise HTTPException(status_code=404, detail="No stores found matching the profile")
        
        store_map = {}
        lines = []
        for i, m in enumerate(matches, start=1):
            metadata = m.get('metadata', {})
            
            # Try different possible field names for store name
            name = (metadata.get("store") or 
                   metadata.get("store_name") or 
                   metadata.get("name") or 
                   f"Store_{i}")
                
            score = m.get("score", 0.0)
            lines.append(f"{i}) {name} (score: {score:.3f})")
            store_map[str(i)] = name
            
            print(f"Store {i}: {name} (score: {score:.3f})")

        prompt = (
            f"Given user profile: {profile}\n\n"
            f"Here are {len(lines)} stores ranked by relevance:\n" + 
            "\n".join(lines) + 
            "\n\nPlease rank these stores in order of best fit for this user profile. "
            "Provide a brief explanation for your top 3 choices."
        )
        
        print("Sending prompt to Ollama LLM...")
        try:
            response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
                {"role": "user", "content": prompt}
            ])
            ranking = response["message"]["content"]
        except Exception as ollama_error:
            print(f"Ollama LLM error: {ollama_error}")
            raise HTTPException(
                status_code=503,
                detail=f"Ollama LLM service unavailable. Make sure model '{OLLAMA_LLM_MODEL}' is available."
            )
        
        print(f"✅ Store ranking generated")
        return ranking, store_map
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Store recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Store recommendation failed: {str(e)}")

def get_product_recommendations(profile: str, namespace: str, top_k: int = 3) -> str:
    """Get product recommendations for a specific store namespace."""
    try:
        if index_products is None:
            return f"Product index not available for {namespace}"
            
        print(f"Getting product recommendations for namespace: {namespace}")
        emb = generate_embedding(profile)
        
        # Query with namespace - handle potential namespace issues
        try:
            resp = index_products.query(
                vector=emb,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace.lower()  # Ensure lowercase for consistency
            )
        except Exception as query_error:
            print(f"Namespace query failed for '{namespace}': {query_error}")
            # Try without namespace as fallback
            resp = index_products.query(
                vector=emb,
                top_k=top_k,
                include_metadata=True
            )
        
        matches = resp.get('matches', [])
        print(f"Product query response for {namespace}: {len(matches)} matches")
        
        if not matches:
            return f"No products found for {namespace}"
        
        lines = []
        for i, m in enumerate(matches, start=1):
            metadata = m.get('metadata', {})
            name = metadata.get("name", f"Product_{i}")
            desc = metadata.get("text", metadata.get("description", ""))
            price = metadata.get("price", "")
            
            if price:
                lines.append(f"{i}) {name} - {desc} (${price})")
            else:
                lines.append(f"{i}) {name} - {desc}")

        if not lines:
            return f"No product details available for {namespace}"

        prompt = (
            f"User profile: {profile}\n\n"
            f"Here are products from {namespace}:\n" + 
            "\n".join(lines) + 
            f"\n\nBased on the user profile, recommend the top {min(3, len(lines))} products "
            "with a brief reason for each choice."
        )
        
        try:
            response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
                {"role": "user", "content": prompt}
            ])
            result = response["message"]["content"]
        except Exception as ollama_error:
            print(f"Ollama LLM error for products: {ollama_error}")
            return f"Failed to generate product recommendations for {namespace}: LLM service unavailable"
        
        print(f"✅ Product recommendations generated for {namespace}")
        return result
        
    except Exception as e:
        print(f"❌ Product recommendation error for {namespace}: {e}")
        return f"Failed to get product recommendations for {namespace}: {str(e)}"

# ─── API endpoints ──────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Main recommendation endpoint."""
    if not req.profile.strip():
        raise HTTPException(status_code=400, detail="Profile field is required and cannot be empty.")

    try:
        print(f"\n=== New recommendation request ===")
        print(f"Profile: {req.profile}")
        print(f"Top K Stores: {req.top_k_stores}")
        print(f"Top K Products: {req.top_k_products}")
        
        # Get store recommendations
        store_ranking, store_map = get_store_recommendations(req.profile, req.top_k_stores)
        
        # Extract store numbers and get top stores for product recommendations
        nums = extract_store_numbers(store_ranking)
        chosen = [store_map[n] for n in nums[:2] if n in store_map]
        
        # If no stores extracted from ranking, use first 2 from store_map
        if not chosen:
            chosen = list(store_map.values())[:2]
        
        print(f"Chosen stores for product recommendations: {chosen}")

        # Get product recommendations for each chosen store
        prod_results = {}
        for store_name in chosen:
            try:
                # Convert store name to namespace format (lowercase, no spaces)
                namespace = store_name.lower().replace(' ', '').replace("'", "")
                prod_results[store_name] = get_product_recommendations(
                    req.profile, namespace, req.top_k_products
                )
            except Exception as e:
                print(f"Failed to get products for {store_name}: {e}")
                prod_results[store_name] = f"Products currently unavailable for {store_name}"

        print("✅ Recommendation completed successfully")
        return RecommendResponse(
            stores_prompt=store_ranking,
            product_prompts=prod_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Recommendation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Comprehensive health check endpoint."""
    # Test Ollama connection
    ollama_connected = False
    try:
        ollama.chat(model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": "test"}])
        ollama_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if (pc and ollama_connected) else "degraded",
        pinecone_connected=pc is not None,
        ollama_connected=ollama_connected,
        store_index_available=index_stores is not None,
        product_index_available=index_products is not None
    )

@app.get("/test-ollama")
def test_ollama():
    """Test Ollama connectivity and models."""
    try:
        # Test embedding
        test_embedding = generate_embedding("test text")
        
        # Test chat
        response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
            {"role": "user", "content": "Say hello"}
        ])
        
        return {
            "status": "success",
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
            "embedding_length": len(test_embedding) if test_embedding else 0,
            "llm_model": OLLAMA_LLM_MODEL,
            "chat_response": response["message"]["content"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama test failed: {str(e)}")

@app.get("/debug-indexes")
def debug_indexes():
    """Debug endpoint to check Pinecone indexes and configuration."""
    try:
        indexes = pc.list_indexes().names() if pc else []
        
        store_stats = None
        product_stats = None
        
        if index_stores:
            try:
                store_stats = index_stores.describe_index_stats()
            except Exception as e:
                store_stats = f"Error: {str(e)}"
                
        if index_products:
            try:
                product_stats = index_products.describe_index_stats()
            except Exception as e:
                product_stats = f"Error: {str(e)}"
        
        return {
            "status": "success",
            "pinecone_connected": pc is not None,
            "available_indexes": indexes,
            "store_index_stats": store_stats,
            "product_index_stats": product_stats,
            "config": {
                "store_index_name": STORE_INDEX_NAME,
                "product_index_name": PRODUCT_INDEX_NAME,
                "embedding_model": OLLAMA_EMBEDDING_MODEL,
                "llm_model": OLLAMA_LLM_MODEL,
                "api_key_configured": bool(PINECONE_API_KEY)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Store & Product Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "POST /recommend - Get store and product recommendations",
            "health": "GET /health - Health check",
            "test_ollama": "GET /test-ollama - Test Ollama connectivity",
            "debug": "GET /debug-indexes - Debug Pinecone indexes",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

# ─── Startup event ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("🚀 Starting Store & Product Recommender API...")
    print(f"📊 Pinecone connected: {pc is not None}")
    print(f"🏪 Store index available: {index_stores is not None}")
    print(f"🛍️ Product index available: {index_products is not None}")
    print(f"🤖 Ollama embedding model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"💬 Ollama LLM model: {OLLAMA_LLM_MODEL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
# ─── Run with:
# 1) Activate your venv:
#    PS> .\myenv\Scripts\Activate.ps1
# 2) Install dependencies (inside your venv):
#    (myenv) PS> pip install fastapi uvicorn python-dotenv
# 3) Start the server:
#    (myenv) PS> python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
# 4) In your browser go to http://localhost:8000 (or 127.0.0.1:8000)
#    Interactive docs available at http://localhost:8000/docs
# 5) To stop the server, press CTRL+C in this terminal






# # ----------------------
# import os
# import re
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from pinecone import Pinecone
# import ollama

# # ─── Load configuration ──────────────────────────────────────────────────────
# load_dotenv()  # load .env from project root

# PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT   = os.getenv("PINECONE_ENVIRONMENT")
# STORE_INDEX_NAME       = os.getenv("STORE_INDEX_NAME",   "stores-list")
# PRODUCT_INDEX_NAME     = os.getenv("PRODUCT_INDEX_NAME", "catalog")
# OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
# OLLAMA_LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL",       "llama2")

# # ─── Initialize Pinecone client & indexes ───────────────────────────────────
# pc = Pinecone(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENVIRONMENT
# )
# index_stores   = pc.Index(STORE_INDEX_NAME)
# index_products = pc.Index(PRODUCT_INDEX_NAME)

# # ─── FastAPI app & schema ───────────────────────────────────────────────────
# app = FastAPI(
#     title="Store & Product Recommender",
#     description="Query Pinecone & Ollama to get personalized store & product picks",
# )

# class RecommendRequest(BaseModel):
#     profile: str
#     top_k_stores: int = 5
#     top_k_products: int = 3

# class RecommendResponse(BaseModel):
#     stores_prompt: str
#     product_prompts: dict[str, str]

# # ─── Helper functions ───────────────────────────────────────────────────────
# def generate_embedding(text: str) -> list[float]:
#     """Return a 768‑dim embedding via Ollama."""
#     try:
#         # Use the correct Ollama embeddings API call
#         resp = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
#         return resp.get("embedding")
#     except Exception as e:
#         print(f"Embedding error: {e}")
#         raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


# def extract_store_numbers(store_prompt: str) -> list[str]:
#     return re.findall(r"(\d+)\)", store_prompt)


# def get_store_recommendations(profile: str, top_k: int = 5) -> tuple[str, dict[str,str]]:
#     try:
#         emb = generate_embedding(profile)
#         resp = index_stores.query(
#             vector=emb,
#             top_k=top_k,
#             include_metadata=True
#         )
#         store_map = {}
#         lines = []
#         for i, m in enumerate(resp["matches"], start=1):
#             name = m["metadata"]["store"]
#             score = m["score"]
#             lines.append(f"{i}) {name} (score: {score:.3f})")
#             store_map[str(i)] = name

#         prompt = (
#             f"Given user profile:\n{profile}\n\n"
#             "Rank these stores in order of best fit:\n" + "\n".join(lines)
#         )
        
#         # Fixed Ollama chat API call
#         response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
#             {"role": "user", "content": prompt}
#         ])
#         ranking = response["message"]["content"]
#         return ranking, store_map
#     except Exception as e:
#         print(f"Store recommendation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Store recommendation failed: {e}")


# def get_product_recommendations(profile: str, namespace: str, top_k: int = 3) -> str:
#     try:
#         emb = generate_embedding(profile)
#         resp = index_products.query(
#             vector=emb,
#             top_k=top_k,
#             include_metadata=True,
#             namespace=namespace
#         )
#         lines = []
#         for i, m in enumerate(resp["matches"], start=1):
#             name = m["metadata"]["name"]
#             desc = m["metadata"].get("text", "")
#             lines.append(f"{i}) {name} - {desc}")

#         prompt = (
#             f"User profile:\n{profile}\n\n"
#             f"Here are some products from {namespace}:\n" + "\n".join(lines)
#             + "\n\nPick the TOP 3 (with a one-sentence reason each)."
#         )
        
#         # Fixed Ollama chat API call
#         response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
#             {"role": "user", "content": prompt}
#         ])
#         return response["message"]["content"]
#     except Exception as e:
#         print(f"Product recommendation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Product recommendation failed: {e}")

# # ─── API endpoint ───────────────────────────────────────────────────────────
# @app.post("/recommend", response_model=RecommendResponse)
# def recommend(req: RecommendRequest):
#     if not req.profile.strip():
#         raise HTTPException(status_code=400, detail="Field 'profile' is required.")

#     try:
#         store_ranking, store_map = get_store_recommendations(req.profile, req.top_k_stores)
#         nums = extract_store_numbers(store_ranking)
#         chosen = [store_map[n] for n in nums[:2] if n in store_map]

#         prod_results = {}
#         for ns in chosen:
#             prod_results[ns] = get_product_recommendations(req.profile, ns, req.top_k_products)

#         return RecommendResponse(
#             stores_prompt=store_ranking,
#             product_prompts=prod_results
#         )
#     except Exception as e:
#         print(f"Recommendation endpoint error: {e}")
#         raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

# # ─── Health check endpoint ──────────────────────────────────────────────────
# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}

# # ─── Test endpoint for debugging ────────────────────────────────────────────
# @app.get("/test-ollama")
# def test_ollama():
#     try:
#         # Test embedding
#         test_embedding = generate_embedding("test text")
        
#         # Test chat
#         response = ollama.chat(model=OLLAMA_LLM_MODEL, messages=[
#             {"role": "user", "content": "Say hello"}
#         ])
        
#         return {
#             "embedding_length": len(test_embedding) if test_embedding else 0,
#             "chat_response": response["message"]["content"]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ollama test failed: {e}")

# # ─── Run with:
# # 1) Activate your venv:
# #    PS> .\myenv\Scripts\Activate.ps1
# # 2) Install dependencies (inside your venv):
# #    (myenv) PS> pip install fastapi uvicorn python-dotenv pinecone-client ollama
# # 3) Start the server:
# #    (myenv) PS> python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
# # 4) In your browser go to http://localhost:8000 (or 127.0.0.1:8000)
# #    Interactive docs available at http://localhost:8000/docs
# # 5) To stop the server, press CTRL+C in this terminal