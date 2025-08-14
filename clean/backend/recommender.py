import os
import json
import re
import ollama
from dotenv import load_dotenv
from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4cH8kP_SEu6dcG2q2iPPFW8HwG1famWWy5BYNccqVdK7kXLezc6YAFfW8GTdsLZS96nRH8")

# Indexes
store_index_name = "stores-list"
product_index_name = "catalog"
index_s = pc.Index(store_index_name)
index_p = pc.Index(product_index_name)

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"   # Must be pulled with ollama, 768-dimensional output
OLLAMA_LLM_MODEL = "llama2"                   # Or "mistral" etc; must also be pulled

def generate_ollama_embedding(text, model=OLLAMA_EMBEDDING_MODEL):
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating Ollama embedding: {e}")
        return None


def ollama_chat(user_prompt, model=OLLAMA_LLM_MODEL, max_tokens=450):
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': user_prompt}],
            options={'num_predict': max_tokens}
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating Ollama LLM completion: {e}")
        return "Error generating recommendations."


def get_store_recommendations(profile_description, index=index_s, top_k=5):
    query_embedding = generate_ollama_embedding(profile_description)
    if query_embedding is None:
        return "Could not generate embedding for profile description.", {}

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    if results['matches']:
        store_map = {}
        store_descriptions = []
        for idx, match in enumerate(results['matches'], 1):
            store_namespace = match['metadata']['store_name'].strip().lower().replace("'", "").replace(" ", "")
            store_map[str(idx)] = store_namespace
            display_name = match['metadata']['store_name']
            desc = match['metadata'].get('description', 'No description available.')[:200] + "..."
            store_descriptions.append(f"{idx}. {display_name}: {desc}")
        prompt = (
            f"Considering the profile: {profile_description}\n"
            f"Here are some stores:\n"
            + "\n".join(store_descriptions) +
            (
                "\n\n"
                "Based on this profile, please select the three most suitable stores by their number (e.g., 1, 3, 5) and give reasons for each choice. "
                "Please stick to the given stores, don't generate new ones. Start each suggestion with the store number."
            )
        )
        response = ollama_chat(prompt)
        return response.strip(), store_map
    else:
        print("No suitable stores were found based on your preferences.")
        return "No suitable stores were found based on your preferences.", {}


def extract_store_numbers(recommendation_text):
    # Looks for lines starting with "1.", "2." (or "- 1.", "-- 1.", etc.)
    return re.findall(r"^\s*-?\s*(\d+)\.", recommendation_text, flags=re.MULTILINE)


def get_recommendations(profile_description, index=index_p, top_k=5):
    recs, store_map = get_store_recommendations(profile_description, index=index_s)
    if not recs or not store_map:
        print("Unable to determine recs from the profile description.")
        return

    print("Store Recommendations:\n")
    print(recs)
    numbers = extract_store_numbers(recs)
    namespaces = [store_map.get(num) for num in numbers if store_map.get(num)]

    query_embedding = generate_ollama_embedding(profile_description)
    if query_embedding is None:
        print("Could not generate embedding for profile description (product search).")
        return

    for namespace in namespaces:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        if results['matches']:
            product_descriptions = [
                f"{match['metadata']['name']} - {match['metadata'].get('description', match['metadata'].get('text', 'No description available.'))}"
                for match in results['matches']
            ]

            # Compose prompt for product recommendations
            prompt = (
                f"Considering the profile: {profile_description}.\n"
                f"Here are some products from {namespace}:\n"
                + "\n".join([f"- {desc}" for desc in product_descriptions]) +
                (
                    "\n\n"
                    "Based on this profile, please suggest only three products that would be most suitable, including reasons for each choice. "
                    "Please stick to the given products, don't generate new ones."
                )
            )

            response = ollama_chat(prompt)
            if response:
                # Clean up and print response
                lines = response.split('\n')
                structured_response = ""
                count = 1
                for line in lines:
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                    clean_line = re.sub(r'^##\s*\d+\.\s*', '', clean_line)
                    if clean_line:
                        structured_response += f"{clean_line}\n"
                        count += 1
                structured_response = '\n'.join([line for line in structured_response.split('\n') if line.strip()])
                print("\n")
                print(f"Recommendations for {namespace}:\n")
                print(structured_response.strip())
            else:
                print(f"Unexpected response format from Ollama LLM for {namespace}.")
        else:
            print(f"No suitable products were found in the {namespace} store based on your preferences.")


# Example usage
if __name__ == "__main__":
    profile_description = "A girl of age 3, who likes the color yellow, is super energetic and likes to talk to people."
    get_recommendations(profile_description)