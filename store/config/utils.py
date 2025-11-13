import os
from dotenv import load_dotenv # type: ignore

load_dotenv()
ORKG_SPARQL_ENDPOINT = os.getenv("ORKG_SPARQL_ENDPOINT", "https://orkg.org/triplestore")
HEADERS = {"Accept": "application/sparql-results+json"}
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")