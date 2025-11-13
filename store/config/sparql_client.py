import requests
from data.config.utils import ORKG_SPARQL_ENDPOINT, HEADERS # type: ignore

def execute_sparql_query(query: str):
    """Send SPARQL query to ORKG et return result"""
    response = requests.get(f"{ORKG_SPARQL_ENDPOINT}?query={query}", headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
# sourcery skip: raise-specific-error
        raise Exception(f"SPARQL error: {response.status_code}, {response.text}")
