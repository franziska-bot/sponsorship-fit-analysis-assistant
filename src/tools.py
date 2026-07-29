import os

from langchain_tavily import TavilySearch

def get_search_tool():
    """Erstellt das Tavily-Such-Tool für die Firmenrecherche."""
    return TavilySearch(
        max_results=5,
        api_key=os.environ["TAVILY_API_KEY"],
    )
