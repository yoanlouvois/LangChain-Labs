"""
agent.py

Agent LangChain de recommandation de technologies / bibliothèques,
avec recherche web (Tavily) et mémoire de conversation (checkpointer).

Ce module ne fait qu'exposer l'agent et une fonction d'invocation.
La partie API (FastAPI) ou UI (Streamlit) viendra l'importer ensuite.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()  # charge les variables depuis .env local

REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TAVILY_API_KEY"]
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing:
    raise RuntimeError(
        f"Variables d'environnement manquantes : {', '.join(missing)}. "
        "Ajoute-les dans un fichier .env à la racine du projet."
    )

MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la recommandation de technologies "
    "et de bibliothèques informatiques.\n\n"
    "L'utilisateur fournit :\n"
    "- un besoin ou un type de projet ;\n"
    "- un langage ou une stack technique.\n\n"
    "Propose entre 3 et 5 solutions adaptées.\n"
    "Utilise l'outil de recherche web pour vérifier la licence, "
    "la tarification et les éventuelles restrictions d'utilisation.\n"
    "Privilégie les sources officielles.\n\n"
    "Retourne uniquement un tableau Markdown avec les colonnes suivantes :\n"
    "| Solution | Description | Licence ou tarification |\n\n"
    "Dans la dernière colonne, précise par exemple : MIT, Apache 2.0, "
    "open source, gratuit, freemium ou payant. "
    "Si l'information n'a pas pu être vérifiée, indique 'À vérifier'."
)

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.7,
    max_tokens=512,
)

search_tool = TavilySearch(
    max_results=2,
    search_depth="basic",
    include_answer=True,
)

tools = [search_tool]
checkpointer = InMemorySaver()

# Agent
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
)


def ask_agent(user_input: str, thread_id: str = "default") -> str:
    """
    Envoie une requête à l'agent et retourne uniquement sa réponse finale.

    Parameters
    ----------
    user_input : str
        Besoin/projet formulé par l'utilisateur.
    thread_id : str
        Identifiant de conversation utilisé par le checkpointer. Réutiliser
        le même thread_id permet à l'agent de garder le contexte entre
        plusieurs appels (ex: une session utilisateur).

    Returns
    -------
    str
        Réponse finale générée par l'agent (tableau Markdown).
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    return result["messages"][-1].content


# Test en standalone (python agent.py)

if __name__ == "__main__":
    reponse = ask_agent(
        user_input=(
            "Type de projet : site web de cartographie 2D interactive\n"
            "Stack : JavaScript"
        ),
        thread_id="test-standalone",
    )
    print(reponse)