"""
agent.py

Agent LangChain de recommandation de technologies / bibliothèques,
avec recherche web (Tavily) et mémoire de conversation (checkpointer).

Ce module ne fait qu'exposer l'agent et une fonction d'invocation.
La partie API (FastAPI) ou UI (Streamlit) viendra l'importer ensuite.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------

load_dotenv()  # charge les variables depuis un fichier .env local

REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TAVILY_API_KEY"]
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing:
    raise RuntimeError(
        f"Variables d'environnement manquantes : {', '.join(missing)}. "
        "Ajoute-les dans un fichier .env à la racine du projet."
    )

# llama-3.1-8b-instant est rapide mais peu fiable pour le tool calling :
# il lui arrive d'écrire l'appel d'outil en texte brut dans sa réponse
# finale (ex: "<function=tavily_search>...") au lieu de faire un vrai appel
# structuré. Le 70B est beaucoup plus fiable sur ce point.
MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la recommandation de technologies "
    "et de bibliothèques informatiques.\n\n"
    "L'utilisateur fournit :\n"
    "- un besoin ou un type de projet ;\n"
    "- un langage ou une stack technique.\n\n"
    "Propose entre 3 et 5 solutions adaptées.\n"
    "Utilise l'outil de recherche web pour vérifier la licence, "
    "la tarification, et trouver le lien officiel (site ou doc) de chaque "
    "solution. Privilégie les sources officielles.\n\n"
    "Si une information n'a pas pu être vérifiée, indique 'À vérifier'."
)


# --------------------------------------------------------------------------
# Schéma de sortie structurée
# --------------------------------------------------------------------------
# On force l'agent à répondre selon ce schéma plutôt que de lui demander un
# tableau Markdown en texte libre : ça élimine le risque que du texte
# parasite (raisonnement, appels d'outils mal formatés...) se glisse dans
# la réponse finale.

class TechSolution(BaseModel):
    name: str = Field(description="Nom de la solution/bibliothèque")
    description: str = Field(description="Description courte, une phrase")
    license_or_pricing: str = Field(
        description="Licence ou tarification, ex: MIT, Apache 2.0, open source, gratuit, freemium, payant, À vérifier"
    )
    url: str = Field(description="Lien officiel (site ou documentation)")


class TechRecommendations(BaseModel):
    solutions: list[TechSolution]

# --------------------------------------------------------------------------
# 2. Modèle
# --------------------------------------------------------------------------

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.7,
    max_tokens=512,
)

# --------------------------------------------------------------------------
# 3. Outils
# --------------------------------------------------------------------------

search_tool = TavilySearch(
    max_results=2,
    search_depth="basic",
    include_answer=True,
)

tools = [search_tool]

# --------------------------------------------------------------------------
# 4. Mémoire (checkpointer)
# --------------------------------------------------------------------------

# InMemorySaver : suffisant pour du dev/test local (rien n'est persisté sur
# disque). Pour une vraie appli il faudra basculer sur un SqliteSaver ou
# PostgresSaver -- même interface, seul le stockage change.
checkpointer = InMemorySaver()

# --------------------------------------------------------------------------
# 5. Agent
# --------------------------------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(TechRecommendations),
)


# --------------------------------------------------------------------------
# 6. Fonction d'invocation
# --------------------------------------------------------------------------

def ask_agent(user_input: str, thread_id: str = "default") -> list[dict]:
    """
    Envoie une requête à l'agent et retourne sa réponse structurée.

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
    list[dict]
        Liste de solutions, chacune avec name / description /
        license_or_pricing / url.
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    structured: TechRecommendations = result["structured_response"]
    return [s.model_dump() for s in structured.solutions]


# --------------------------------------------------------------------------
# 7. Test en standalone (python agent.py)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    reponse = ask_agent(
        user_input=(
            "Type de projet : site web de cartographie 2D interactive\n"
            "Stack : JavaScript"
        ),
        thread_id="test-standalone",
    )
    print(reponse)