# Cours complet — Les Agents avec LangChain

> Basé sur la documentation officielle : https://docs.langchain.com/oss/python/langchain/agents
> Adapté avec des exemples pratiques pour un TP (Colab + Groq)

---

## 1. Qu'est-ce qu'un agent ?

Un **agent** combine un modèle de langage (LLM) avec des **tools** (outils) pour créer un système capable de :
- raisonner sur une tâche,
- décider quel outil utiliser,
- itérer jusqu'à obtenir une solution.

Concrètement : **un agent LLM exécute des outils en boucle jusqu'à atteindre un objectif.** La boucle s'arrête quand le modèle produit une réponse finale, ou qu'une limite d'itérations est atteinte.

Dans LangChain, on crée un agent avec la fonction `create_agent`. Elle construit en interne un **graphe** (via LangGraph) : des nœuds (étapes : appel modèle, exécution d'outil, middleware) reliés par des arêtes qui définissent le flux d'exécution.

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=tools)
```

---

## 2. Les composants principaux

### 2.1 Le modèle (Model)

Le modèle est le **moteur de raisonnement** de l'agent. Deux façons de le définir :

**Modèle statique** (le plus courant) — fixé une fois pour toutes :

```python
from langchain.agents import create_agent

agent = create_agent("groq:llama-3.1-8b-instant", tools=tools)
```

Ou en instanciant directement le modèle pour un contrôle plus fin (température, max_tokens, timeout...) :

```python
from langchain.agents import create_agent
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)
agent = create_agent(model, tools=tools)
```

**Modèle dynamique** — sélectionné à l'exécution, selon l'état de la conversation (ex : basculer sur un modèle plus puissant si la conversation devient complexe). Cela se fait via un **middleware** avec le décorateur `@wrap_model_call` :

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_groq import ChatGroq

basic_model = ChatGroq(model="llama-3.1-8b-instant")
advanced_model = ChatGroq(model="llama-3.3-70b-versatile")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choisit le modèle selon la longueur de la conversation."""
    message_count = len(request.state["messages"])
    model = advanced_model if message_count > 10 else basic_model
    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # modèle par défaut
    tools=tools,
    middleware=[dynamic_model_selection],
)
```

### 2.2 Les tools (outils)

Les tools donnent à l'agent la capacité d'**agir**. L'agent gère automatiquement :
- plusieurs appels d'outils en séquence,
- des appels en parallèle,
- la sélection dynamique d'outils selon les résultats précédents,
- la logique de retry et la gestion d'erreurs,
- la persistance d'état entre les appels.

**Tools statiques** — définis une fois à la création de l'agent :

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Résultats pour : {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Météo à {location} : ensoleillé, 22°C"

agent = create_agent(model, tools=[search, get_weather])
```

Si la liste de tools est vide, l'agent devient un simple nœud LLM sans capacité d'appel d'outils.

**Tools dynamiques** — le jeu d'outils change à l'exécution. Deux cas :

1. **Filtrer des outils pré-enregistrés** (tous connus à l'avance, mais activés/désactivés selon le contexte — authentification, permissions, feature flags). On utilise `@wrap_model_call` pour intercepter la requête et filtrer `request.tools` :

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def state_based_tools(request: ModelRequest, handler) -> ModelResponse:
    is_authenticated = request.state.get("authenticated", False)
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    return handler(request)

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[public_search, private_search],
    middleware=[state_based_tools],
)
```

2. **Enregistrement d'outils à la volée** (ex : outils venant d'un serveur MCP, générés dynamiquement). Il faut alors DEUX hooks de middleware : `wrap_model_call` pour ajouter l'outil à la requête, et `wrap_tool_call` pour gérer son exécution :

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Pourboire : {tip:.2f}€, Total : {bill_amount + tip:.2f}€"

class DynamicToolMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[get_weather],  # seulement les tools statiques ici
    middleware=[DynamicToolMiddleware()],
)
```

**Gestion des erreurs d'outils** — avec `@wrap_tool_call`, on peut intercepter une exception et renvoyer un message propre au modèle plutôt que de crasher :

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Erreur outil : vérifie ton input. ({e})",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(model, tools=[search, get_weather], middleware=[handle_tool_errors])
```

### 2.3 La boucle ReAct

Les agents suivent le pattern **ReAct** (« Reasoning + Acting ») : ils alternent entre de courtes étapes de raisonnement, des appels d'outils ciblés, et intègrent le résultat (« observation ») dans la décision suivante — jusqu'à pouvoir donner une réponse finale.

**Exemple** — *« Trouve les écouteurs sans fil les plus populaires et vérifie leur disponibilité »* :

1. **Raisonnement** : « La popularité change dans le temps, je dois utiliser l'outil de recherche. »
   **Action** : `search_products("écouteurs sans fil")` → Observation : 5 résultats trouvés
2. **Raisonnement** : « Je dois vérifier le stock du modèle le mieux classé avant de répondre. »
   **Action** : `check_inventory("WH-1000XM5")` → Observation : 10 unités en stock
3. **Raisonnement** : « J'ai le modèle et son stock, je peux répondre. »
   **Action** : réponse finale à l'utilisateur.

C'est exactement cette boucle que `create_agent` orchestre automatiquement pour toi.

### 2.4 Le system prompt

Le prompt système façonne le comportement de l'agent :

```python
agent = create_agent(
    model,
    tools,
    system_prompt="Tu es un assistant utile. Sois concis et précis.",
)
```

Sans `system_prompt`, l'agent déduit sa tâche directement des messages. On peut aussi passer un objet `SystemMessage` (utile pour des fonctionnalités spécifiques à un provider comme le prompt caching Anthropic).

**Prompt système dynamique** — via le décorateur `@dynamic_prompt`, pour adapter le prompt selon le contexte d'exécution :

```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")
    base = "Tu es un assistant utile."
    if user_role == "expert":
        return f"{base} Donne des réponses techniques détaillées."
    elif user_role == "beginner":
        return f"{base} Explique simplement, sans jargon."
    return base

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explique le machine learning"}]},
    context={"user_role": "expert"},
)
```

### 2.5 Le nom de l'agent

Utile surtout en contexte **multi-agents**, où l'agent devient un sous-graphe :

```python
agent = create_agent(model, tools, name="research_assistant")
```

⚠️ Préférer le `snake_case` (`research_assistant`) : certains providers rejettent les noms avec espaces ou caractères spéciaux — ça vaut aussi pour les noms de tools.

---

## 3. Invoquer l'agent

On invoque l'agent en lui passant un état — a minima une liste de messages :

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Quel temps fait-il à Paris ?"}]}
)
```

Pour suivre l'exécution étape par étape (utile en debug ou pour un affichage progressif) :

```python
from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Cherche des news IA et résume-les"}]},
    stream_mode="values",
):
    latest = chunk["messages"][-1]
    if latest.content:
        if isinstance(latest, HumanMessage):
            print(f"User: {latest.content}")
        elif isinstance(latest, AIMessage):
            print(f"Agent: {latest.content}")
    elif latest.tool_calls:
        print(f"Appel d'outils : {[tc['name'] for tc in latest.tool_calls]}")
```

---

## 4. Concepts avancés

### 4.1 Sortie structurée (`response_format`)

Deux stratégies pour forcer l'agent à répondre dans un format précis (ex : un objet Pydantic) :

**`ToolStrategy`** — utilise un « faux » appel d'outil pour générer la sortie structurée. Fonctionne avec n'importe quel modèle supportant le tool calling :

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extrait les infos : John Doe, john@example.com, 06 12 34 56 78"}]
})
result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='06 12 34 56 78')
```

**`ProviderStrategy`** — utilise la génération de sortie structurée **native** du provider (plus fiable, mais dispo seulement chez certains providers) :

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(model="openai:gpt-5.4", response_format=ProviderStrategy(ContactInfo))
```

Depuis `langchain 1.0`, passer directement un schéma (`response_format=ContactInfo`) choisit automatiquement `ProviderStrategy` si le modèle le supporte, sinon bascule sur `ToolStrategy`.

### 4.2 Mémoire (memory)

L'agent conserve automatiquement l'historique de conversation via l'état des messages — c'est sa **mémoire à court terme**. On peut étendre cet état pour lui faire retenir des infos additionnelles, via un `state_schema` personnalisé (doit hériter de `AgentState`, en `TypedDict`) :

```python
from langchain.agents import AgentState, create_agent

class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Je préfère les explications techniques"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

Autre approche, préférée pour garder l'extension d'état liée à un middleware précis :

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state, runtime):
        ...

agent = create_agent(model, tools=tools, middleware=[CustomMiddleware()])
```

Pour de la **mémoire long terme** persistante entre plusieurs sessions (au-delà d'une seule conversation), LangChain propose un concept dédié (`Store`) — hors du périmètre de ce cours mais à explorer dans la doc « Long-term memory ».

### 4.3 Middleware

Le middleware permet d'intercepter et modifier le comportement de l'agent à différents moments de son exécution :
- traiter l'état **avant** l'appel modèle (ex : trimming de messages, injection de contexte),
- modifier ou valider la réponse du modèle (guardrails, filtrage),
- gérer les erreurs d'outils avec une logique custom,
- faire de la sélection dynamique de modèle,
- ajouter du logging/monitoring.

Principaux décorateurs : `@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`, `@dynamic_prompt`.

---

### 4.4 Checkpointer et `InMemorySaver` — donner une vraie mémoire à l'agent
 
Jusqu'ici, chaque `agent.invoke(...)` est **stateless** : l'agent ne se souvient de rien d'un appel à l'autre, sauf si tu renvoies toi-même tout l'historique des messages à chaque fois. Le **checkpointer** résout ce problème.
 
#### Le concept
 
Le checkpointer est le mécanisme de LangGraph (le moteur sous-jacent de `create_agent`) qui **sauvegarde un instantané ("checkpoint") de l'état du graphe à chaque étape** de son exécution (après chaque appel modèle, chaque exécution d'outil...). Il permet :
 
- **La mémoire conversationnelle** : l'agent retrouve automatiquement l'historique d'une conversation, sans que tu aies à le renvoyer manuellement.
- **La reprise après interruption** : utile notamment pour le [human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop), où l'exécution peut être mise en pause en attendant une validation humaine, puis reprise exactement où elle s'était arrêtée.
- **L'isolation multi-conversations** : plusieurs utilisateurs peuvent dialoguer avec le même agent sans que leurs historiques se mélangent, grâce à un **`thread_id`**.
Le checkpointer est **invisible dans ton code métier** : tu ne l'appelles jamais toi-même dans un tool ou un middleware, LangGraph s'en charge automatiquement dès que tu le passes à `create_agent`.
 
Chaque conversation est identifiée par un `thread_id` (un simple string que tu choisis), passé dans le `config` à chaque `invoke` :
 
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
 
checkpointer = InMemorySaver()
 
agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    checkpointer=checkpointer,
)
 
config = {"configurable": {"thread_id": "demo-thread-1"}}
 
agent.invoke({"messages": [{"role": "user", "content": "Salut, je m'appelle Bob."}]}, config)
result = agent.invoke({"messages": [{"role": "user", "content": "Je m'appelle comment ?"}]}, config)
print(result["messages"][-1].content)
# "Tu t'appelles Bob !"
```
 
`InMemorySaver` (nom actuel de la classe — l'ancienne s'appelait `MemorySaver`, désormais dépréciée) est l'implémentation **la plus simple** : elle stocke tout **en RAM, dans le process Python**. Parfait pour prototyper et pour un TP, mais tout est perdu au redémarrage du notebook/serveur — ce n'est **pas persistant**.
 
#### Checkpointer vs Store — ne pas confondre
 
| | **Checkpointer** | **Store** |
|---|---|---|
| Portée | Une conversation (un `thread_id`) | Transversal, entre plusieurs threads/utilisateurs |
| Usage | Historique des messages, état d'exécution | Faits appris sur un utilisateur, préférences durables |
| Écriture | Automatique, géré par LangGraph | Explicite, dans tes tools/middleware |
| Analogie | Mémoire **court terme** de la conversation | Mémoire **long terme**, façon profil utilisateur |
 
#### Passer en vraie application : persister dans une base de données
 
En production, `InMemorySaver` ne suffit pas (perte de données au redémarrage, ne scale pas sur plusieurs instances de ton backend). LangGraph fournit des checkpointers backés par une vraie base de données, avec **exactement la même interface** — seul le stockage change.
 
**Option SQLite** — fichier local, simple pour une petite app ou un prototype qui doit survivre aux redémarrages :
 
```python
!pip install -qU langgraph-checkpoint-sqlite
 
from langgraph.checkpoint.sqlite import SqliteSaver
 
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    agent = create_agent(
        model="groq:llama-3.1-8b-instant",
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "user-42"}}
    result = agent.invoke({"messages": [{"role": "user", "content": "Salut !"}]}, config)
```
 
**Option PostgreSQL** — la solution recommandée pour une vraie application en production (multi-utilisateurs, plusieurs instances de serveur, besoin de fiabilité) :
 
```python
!pip install -qU langgraph-checkpoint-postgres psycopg[binary]
 
from langgraph.checkpoint.postgres import PostgresSaver
 
DB_URI = "postgresql://user:password@localhost:5432/ma_base"
 
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # ⚠️ à appeler une seule fois : crée les tables nécessaires
 
    agent = create_agent(
        model="groq:llama-3.1-8b-instant",
        checkpointer=checkpointer,
    )
 
    config = {"configurable": {"thread_id": "user-42"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Salut, je suis de retour !"}]},
        config,
    )
    print(result["messages"][-1].content)
```
 
Points importants pour une vraie app :
 
- **`.setup()`** doit être appelé **une seule fois** (à l'installation/déploiement), pas à chaque démarrage de l'app : il crée les tables SQL nécessaires au stockage des checkpoints.
- Pour une app **asynchrone** (ex : backend FastAPI), utilise les variantes async : `AsyncPostgresSaver` / `AsyncSqliteSaver`, avec `await checkpointer.asetup()` et `await agent.ainvoke(...)`.
- Le `thread_id` correspond en général à une **session de chat** ou une **conversation** dans ta base applicative (souvent stocké en lien avec ton `user_id` côté app). C'est toi qui décides de la stratégie : un `thread_id` par utilisateur (mémoire globale), ou un par conversation (comme des "chats" séparés façon ChatGPT).
- Sur des conversations très longues, les checkpoints s'accumulent et peuvent alourdir le stockage/la latence : en production, on met en place une politique de rétention (ex : cron job qui supprime les checkpoints de plus de N jours) ou on trimme régulièrement les messages anciens (cf. middleware `@before_model`, section 2.2 / doc [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)).
- Pour une app qui doit se souvenir d'infos **entre plusieurs conversations différentes** (pas juste dans un thread), c'est le rôle du **`store`** (`InMemoryStore`, ou une version persistante) plutôt que du checkpointer — voir [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory).
 
---
 
## 6. Pour aller plus loin
 
- Doc officielle Agents : https://docs.langchain.com/oss/python/langchain/agents
- Tools : https://docs.langchain.com/oss/python/langchain/tools
- Middleware : https://docs.langchain.com/oss/python/langchain/middleware/overview
- Short-term memory : https://docs.langchain.com/oss/python/langchain/short-term-memory
- Long-term memory : https://docs.langchain.com/oss/python/langchain/long-term-memory
- Retrieval (RAG) : https://docs.langchain.com/oss/python/langchain/retrieval
- Structured output : https://docs.langchain.com/oss/python/langchain/structured-output
- Human-in-the-loop : https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- Streaming : https://docs.langchain.com/oss/python/langchain/streaming
