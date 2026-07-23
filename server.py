"""
server.py

Mini serveur FastAPI qui expose l'agent défini dans agent.py.

Endpoints :
- POST   /configs             -> crée une nouvelle configuration (conversation)
- GET    /configs             -> liste toutes les configurations
- GET    /configs/{config_id} -> détail d'une configuration
- DELETE /configs/{config_id} -> supprime une configuration (et sa mémoire)
- POST   /configs/{config_id}/ask -> envoie un message à l'agent dans cette configuration

Une "configuration" = une conversation isolée avec l'agent, identifiée par
un thread_id géré par le checkpointer (InMemorySaver dans agent.py).
"""

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import agent, checkpointer

app = FastAPI(title="Tech Advisor Agent API")


# Stockage des configurations (métadonnées uniquement -- l'historique des
# messages, lui, vit dans le checkpointer de l'agent)-

class ConfigMeta(BaseModel):
    id: str
    name: str
    created_at: str


configs: dict[str, ConfigMeta] = {}

class CreateConfigRequest(BaseModel):
    name: str = Field(default="Sans nom", description="Nom lisible de la configuration")


class AskRequest(BaseModel):
    message: str = Field(description="Message envoyé à l'agent")


class TechSolution(BaseModel):
    name: str
    description: str
    license_or_pricing: str
    url: str


class AskResponse(BaseModel):
    config_id: str
    solutions: list[TechSolution]


# --------------------------------------------------------------------------
# Endpoints - Configurations
# --------------------------------------------------------------------------

@app.post("/configs", response_model=ConfigMeta)
def create_config(req: CreateConfigRequest):
    """Crée une nouvelle configuration (= une nouvelle conversation avec l'agent)."""
    config_id = str(uuid.uuid4())
    meta = ConfigMeta(
        id=config_id,
        name=req.name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    configs[config_id] = meta
    return meta


@app.get("/configs", response_model=list[ConfigMeta])
def list_configs():
    """Liste toutes les configurations existantes."""
    return list(configs.values())


@app.get("/configs/{config_id}", response_model=ConfigMeta)
def get_config(config_id: str):
    """Récupère le détail d'une configuration."""
    meta = configs.get(config_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Configuration introuvable")
    return meta


@app.delete("/configs/{config_id}")
def delete_config(config_id: str):
    """Supprime une configuration ET sa mémoire de conversation associée."""
    if config_id not in configs:
        raise HTTPException(status_code=404, detail="Configuration introuvable")

    # Supprime tous les checkpoints/mémoire liés à ce thread_id
    checkpointer.delete_thread(config_id)
    del configs[config_id]

    return {"detail": f"Configuration {config_id} supprimée"}


# --------------------------------------------------------------------------
# Endpoint - Utilisation de l'agent
# --------------------------------------------------------------------------

@app.post("/configs/{config_id}/ask", response_model=AskResponse)
def ask(config_id: str, req: AskRequest):
    """Envoie un message à l'agent dans le contexte d'une configuration donnée."""
    if config_id not in configs:
        raise HTTPException(status_code=404, detail="Configuration introuvable")

    thread_config = {"configurable": {"thread_id": config_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=thread_config,
    )
    structured = result["structured_response"]

    return AskResponse(
        config_id=config_id,
        solutions=[TechSolution(**s.model_dump()) for s in structured.solutions],
    )



# Lancement direct (python server.py) -- sinon: uvicorn server:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)