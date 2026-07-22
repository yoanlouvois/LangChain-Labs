"""
streamlit_app.py

Interface Streamlit pour consommer l'API FastAPI de l'agent (server.py).

Lancer (dans un terminal séparé du serveur FastAPI) :
    streamlit run streamlit_app.py

Le serveur FastAPI doit tourner en parallèle sur http://127.0.0.1:8000
    uvicorn server:app --reload --port 8000
"""

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Tech Advisor Agent", layout="wide")


# --------------------------------------------------------------------------
# Fonctions d'appel à l'API
# --------------------------------------------------------------------------

def api_list_configs() -> list[dict]:
    r = requests.get(f"{API_URL}/configs")
    r.raise_for_status()
    return r.json()


def api_create_config(name: str) -> dict:
    r = requests.post(f"{API_URL}/configs", json={"name": name})
    r.raise_for_status()
    return r.json()


def api_delete_config(config_id: str) -> None:
    r = requests.delete(f"{API_URL}/configs/{config_id}")
    r.raise_for_status()


def api_ask(config_id: str, message: str) -> list[dict]:
    r = requests.post(f"{API_URL}/configs/{config_id}/ask", json={"message": message})
    r.raise_for_status()
    return r.json()["solutions"]


def render_solutions(solutions: list[dict]) -> None:
    """Affiche les solutions sous forme de tableau avec liens cliquables."""
    if not solutions:
        st.caption("Aucune solution retournée.")
        return

    df = pd.DataFrame(solutions)[["name", "description", "license_or_pricing", "url"]]
    df.columns = ["Solution", "Description", "Licence / tarification", "Lien"]

    st.dataframe(
        df,
        column_config={
            "Lien": st.column_config.LinkColumn("Lien", display_text="Ouvrir ↗"),
        },
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# État de session
# --------------------------------------------------------------------------

if "active_config_id" not in st.session_state:
    st.session_state.active_config_id = None

if "chat_history" not in st.session_state:
    # dict {config_id: [(role, content), ...]}
    st.session_state.chat_history = {}


# --------------------------------------------------------------------------
# Sidebar - Gestion des configurations
# --------------------------------------------------------------------------

st.sidebar.title("Configurations")

# Vérifie que le serveur FastAPI répond avant d'aller plus loin
try:
    configs = api_list_configs()
except requests.exceptions.ConnectionError:
    st.sidebar.error(
        "Impossible de joindre l'API. Vérifie qu'elle tourne bien sur "
        f"{API_URL} (`uvicorn server:app --reload --port 8000`)."
    )
    st.stop()

with st.sidebar.form("create_config_form", clear_on_submit=True):
    new_name = st.text_input("Nom de la nouvelle configuration", placeholder="ex: projet cartographie")
    submitted = st.form_submit_button("+ Nouvelle configuration")
    if submitted:
        name = new_name.strip() or "Sans nom"
        new_config = api_create_config(name)
        st.session_state.active_config_id = new_config["id"]
        st.session_state.chat_history[new_config["id"]] = []
        st.rerun()

st.sidebar.divider()

if not configs:
    st.sidebar.caption("Aucune configuration pour l'instant.")
else:
    for cfg in configs:
        col1, col2 = st.sidebar.columns([4, 1])
        is_active = cfg["id"] == st.session_state.active_config_id

        with col1:
            label = f"**{cfg['name']}**" if is_active else cfg["name"]
            if st.button(label, key=f"select_{cfg['id']}", use_container_width=True):
                st.session_state.active_config_id = cfg["id"]
                st.session_state.chat_history.setdefault(cfg["id"], [])
                st.rerun()

        with col2:
            if st.button("🗑", key=f"delete_{cfg['id']}"):
                api_delete_config(cfg["id"])
                st.session_state.chat_history.pop(cfg["id"], None)
                if st.session_state.active_config_id == cfg["id"]:
                    st.session_state.active_config_id = None
                st.rerun()


# --------------------------------------------------------------------------
# Zone principale - Chat avec l'agent
# --------------------------------------------------------------------------

st.title("💬 Tech Advisor Agent")

active_id = st.session_state.active_config_id

if active_id is None:
    st.info("Crée ou sélectionne une configuration dans la barre latérale pour commencer.")
    st.stop()

active_config = next((c for c in configs if c["id"] == active_id), None)
if active_config is None:
    st.session_state.active_config_id = None
    st.rerun()

st.caption(f"Configuration active : **{active_config['name']}**")

history = st.session_state.chat_history.setdefault(active_id, [])

for role, content in history:
    with st.chat_message(role):
        if role == "assistant":
            render_solutions(content)
        else:
            st.markdown(content)

user_message = st.chat_input("Décris ton projet et ta stack technique...")

if user_message:
    history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            try:
                solutions = api_ask(active_id, user_message)
                error = None
            except requests.exceptions.RequestException as e:
                solutions = []
                error = f"Erreur lors de l'appel à l'agent : {e}"

        if error:
            st.error(error)
        else:
            render_solutions(solutions)

    history.append(("assistant", solutions if not error else []))