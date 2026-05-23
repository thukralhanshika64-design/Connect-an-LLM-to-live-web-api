import os

import requests
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI GitHub Trends",
    page_icon="🤖",
    layout="wide",
)

st.title("AI GitHub Trend Analyzer")
st.write(
    "Fetch trending GitHub repositories for a topic and use a small local language model to highlight which repos are best for learning AI."
)

query = st.text_input("Search topic", value="machine learning")
repo_count = st.slider("Number of repositories to fetch", min_value=1, max_value=10, value=5)
show_descriptions = st.checkbox("Show repository descriptions", value=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


@st.cache_data(show_spinner=False)
def get_trending_repos(query: str, per_page: int = 5):
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    }
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API Error {response.status_code}: {response.text}"
        )

    data = response.json()
    repos = []
    for repo in data.get("items", []):
        repos.append(
            {
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description") or "No description provided.",
                "stars": repo.get("stargazers_count", 0),
                "url": repo.get("html_url"),
            }
        )
    return repos


@st.cache_resource(show_spinner=False)
def load_llm():
    return pipeline("text-generation", model="distilgpt2")


def generate_insights(repos):
    repo_text = "\n".join(
        [
            f"{r['full_name']} ({r['stars']}⭐): {r['description']}"
            for r in repos
        ]
    )

    prompt = (
        "Analyze the following GitHub repositories and recommend which are most useful for learning AI. "
        "Give a short, clear explanation for each top recommendation.\n\n"
        f"{repo_text}\n\n"
    )

    llm = load_llm()
    result = llm(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"].strip()


if st.button("Fetch and Analyze"):
    try:
        with st.spinner("Fetching trending repositories..."):
            repos = get_trending_repos(query, repo_count)

        if not repos:
            st.warning("No repositories returned for this query. Try a different topic.")
        else:
            st.success(f"Found {len(repos)} repositories for '{query}'.")
            st.markdown("### Trending repositories")
            for repo in repos:
                st.write(
                    f"**[{repo['full_name']}]({repo['url']})** — {repo['stars']}⭐"
                )
                if show_descriptions:
                    st.write(repo["description"])

            with st.spinner("Generating AI insights..."):
                analysis = generate_insights(repos)
            st.markdown("### AI recommendations")
            st.write(analysis)

    except Exception as exc:
        st.error(f"Error: {exc}")

if GITHUB_TOKEN:
    st.caption("Using GitHub token from environment for higher API rate limits.")
else:
    st.caption(
        "No GitHub token configured. Add one in Streamlit Cloud secrets as GITHUB_TOKEN for more reliable API access."
    )
