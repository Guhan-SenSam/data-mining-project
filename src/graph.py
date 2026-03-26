"""Build user interaction graph from Reddit posts and comments."""

import logging

import networkx as nx
import pandas as pd

from src.utils import PROCESSED_DIR

logger = logging.getLogger(__name__)

EXCLUDED_AUTHORS = {"[deleted]", "AutoModerator"}


def build_interaction_graph(posts: pd.DataFrame, comments: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph where edges represent comment interactions.

    Edges go from commenter -> post author (for top-level comments)
    and from commenter -> parent comment author (for replies).
    """
    G = nx.DiGraph()

    # Build lookup: post_id -> author
    post_authors = dict(zip(posts["post_id"], posts["author"]))

    # Build lookup: comment_id -> author
    comment_authors = dict(zip(comments["comment_id"], comments["author"]))

    for _, comment in comments.iterrows():
        commenter = comment["author"]
        if commenter in EXCLUDED_AUTHORS:
            continue

        parent_id = comment["parent_id"]
        post_id = comment["post_id"]
        timestamp = comment["created_utc"]
        subreddit = post_authors.get(post_id, "")

        # Determine who the commenter is replying to
        if parent_id.startswith("t3_"):
            target_author = post_authors.get(parent_id[3:])
        elif parent_id.startswith("t1_"):
            target_author = comment_authors.get(parent_id[3:])
        else:
            continue

        if not target_author or target_author in EXCLUDED_AUTHORS:
            continue
        if commenter == target_author:
            continue

        if G.has_edge(commenter, target_author):
            G[commenter][target_author]["weight"] += 1
        else:
            G.add_edge(commenter, target_author, weight=1,
                       subreddit=subreddit, timestamp=timestamp, post_id=post_id)

    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    posts = pd.read_csv(PROCESSED_DIR / "posts.csv")
    comments = pd.read_csv(PROCESSED_DIR / "comments.csv")
    G = build_interaction_graph(posts, comments)
    nx.write_gexf(G, str(PROCESSED_DIR / "interaction_graph.gexf"))
    logger.info("Saved graph to %s", PROCESSED_DIR / "interaction_graph.gexf")
