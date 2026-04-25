"""network_synergy.py — Graph-based player synergy model for football.

Models a team as a directed pass/assist network (graph). Each player is a
node; edges represent pass/assist connections weighted by frequency and danger.
Computes PageRank-style centrality to measure how "critical" each player is to
the team's structure. When a key player is absent, the lambda (expected goals)
drops non-linearly based on how central they were.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 50
_PAGERANK_TOL = 1e-6
_KEY_NODE_CENTRALITY = 0.20  # centrality above this → key player
_CRITICAL_NODE_CENTRALITY = 0.35  # centrality above this → critical player


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlayerNode:
    """A player (node) in the team pass network."""

    name: str
    position: str  # "GK" | "DEF" | "MID" | "FWD"
    xg_direct: float  # direct xG contribution (shots)
    xa_direct: float  # direct xA contribution (key passes)
    pass_accuracy: float  # 0-1
    centrality: float = 0.0  # PageRank centrality (computed)
    is_key: bool = False  # centrality > _KEY_NODE_CENTRALITY
    is_critical: bool = False  # centrality > _CRITICAL_NODE_CENTRALITY


@dataclass
class PassEdge:
    """A directed pass connection between two players."""

    from_player: str
    to_player: str
    pass_count: int  # number of passes in dataset
    dangerous_count: int  # passes that led to shots/xG
    weight: float = 0.0  # normalised edge weight (computed)


@dataclass
class TeamNetwork:
    """Complete pass network for one team."""

    team: str
    players: Dict[str, PlayerNode]  # name → PlayerNode
    edges: List[PassEdge]
    total_xg: float
    total_xa: float

    def adjacency(self) -> Dict[str, Dict[str, float]]:
        """Return {from_name: {to_name: weight}} dict."""
        adj: Dict[str, Dict[str, float]] = {}
        for edge in self.edges:
            adj.setdefault(edge.from_player, {})[edge.to_player] = edge.weight
        return adj


@dataclass
class SynergyImpact:
    """Impact of removing one or more players from the network."""

    team: str
    absent_players: List[str]
    original_centrality_sum: float  # sum of absent players' centralities
    lambda_multiplier: float  # multiply team lambda by this (< 1 if loss)
    xg_delta: float  # absolute change in team xG
    network_disruption: float  # 0–1 score of how disrupted the network is
    impact_level: str  # "NONE" | "MINOR" | "MAJOR" | "CRITICAL"
    description: str

    def __str__(self) -> str:
        return (
            f"SynergyImpact [{self.team}] absent={self.absent_players} "
            f"λ_mult={self.lambda_multiplier:.3f} "
            f"xG_delta={self.xg_delta:+.3f} "
            f"disruption={self.network_disruption:.2f} "
            f"impact={self.impact_level}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NetworkSynergyEngine:
    """
    Builds and analyses team pass networks to quantify player synergy.

    PageRank centrality:
      Standard PageRank with damping factor 0.85, 50 iterations.
      Adjacency matrix built from PassEdge weights (normalised per out-node).

    Lambda adjustment when player absent:
      disruption = sum(centrality_i for i in absent)
      lambda_multiplier = max(0.40, (1 - disruption) ** 1.5)
      (non-linear: losing a 0.40-centrality player hurts more than losing two
      0.20s)

    xG delta:
      xg_delta = -(sum(player.xg_direct) + network_effect × team_total_xg)
      where network_effect = disruption × 0.30  (indirect losses via broken
      connections)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_network(
        self,
        team: str,
        players: List[PlayerNode],
        edges: List[PassEdge],
    ) -> TeamNetwork:
        """
        Build TeamNetwork, normalise edge weights, compute PageRank centrality.

        Edge normalisation: weight = dangerous_count / max(total_dangerous, 1)
        PageRank: standard algorithm on adjacency dict.
        After PageRank, set player.centrality, player.is_key, player.is_critical.
        total_xg = sum(p.xg_direct), total_xa = sum(p.xa_direct).
        """
        # Collect known player names from both player list and edges so orphan
        # edge endpoints are still represented in the node set.
        player_map: Dict[str, PlayerNode] = {p.name: p for p in players}

        # Normalise edge weights
        total_dangerous = sum(e.dangerous_count for e in edges)
        denom = max(total_dangerous, 1)
        for edge in edges:
            edge.weight = edge.dangerous_count / denom

        logger.debug(
            "build_network: team=%s players=%d edges=%d total_dangerous=%d",
            team,
            len(players),
            len(edges),
            total_dangerous,
        )

        # Compute totals
        total_xg = sum(p.xg_direct for p in players)
        total_xa = sum(p.xa_direct for p in players)

        network = TeamNetwork(
            team=team,
            players=player_map,
            edges=edges,
            total_xg=total_xg,
            total_xa=total_xa,
        )

        # Compute PageRank
        adj = network.adjacency()
        node_names = list(player_map.keys())
        centrality = self.pagerank(adj, node_names)

        # Assign centrality and flag key/critical players
        for name, score in centrality.items():
            if name in player_map:
                p = player_map[name]
                p.centrality = score
                p.is_key = score > _KEY_NODE_CENTRALITY
                p.is_critical = score > _CRITICAL_NODE_CENTRALITY
                if p.is_critical:
                    logger.info(
                        "Critical player detected: %s (centrality=%.4f)", name, score
                    )
                elif p.is_key:
                    logger.debug(
                        "Key player detected: %s (centrality=%.4f)", name, score
                    )

        return network

    def pagerank(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
    ) -> Dict[str, float]:
        """
        Compute PageRank centrality from adjacency dict.

        Algorithm:
          1. Init all scores = 1/N.
          2. At each iteration:
               new_score[v] = (1-d)/N + d × sum(score[u] × adj[u][v] /
                                                 out_weight[u]
                                                 for u in predecessors[v])
          3. Normalise so scores sum to 1.
          4. Converge when max|new - old| < tol.
        Returns {node_name: centrality_score}.
        """
        n = len(nodes)
        if n == 0:
            return {}

        d = _PAGERANK_DAMPING
        teleport = (1.0 - d) / n

        # Initial uniform scores
        scores: Dict[str, float] = {name: 1.0 / n for name in nodes}

        # Pre-compute total out-weight per source node (sum of edge weights)
        out_weight: Dict[str, float] = {}
        for src, targets in adj.items():
            out_weight[src] = sum(targets.values())

        # Build reverse adjacency: predecessors[v] = list of u that point to v
        predecessors: Dict[str, List[str]] = {name: [] for name in nodes}
        for src, targets in adj.items():
            if src not in predecessors:
                predecessors[src] = []
            for tgt in targets:
                if tgt in predecessors:
                    predecessors[tgt].append(src)

        node_set = set(nodes)

        for iteration in range(_PAGERANK_ITERATIONS):
            new_scores: Dict[str, float] = {}

            for v in nodes:
                # Dangling nodes (no predecessors pointing to them) get only
                # the teleport contribution from those with zero out-weight.
                rank_sum = 0.0
                for u in predecessors.get(v, []):
                    if u in node_set:
                        w_uv = adj.get(u, {}).get(v, 0.0)
                        ow = out_weight.get(u, 0.0)
                        if ow > 0.0:
                            rank_sum += scores[u] * w_uv / ow

                new_scores[v] = teleport + d * rank_sum

            # Normalise
            total = sum(new_scores.values())
            if total > 0.0:
                for name in nodes:
                    new_scores[name] /= total

            # Check convergence
            max_delta = max(abs(new_scores[name] - scores[name]) for name in nodes)
            scores = new_scores

            if max_delta < _PAGERANK_TOL:
                logger.debug("PageRank converged at iteration %d", iteration + 1)
                break
        else:
            logger.debug(
                "PageRank reached max iterations (%d) without convergence",
                _PAGERANK_ITERATIONS,
            )

        return scores

    def simulate_absence(
        self,
        network: TeamNetwork,
        absent_player_names: List[str],
    ) -> SynergyImpact:
        """
        Compute the impact of removing given players from the network.

        Steps:
          1. Sum centralities of absent players.
          2. Compute network_disruption = sum_centralities (capped at 1.0).
          3. lambda_multiplier = max(0.40, (1 - network_disruption) ** 1.5)
          4. xg_delta = -(direct_xg_lost + network_disruption × 0.30 ×
                          network.total_xg)
          5. Classify impact level.
          6. Build description string listing absent players and their
             centralities.
        """
        unknown = [n for n in absent_player_names if n not in network.players]
        if unknown:
            logger.warning(
                "simulate_absence: unknown players %s in team %s",
                unknown,
                network.team,
            )

        # Step 1 – centralise sum and direct xG lost
        centrality_sum = 0.0
        direct_xg_lost = 0.0
        player_details: List[str] = []

        for name in absent_player_names:
            if name in network.players:
                p = network.players[name]
                centrality_sum += p.centrality
                direct_xg_lost += p.xg_direct
                player_details.append(
                    f"{name} (centrality={p.centrality:.4f}, xG={p.xg_direct:.3f})"
                )
            else:
                player_details.append(f"{name} (not in network)")

        # Step 2 – disruption capped at 1.0
        network_disruption = min(centrality_sum, 1.0)

        # Step 3 – lambda multiplier (non-linear)
        lambda_multiplier = max(0.40, (1.0 - network_disruption) ** 1.5)

        # Step 4 – xG delta
        network_effect = network_disruption * 0.30 * network.total_xg
        xg_delta = -(direct_xg_lost + network_effect)

        # Step 5 – classify impact
        impact_level = self._impact_level(network_disruption)

        # Step 6 – description
        absent_str = ", ".join(player_details) if player_details else "none"
        description = (
            f"Team {network.team} loses: {absent_str}. "
            f"Network disruption={network_disruption:.4f}. "
            f"Lambda multiplier={lambda_multiplier:.4f}. "
            f"Estimated xG delta={xg_delta:+.4f}. "
            f"Impact level: {impact_level}."
        )

        logger.info(
            "simulate_absence [%s] absent=%s disruption=%.4f λ=%.4f impact=%s",
            network.team,
            absent_player_names,
            network_disruption,
            lambda_multiplier,
            impact_level,
        )

        return SynergyImpact(
            team=network.team,
            absent_players=list(absent_player_names),
            original_centrality_sum=centrality_sum,
            lambda_multiplier=lambda_multiplier,
            xg_delta=xg_delta,
            network_disruption=network_disruption,
            impact_level=impact_level,
            description=description,
        )

    def find_key_synergies(self, network: TeamNetwork) -> List[Tuple[str, str, float]]:
        """
        Find the top 5 most dangerous pass combinations (edges by
        dangerous_count).
        Return list of (from_player, to_player, dangerous_passes) sorted
        descending.
        """
        sorted_edges = sorted(
            network.edges,
            key=lambda e: e.dangerous_count,
            reverse=True,
        )
        result: List[Tuple[str, str, float]] = [
            (e.from_player, e.to_player, float(e.dangerous_count))
            for e in sorted_edges[:5]
        ]
        logger.debug(
            "find_key_synergies [%s]: top %d synergies found",
            network.team,
            len(result),
        )
        return result

    def team_summary(self, network: TeamNetwork) -> str:
        """
        Return ASCII table: Player | Position | Centrality | xG | xA | Key?
        Sorted by centrality descending.
        """
        players = sorted(
            network.players.values(),
            key=lambda p: p.centrality,
            reverse=True,
        )

        # Column headers
        col_player = "Player"
        col_position = "Position"
        col_centrality = "Centrality"
        col_xg = "xG"
        col_xa = "xA"
        col_key = "Key?"

        # Compute column widths based on data
        w_player = max(len(col_player), max((len(p.name) for p in players), default=0))
        w_position = max(
            len(col_position), max((len(p.position) for p in players), default=0)
        )
        w_centrality = max(len(col_centrality), 10)
        w_xg = max(len(col_xg), 7)
        w_xa = max(len(col_xa), 7)
        w_key = max(len(col_key), 8)

        sep = (
            "+"
            + "-" * (w_player + 2)
            + "+"
            + "-" * (w_position + 2)
            + "+"
            + "-" * (w_centrality + 2)
            + "+"
            + "-" * (w_xg + 2)
            + "+"
            + "-" * (w_xa + 2)
            + "+"
            + "-" * (w_key + 2)
            + "+"
        )

        def row(player, pos, cent, xg, xa, key_str):
            return (
                f"| {player:<{w_player}} "
                f"| {pos:<{w_position}} "
                f"| {cent:>{w_centrality}} "
                f"| {xg:>{w_xg}} "
                f"| {xa:>{w_xa}} "
                f"| {key_str:<{w_key}} |"
            )

        header = row(
            col_player,
            col_position,
            col_centrality,
            col_xg,
            col_xa,
            col_key,
        )

        lines: List[str] = [
            f"Team Network Summary — {network.team}",
            f"Total xG: {network.total_xg:.3f}  |  Total xA: {network.total_xa:.3f}  |  Players: {len(players)}",
            sep,
            header,
            sep,
        ]

        for p in players:
            if p.is_critical:
                key_str = "CRITICAL"
            elif p.is_key:
                key_str = "KEY"
            else:
                key_str = "-"
            lines.append(
                row(
                    p.name,
                    p.position,
                    f"{p.centrality:.6f}",
                    f"{p.xg_direct:.3f}",
                    f"{p.xa_direct:.3f}",
                    key_str,
                )
            )

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _impact_level(self, disruption: float) -> str:
        """CRITICAL > 0.40, MAJOR > 0.25, MINOR > 0.10, else NONE."""
        if disruption > 0.40:
            return "CRITICAL"
        if disruption > 0.25:
            return "MAJOR"
        if disruption > 0.10:
            return "MINOR"
        return "NONE"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def build_network_from_dicts(
    team: str,
    player_dicts: List[dict],
    edge_dicts: List[dict],
) -> TeamNetwork:
    """
    Build a TeamNetwork from plain dicts.

    player_dicts keys: name, position, xg_direct, xa_direct, pass_accuracy
    edge_dicts keys: from_player, to_player, pass_count, dangerous_count
    """
    players: List[PlayerNode] = []
    for pd in player_dicts:
        players.append(
            PlayerNode(
                name=pd["name"],
                position=pd["position"],
                xg_direct=float(pd["xg_direct"]),
                xa_direct=float(pd["xa_direct"]),
                pass_accuracy=float(pd["pass_accuracy"]),
            )
        )

    edges: List[PassEdge] = []
    for ed in edge_dicts:
        edges.append(
            PassEdge(
                from_player=ed["from_player"],
                to_player=ed["to_player"],
                pass_count=int(ed["pass_count"]),
                dangerous_count=int(ed["dangerous_count"]),
            )
        )

    engine = NetworkSynergyEngine()
    return engine.build_network(team, players, edges)


def simulate_player_absence(
    team: str,
    player_dicts: List[dict],
    edge_dicts: List[dict],
    absent: List[str],
) -> SynergyImpact:
    """One-shot convenience: build network then simulate absence."""
    network = build_network_from_dicts(team, player_dicts, edge_dicts)
    engine = NetworkSynergyEngine()
    return engine.simulate_absence(network, absent)
