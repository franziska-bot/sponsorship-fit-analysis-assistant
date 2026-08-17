"""Per-run timing breakdown across the three specialist agents (search,
analysis, fit) plus the outreach/rejection step, so main.py can show a
"where did the time go" panel and the orchestrator's logs can flag a
bottleneck agent.
"""


class PerformanceMonitor:
    def __init__(self):
        self._durations_ms: dict[str, list[float]] = {}

    def record_agent_execution(self, agent_name: str, duration_ms: float) -> None:
        self._durations_ms.setdefault(agent_name, []).append(duration_ms)

    def get_agent_average(self, agent_name: str) -> float:
        durations = self._durations_ms.get(agent_name, [])
        return sum(durations) / len(durations) if durations else 0.0

    def get_agent_total(self, agent_name: str) -> float:
        return sum(self._durations_ms.get(agent_name, []))

    def get_slowest_agent(self) -> str | None:
        """Ranks by total time spent (not average), since an agent called
        once for 3s is a bigger bottleneck than one called three times for
        1.5s each on average."""
        if not self._durations_ms:
            return None
        return max(self._durations_ms, key=self.get_agent_total)

    def report(self) -> None:
        for agent_name in self._durations_ms:
            print(
                f"{agent_name}: avg={self.get_agent_average(agent_name):.0f}ms "
                f"total={self.get_agent_total(agent_name):.0f}ms "
                f"calls={len(self._durations_ms[agent_name])}"
            )
        slowest = self.get_slowest_agent()
        if slowest:
            print(f"Slowest agent (bottleneck): {slowest}")

    def to_dict(self) -> dict:
        """One `<agent>_avg`/`<agent>_total`/`<agent>_count` triple per
        recorded agent, plus `total_time` (sum of all agents' totals — equal
        to the full pipeline's wall time, since every graph node's duration
        is attributed to exactly one agent bucket) and `slowest_agent`."""
        metrics: dict = {}
        total_time = 0.0
        for agent_name, durations in self._durations_ms.items():
            metrics[f"{agent_name}_avg"] = self.get_agent_average(agent_name)
            metrics[f"{agent_name}_total"] = self.get_agent_total(agent_name)
            metrics[f"{agent_name}_count"] = len(durations)
            total_time += sum(durations)
        metrics["total_time"] = total_time
        metrics["slowest_agent"] = self.get_slowest_agent()
        return metrics
