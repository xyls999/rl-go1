"""Isolated V132.2 first-episode gate collector."""


class FirstEpisodeCollector:
    def __init__(self, num_envs: int, trace_required: bool):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.trace_required = trace_required
        self._records: dict[int, dict] = {}
        self._trace: dict[int, dict] = {}

    @property
    def complete(self) -> bool:
        return len(self._records) == self.num_envs

    def accept(self, episode_rows: list[dict], trace_rows: list[dict] | None) -> None:
        def ids(rows: list[dict], label: str) -> list[int]:
            result = []
            for row in rows:
                value = row.get("env_id")
                if type(value) is not int or not 0 <= value < self.num_envs:
                    raise RuntimeError(f"{label} env_id outside [0, {self.num_envs})")
                result.append(value)
            if len(set(result)) != len(result):
                raise RuntimeError(f"duplicate {label} env_id in done batch")
            return result

        episode_ids = ids(episode_rows, "episode")
        if self.trace_required and trace_rows is None:
            raise RuntimeError("episode trace missing for done batch")
        if trace_rows is not None:
            trace_ids = ids(trace_rows, "trace")
            if set(episode_ids) != set(trace_ids):
                raise RuntimeError("episode trace env_ids do not match done batch")
            for row in trace_rows:
                if self.trace_required and not {"required_gap_count", "credited_gap_count"} <= row.keys():
                    raise RuntimeError("episode trace missing required gap counts")
            traced = {row["env_id"]: row for row in trace_rows}
        else:
            traced = {}
        for row in episode_rows:
            env_id = row["env_id"]
            if env_id in self._records:
                continue
            self._records[env_id] = row
            if self.trace_required:
                self._trace[env_id] = {**traced[env_id], "episode_index": 0}

    def finish(self) -> tuple[list[dict], list[dict]]:
        missing = set(range(self.num_envs)) - self._records.keys()
        if missing:
            raise RuntimeError(f"missing first episode for env_ids {sorted(missing)}")
        if self.trace_required and self._trace.keys() != self._records.keys():
            raise RuntimeError("episode trace missing or mismatched first episodes")
        return ([self._records[i] for i in range(self.num_envs)],
                [self._trace[i] for i in range(self.num_envs)] if self.trace_required else [])
