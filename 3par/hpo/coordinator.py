"""Fixed-budget joint search with crossover and local refinement."""

import itertools
import json
import random

from .models import WIDTH_PARAMETERS, active_space, default_config, inspect_candidate


STAGE_SLOTS = {"baseline": 1, "global": 16, "crossover": 9, "local": 6}
TRAINING_PARAMETERS = ("lr", "weight_decay", "batch_size", "loss_profile")
GLOBAL_REVIEW_STAGES = {5: "structural_parameters", 10: "training_parameters",
                        16: "training_parameters"}


def slot_priority(position):
    """Alternate model and training exploration so each full stage is balanced."""
    return "model" if position % 2 else "training"


def record_complete(record):
    """A dual-seed slot is eligible only after both screening fits succeeded."""
    return record.get("screening_status", record["result"]["status"]) == "COMPLETE"


def record_score(record):
    """Rank by the screening mean, retaining readable historical records."""
    return record.get("screening_val_rmse_mean", record["result"].get("val_rmse"))


def coordinate_order(task, settings):
    """Order fields consistently for joint designs, crossover, and local moves."""
    active = active_space(task, settings)
    backbone_widths = [WIDTH_PARAMETERS[module][0] for module in task["spec"]["B"]
                       if module in WIDTH_PARAMETERS and WIDTH_PARAMETERS[module][0] in active]
    head_width = WIDTH_PARAMETERS.get(task["spec"]["C"], (None,))[0]
    order = [*TRAINING_PARAMETERS, "time_lag", "patch_profile", *backbone_widths,
             "lstm_layers", "tcn_kernel", head_width]
    return list(dict.fromkeys(name for name in order if name in active))


def probe_values(parameter, settings):
    """Keep the former broad values as preferred levels in the joint design."""
    key = "width_scale" if parameter.endswith("_width_scale") else parameter
    values = settings["search"]["probe_values"][key]
    allowed = settings["space"]["width_scales"] if key == "width_scale" else settings["space"][parameter]
    if isinstance(allowed, dict):
        if any(not allowed["low"] <= value <= allowed["high"] for value in values):
            raise ValueError(f"Probe values for {parameter} are outside the search range")
    elif any(value not in allowed for value in values):
        raise ValueError(f"Probe values for {parameter} are not in the search space")
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError(f"{parameter} must define two distinct preferred values")
    return list(values)


class Coordinator:
    """Replayable 1-baseline + 16-joint + 9-crossover + 6-local search."""

    def __init__(self, task, settings, root, tuning_allowed=True):
        self.task, self.settings, self.root = task, settings, root
        self.space = active_space(task, settings)
        self.defaults = default_config(task, settings)
        self.parameters = coordinate_order(task, settings)
        self.training_parameters = [name for name in TRAINING_PARAMETERS if name in self.parameters]
        self.model_parameters = [name for name in self.parameters if name not in TRAINING_PARAMETERS]
        self.probes = {name: probe_values(name, settings) for name in self.parameters}
        self.max_slots = settings["search"]["trials"]
        self.screening_budget = self.max_slots
        remaining = max(0, self.max_slots - STAGE_SLOTS["baseline"])
        self.stage_slots = {"baseline": min(self.max_slots, STAGE_SLOTS["baseline"])}
        for stage in ("global", "crossover", "local"):
            count = min(remaining, STAGE_SLOTS[stage])
            self.stage_slots[stage] = count
            remaining -= count
        self.tuning_allowed = tuning_allowed
        self.records = []
        self.decisions = []
        self.incumbent = None
        self.baseline = None
        self.screening_complete = False
        self.exhausted_stages = {}
        self._validity = {}
        self.global_candidates = self._build_global_candidates()

    @staticmethod
    def _key(config):
        return json.dumps(config, sort_keys=True)

    @property
    def seen(self):
        return {self._key(record["executed_config"])
                for record in self.records
                if record.get("executed_config") is not None and record.get("phase") != "confirmation"}

    def _preferred_values(self, parameter):
        allowed = list(self.space[parameter])
        preferred = [*self.probes[parameter],
                     *(value for value in allowed
                       if value != self.defaults[parameter] and value not in self.probes[parameter])]
        return list(dict.fromkeys(preferred))

    def _build_global_candidates(self):
        """Cover every non-default level, retaining all pairs as validity fallbacks."""
        max_width = min(4, len(self.parameters))
        rng = random.Random(2026)  # Independent of training and preflight RNG state.
        candidates, used = [], set()
        for width in range(2, max_width + 1):
            for subset in itertools.combinations(self.parameters, width):
                levels = [self._preferred_values(name) for name in subset]
                if width == 2:
                    designs = itertools.product(*levels)
                else:
                    designs = []
                    for _ in range(2):
                        shuffled = [rng.sample(values, len(values)) for values in levels]
                        designs.extend(tuple(values[index % len(values)] for values in shuffled)
                                       for index in range(max(map(len, levels))))
                for values in designs:
                    config = {**self.defaults, **dict(zip(subset, values))}
                    key = self._key(config)
                    if key not in used:
                        used.add(key)
                        candidates.append(config)
        grouped = {"model": [], "training": []}
        for config in candidates:
            priority = self._config_priority(config)
            grouped[priority].append(config)
        return grouped

    def _config_priority(self, config):
        model_changes = sum(config[name] != self.defaults[name]
                            for name in self.model_parameters)
        training_changes = sum(config[name] != self.defaults[name]
                               for name in self.training_parameters)
        return "model" if model_changes >= training_changes and model_changes else "training"

    def _stage(self):
        if self.baseline is None:
            return "baseline"
        for stage in ("global", "crossover", "local"):
            if stage not in self.exhausted_stages and self._stage_index(stage) < self.stage_slots[stage]:
                return stage
        return None

    def _stage_index(self, stage):
        return sum(record.get("search_round") == stage for record in self.records)

    @staticmethod
    def _rank(records, limit):
        complete = [record for record in records if record_complete(record)]
        return sorted(complete, key=lambda record: (record_score(record), record["slot"]))[:limit]

    def _global_proposal(self, priority):
        observed = self._editable_values()
        pairs = {name: list(itertools.combinations(fields, 2))
                 for name, fields in (("model", self.model_parameters),
                                      ("training", self.training_parameters))}
        tested_pairs = {(left, config[left], right, config[right])
                        for record in self.records
                        if (config := record.get("executed_config")) is not None
                        for left, right in [*pairs["model"], *pairs["training"]]}

        def coverage(config, category):
            fields = self.model_parameters if category == "model" else self.training_parameters
            fresh = [(name, config[name]) for name in fields
                     if config[name] not in observed[name]]
            new_probes = sum(value in self.probes[name] for name, value in fresh)
            other_fields = self.training_parameters if category == "model" else self.model_parameters
            other_changes = sum(config[name] != self.defaults[name] for name in other_fields)
            new_pairs = sum((left, config[left], right, config[right]) not in tested_pairs
                            for left, right in pairs[category])
            return (-new_probes, -len(fresh), other_changes, -new_pairs)

        priorities = (priority, "training" if priority == "model" else "model")
        return next((config for name in priorities
                     for config in sorted(self.global_candidates[name], key=lambda item: coverage(item, name))
                     if self._key(config) not in self.seen and self._valid(config)), None)

    def _valid(self, config):
        key = self._key(config)
        if key not in self._validity:
            try:
                inspect_candidate(self.task, config, self.settings, self.root,
                                  check_forward=True, batch_sizes=(1,))
                self._validity[key] = True
            except (ValueError, RuntimeError, FloatingPointError):
                self._validity[key] = False
        return self._validity[key]

    def _global_elites(self):
        records = [record for record in self.records
                   if record.get("search_round") in {"baseline", "global"}]
        return self._rank(records, 4)

    def _crossover_elites(self, limit=4):
        records = [record for record in self.records
                   if record.get("search_round") in {"baseline", "global", "crossover"}]
        return self._rank(records, limit)

    def _crossover_candidates(self, elites=None, new_parent=None):
        elites = self._crossover_elites() if elites is None else elites
        if len(elites) < 2:
            return
        pairs = [(left, right) for left, right in itertools.combinations(elites, 2)
                 if new_parent is None or new_parent in (left["slot"], right["slot"])]
        masks = sorted(range(1, 2 ** len(self.parameters) - 1),
                       key=lambda mask: (abs(mask.bit_count() * 2 - len(self.parameters)), mask))
        used = set()
        # Refresh the elite pool after each result; interleave its parent pairs.
        for mask in masks:
            for left, right in pairs:
                config = {name: (left["executed_config"][name] if mask & (1 << index)
                                 else right["executed_config"][name])
                          for index, name in enumerate(self.parameters)}
                differing = [index for index, name in enumerate(self.parameters)
                             if left["executed_config"][name] != right["executed_config"][name]]
                contributors = {bool(mask & (1 << index)) for index in differing}
                key = self._key(config)
                if len(contributors) < 2 or key in used:
                    continue
                used.add(key)
                yield config, [left["slot"], right["slot"]]

    def _crossover_proposal(self):
        ranked = self._crossover_elites(limit=None)
        for size in range(min(4, len(ranked)), len(ranked) + 1):
            pool = ranked[:size]
            # Keep the original Top-4 order, then try only pairs involving the new parent.
            new_parent = pool[-1]["slot"] if size > 4 else None
            for config, parents in self._crossover_candidates(pool, new_parent):
                if self._key(config) not in self.seen and self._valid(config):
                    return config, parents, [record["slot"] for record in pool]
        return None, [], []

    def _local_elites(self):
        records = [record for record in self.records
                   if record.get("search_round") in {"baseline", "global", "crossover", "local"}]
        return self._rank(records, 2)

    def _local_candidates(self, parameters):
        elites = self._local_elites()
        candidates, used = [], set()
        # Balance executed directions; use a stable task-specific tie order.
        counts = self._local_direction_counts()
        directions = random.Random(self.task["id"]).sample([-1, 1], 2)
        directions.sort(key=lambda direction: counts[direction])
        for elite in elites:
            for direction in directions:
                for parameter in parameters:
                    allowed = list(self.space[parameter])
                    current = elite["executed_config"][parameter]
                    neighbor_index = allowed.index(current) + direction
                    if not 0 <= neighbor_index < len(allowed):
                        continue
                    config = {**elite["executed_config"], parameter: allowed[neighbor_index]}
                    key = self._key(config)
                    if key in used:
                        continue
                    used.add(key)
                    candidates.append((config, elite["slot"], parameter, allowed[neighbor_index]))
        return candidates

    def _local_direction_counts(self):
        counts = {-1: 0, 1: 0}
        by_slot = {record["slot"]: record for record in self.records}
        for record in self.records:
            if record.get("search_round") != "local" or not record.get("executed_config"):
                continue
            parameter = record["parameter"]
            allowed = list(self.space[parameter])
            before = by_slot[record["base_slot"]]["executed_config"][parameter]
            after = record["executed_config"][parameter]
            delta = allowed.index(after) - allowed.index(before)
            if delta:
                counts[1 if delta > 0 else -1] += 1
        return counts

    def _local_proposal(self, priority):
        used_parameters = {record.get("parameter") for record in self.records
                           if record.get("search_round") == "local"}
        preferred = self.model_parameters if priority == "model" else self.training_parameters
        secondary = self.training_parameters if priority == "model" else self.model_parameters
        parameter_groups = ([name for name in preferred if name not in used_parameters],
                            preferred,
                            [name for name in secondary if name not in used_parameters],
                            secondary)
        for parameters in parameter_groups:
            proposal = next(((config, base_slot, parameter, value)
                             for config, base_slot, parameter, value in self._local_candidates(parameters)
                             if self._key(config) not in self.seen and self._valid(config)), None)
            if proposal is not None:
                return proposal
        return None, None, None, None

    def _editable_values(self):
        values = {name: [] for name in self.parameters}
        for record in self.records:
            config = record.get("executed_config") or {}
            for name in self.parameters:
                if name in config and config[name] not in values[name]:
                    values[name].append(config[name])
        return values

    def propose(self, slot):
        if slot != len(self.records) + 1:
            raise ValueError("Slots must be proposed in sequence")
        stage = self._stage()
        if stage == "baseline":
            return {"config": self.defaults, "source": "default",
                    "phase": "baseline", "search_round": "baseline",
                    "search_priority": "default", "effective_priority": "default",
                    "parameter": None, "probe_value": None, "parent_slots": [],
                    "base_slot": None, "seed": self.settings["training"]["search_train_seed"],
                    "review_stage": None, "editable_parameters": [], "editable_values": {}}
        if not self.tuning_allowed or stage is None:
            self.screening_complete = True
            return None

        parameter = probe_value = base_slot = None
        parent_slots = []
        parent_pool_slots = []
        stage_position = self._stage_index(stage) + 1
        if stage == "global":
            priority = slot_priority(stage_position)
            config, source = self._global_proposal(priority), "global_joint"
            effective_priority = self._config_priority(config) if config is not None else priority
        elif stage == "crossover":
            priority = effective_priority = "joint"
            config, parent_slots, parent_pool_slots = self._crossover_proposal()
            base_slot = parent_slots[0] if parent_slots else None
            source = "expanded_elite_crossover" if len(parent_pool_slots) > 4 else "top4_crossover"
        else:
            priority = slot_priority(stage_position)
            config, base_slot, parameter, probe_value = self._local_proposal(priority)
            effective_priority = "training" if parameter in self.training_parameters else "model"
            parent_slots = [base_slot] if base_slot is not None else []
            source = "top2_local_refinement"
        if config is None:
            self.exhausted_stages[stage] = "no_valid_unseen_candidate"
            return self.propose(slot)

        review_stage = GLOBAL_REVIEW_STAGES.get(stage_position) if stage == "global" else None
        if review_stage:
            review_stage = ("structural_parameters" if effective_priority == "model"
                            else "training_parameters")
        editable = ((self.model_parameters if effective_priority == "model" else self.training_parameters)
                    if review_stage else [])
        return {"config": config, "source": source, "phase": "coordinate",
                "search_round": stage, "parameter": parameter, "probe_value": probe_value,
                "search_priority": priority, "effective_priority": effective_priority,
                "parent_slots": parent_slots, "base_slot": base_slot,
                "parent_pool_slots": parent_pool_slots,
                "parent_pool_size": len(parent_pool_slots),
                "parent_pool_expanded": len(parent_pool_slots) > 4,
                "seed": self.settings["training"]["search_train_seed"],
                "review_stage": review_stage, "editable_parameters": editable,
                "editable_values": self._editable_values() if review_stage else {}}

    def observe(self, record):
        if record["slot"] != len(self.records) + 1:
            raise ValueError("Observed slots are out of sequence")
        phase = record.get("phase")
        if phase not in {"baseline", "coordinate"}:
            raise ValueError(f"Unknown search phase: {phase}")
        if phase == "coordinate":
            for stage in ("global", "crossover", "local"):
                if stage == record.get("search_round"):
                    break
                if self._stage_index(stage) < self.stage_slots[stage]:
                    self.exhausted_stages[stage] = "no_valid_unseen_candidate"
        self.records.append(record)
        if phase == "baseline":
            self.baseline = self.incumbent = record
        elif record_complete(record) and (
                self.incumbent is None
                or not record_complete(self.incumbent)
                or record_score(record) < record_score(self.incumbent)):
            self.incumbent = record
        if phase == "coordinate":
            self.decisions.append({"search_round": record.get("search_round"),
                                   "slot": record["slot"],
                                   "base_slot": record.get("base_slot"),
                                   "parent_slots": record.get("parent_slots", []),
                                   "parent_pool_size": record.get("parent_pool_size"),
                                   "parent_pool_expanded": record.get("parent_pool_expanded", False),
                                   "selected_slot": self.incumbent["slot"],
                                   "accepted": self.incumbent is record,
                                   "reason": ("lowest_validation_rmse"
                                              if record_complete(record)
                                              else "candidate_failed")})
        if not self.tuning_allowed or len(self.records) >= self.screening_budget or self._stage() is None:
            self.screening_complete = True

    @property
    def complete(self):
        if self.baseline is None or not record_complete(self.baseline):
            return False
        if not self.tuning_allowed or len(self.records) >= self.screening_budget or self._stage() is None:
            self.screening_complete = True
        return self.screening_complete

    def trace(self):
        tested = {name: [] for name in self.parameters}
        for record in self.records:
            config = record.get("executed_config") or {}
            for name in self.parameters:
                value = config.get(name)
                if value != self.defaults[name] and value not in tested[name]:
                    tested[name].append(value)
        untested = {name: [value for value in self.space[name]
                           if value != self.defaults[name] and value not in tested[name]]
                    for name in self.parameters}
        completed_by_stage = {stage: sum(record.get("search_round") == stage
                                         for record in self.records)
                              for stage in STAGE_SLOTS}
        priority_plan = {stage: {priority: sum(slot_priority(position) == priority
                                             for position in range(1, self.stage_slots[stage] + 1))
                                 for priority in ("model", "training")}
                         for stage in ("global", "local")}
        return {"method": "fixed_1_16_9_6_balanced_dynamic_joint_crossover_local_top4_default_v5",
                "max_slots": self.max_slots, "screening_budget": self.screening_budget,
                "stage_slot_plan": self.stage_slots, "completed_by_stage": completed_by_stage,
                "model_priority_slots": {stage: counts["model"] for stage, counts in priority_plan.items()},
                "training_priority_slots": {stage: counts["training"] for stage, counts in priority_plan.items()},
                "priority_slot_plan": priority_plan,
                "global_review_stages": dict(GLOBAL_REVIEW_STAGES),
                "confirmation_seeds": self.settings["search"]["confirmation_seeds"],
                "screening_seeds": self.settings["search"].get("screening_seeds", [2026, 2027]),
                "confirmation_strategy": "top4_plus_default_seed2028_then_mean_2026_2027_2028",
                "parameter_order": self.parameters, "preferred_global_values": self.probes,
                "tested_target_values": tested, "untested_target_values": untested,
                "untested_preferred_values": {name: [value for value in self.probes[name]
                                                       if value in untested[name]]
                                               for name in self.parameters},
                "exhausted_stages": dict(self.exhausted_stages),
                "local_direction_counts": {str(direction): count
                                           for direction, count in self._local_direction_counts().items()},
                "global_top4_slots": [record["slot"] for record in self._global_elites()],
                "crossover_top4_slots": [record["slot"] for record in self._crossover_elites()],
                "crossover_pool_expansions": [{"slot": record["slot"],
                                                "parent_slots": record.get("parent_slots", []),
                                                "parent_pool_slots": record.get("parent_pool_slots", []),
                                                "parent_pool_size": record.get("parent_pool_size")}
                                               for record in self.records
                                               if record.get("parent_pool_expanded")],
                "local_top2_slots": [record["slot"] for record in self._local_elites()],
                "tuning_allowed": self.tuning_allowed, "coordinate_decisions": self.decisions,
                "baseline_slot": self.baseline["slot"] if self.baseline else None,
                "challenger_slot": self.incumbent["slot"] if self.incumbent else None,
                "confirmation_slots": [], "completed_slots": len(self.records),
                "complete": self.complete}
