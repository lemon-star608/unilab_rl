from __future__ import annotations

import math
from typing import Any

import pytest

from uni_rl.logging.metric_schema import METRIC_SCHEMA_VERSION
from uni_rl.logging.offpolicy import OffPolicyLogger
from uni_rl.logging.runtime_manifest_schema import (
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    validate_runtime_manifest,
)


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "inference_ring_capacity": 1,
        "collector_metrics_interval": 100,
        "collector_backend_device": "cuda:0",
        "inference_flight": {
            "queue_depth": 0,
            "publication_lag": 0,
            "max_in_flight": 1,
            "max_publication_lag": 1,
        },
        "replay_ingress": {
            "ingress_depth": 2,
            "ingress_slot_rows": 1024,
            "published_sequence": 64,
            "release_sequence": 64,
            "occupancy": 0,
            "high_water_occupancy": 1,
            "backpressure_waits": 0,
            "backpressure_wait_s": 0.0,
            "early_returns": 0,
            "dropped_batches": 0,
            "closed_returns": 0,
            "stop_returns": 0,
        },
        "inference_memory_budget": {"ipc_event_count": 3},
        "shutdown": {"classification": "learner_failure", "opaque": object()},
    }


def test_valid_completed_manifest_passes() -> None:
    validate_runtime_manifest(_manifest(), completed=True)


def test_collector_flight_diagnostics_can_add_experimental_fields() -> None:
    manifest = _manifest()
    manifest["inference_flight"].update(
        {
            "action_backlog": 0,
            "max_action_backlog": 1,
            "in_flight": 0,
            "wait_time_ms": 4.5,
        }
    )

    validate_runtime_manifest(manifest, completed=True)


def test_early_failure_manifest_only_requires_schema_version() -> None:
    validate_runtime_manifest({"schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION})


def test_completed_manifest_allows_null_backend_device() -> None:
    manifest = _manifest()
    manifest["collector_backend_device"] = None

    validate_runtime_manifest(manifest, completed=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.pop("schema_version"), "schema_version must be an integer"),
        (lambda manifest: manifest.update(schema_version=2), "unsupported runtime_manifest"),
        (
            lambda manifest: manifest.update(schema_version=True),
            "schema_version must be an integer",
        ),
        (lambda manifest: manifest.update(inference_ring_capacity=0), "positive integer"),
        (lambda manifest: manifest.update(collector_metrics_interval=True), "positive integer"),
        (lambda manifest: manifest.update(collector_backend_device=3), "non-empty string or null"),
        (
            lambda manifest: manifest["inference_flight"].update(queue_depth=-1),
            "non-negative integer",
        ),
        (
            lambda manifest: manifest["replay_ingress"].update(published_sequence=True),
            "non-negative integer",
        ),
        (
            lambda manifest: manifest["replay_ingress"].update(backpressure_wait_s=math.inf),
            "finite",
        ),
        (
            lambda manifest: manifest["inference_memory_budget"].update(ipc_event_count=-1),
            "non-negative integer",
        ),
    ],
)
def test_stable_fields_fail_closed(
    mutation: Any,
    message: str,
) -> None:
    manifest = _manifest()
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        validate_runtime_manifest(manifest, completed=True)


@pytest.mark.parametrize(
    "field",
    [
        "inference_ring_capacity",
        "collector_metrics_interval",
        "collector_backend_device",
        "inference_flight",
        "replay_ingress",
    ],
)
def test_completed_manifest_requires_stable_lifecycle_fields(field: str) -> None:
    manifest = _manifest()
    manifest.pop(field)

    with pytest.raises(ValueError, match=f"{field}.*required|must be a positive integer"):
        validate_runtime_manifest(manifest, completed=True)


def test_shutdown_is_experimental_and_opaque() -> None:
    manifest = _manifest()
    manifest["shutdown"] = {"classification": "collector_failure", "arbitrary": object()}

    validate_runtime_manifest(manifest, completed=True)


def _backend_diagnostics() -> dict[str, Any]:
    return {
        "available": True,
        "backend_type": "mjwarp",
        "diagnostics": {
            "cuda_graph": {"requested": True, "enabled": True, "disable_reason": None},
            "selected_reset_sensor_refresh": {
                "requested": True,
                "enabled": False,
                "disable_reason": "not requested",
            },
        },
    }


def test_backend_diagnostics_available_passes() -> None:
    manifest = _manifest()
    manifest["backend_diagnostics"] = _backend_diagnostics()

    validate_runtime_manifest(manifest, completed=True)


def test_backend_diagnostics_unavailable_requires_explicit_reason() -> None:
    manifest = _manifest()
    manifest["backend_diagnostics"] = {
        "available": False,
        "unavailable_reason": "env does not expose backend runtime diagnostics",
    }

    validate_runtime_manifest(manifest, completed=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda section: section.update(available="yes"), "available must be a bool"),
        (
            lambda section: section.update(backend_type=""),
            "backend_type must be a non-empty string",
        ),
        (lambda section: section.pop("backend_type"), "backend_type is required"),
        (lambda section: section.pop("diagnostics"), "diagnostics must be a mapping"),
        (
            lambda section: section["diagnostics"]["cuda_graph"].update(enabled=1),
            "enabled must be a bool",
        ),
        (
            lambda section: section["diagnostics"]["cuda_graph"].update(disable_reason="x"),
            "must not declare a disable reason",
        ),
        (
            lambda section: section["diagnostics"]["cuda_graph"].update(
                enabled=False, disable_reason=None
            ),
            "must declare a disable reason",
        ),
        (
            lambda section: section.update(
                available=False, unavailable_reason=None, backend_type="mjwarp"
            ),
            "unavailable_reason must be a non-empty string",
        ),
    ],
)
def test_backend_diagnostics_fail_closed(mutation: Any, message: str) -> None:
    manifest = _manifest()
    section = _backend_diagnostics()
    mutation(section)
    manifest["backend_diagnostics"] = section

    with pytest.raises(ValueError, match=message):
        validate_runtime_manifest(manifest, completed=True)


def test_backend_diagnostics_rejects_non_mapping_section() -> None:
    manifest = _manifest()
    manifest["backend_diagnostics"] = "mjwarp"

    with pytest.raises(ValueError, match="backend_diagnostics must be a mapping"):
        validate_runtime_manifest(manifest, completed=True)


def test_metric_schema_version_is_separate_from_runtime_manifest_version() -> None:
    assert METRIC_SCHEMA_VERSION == 1
    assert RUNTIME_MANIFEST_SCHEMA_VERSION == 1
    assert METRIC_SCHEMA_VERSION != "1"


def test_logger_rejects_manifest_schema_version_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OffPolicyLogger, "__init__", lambda self: None)
    logger = OffPolicyLogger()
    logger._runtime_manifest = {"schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION}

    logger.update_runtime_manifest({"collector_tensor_native": True})

    with pytest.raises(ValueError, match="unsupported runtime_manifest.schema_version"):
        logger.update_runtime_manifest({"schema_version": 2})
