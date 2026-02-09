"""Tests for Docker configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Dockerfile (API)
# ------------------------------------------------------------------


class TestDockerfile:
    @pytest.fixture
    def content(self):
        return (ROOT / "Dockerfile").read_text()

    def test_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_multi_stage_build(self, content):
        assert "AS builder" in content

    def test_from_python(self, content):
        assert "python:3.11-slim" in content

    def test_copies_requirements(self, content):
        assert "COPY requirements.txt" in content

    def test_pip_wheel_stage(self, content):
        assert "pip wheel" in content

    def test_copies_wheels(self, content):
        assert "COPY --from=builder /wheels" in content

    def test_copies_src(self, content):
        assert "COPY src/ src/" in content

    def test_copies_configs(self, content):
        assert "COPY configs/ configs/" in content

    def test_exposes_8000(self, content):
        assert "EXPOSE 8000" in content

    def test_healthcheck(self, content):
        assert "HEALTHCHECK" in content
        assert "curl" in content
        assert "/health" in content

    def test_cmd_uvicorn(self, content):
        assert "uvicorn" in content
        assert "src.api.app:app" in content

    def test_no_cache_dir(self, content):
        assert "--no-cache-dir" in content


# ------------------------------------------------------------------
# Dockerfile.dashboard
# ------------------------------------------------------------------


class TestDockerfileDashboard:
    @pytest.fixture
    def content(self):
        return (ROOT / "Dockerfile.dashboard").read_text()

    def test_exists(self):
        assert (ROOT / "Dockerfile.dashboard").exists()

    def test_multi_stage_build(self, content):
        assert "AS builder" in content

    def test_exposes_8501(self, content):
        assert "EXPOSE 8501" in content

    def test_cmd_streamlit(self, content):
        assert "streamlit" in content
        assert "src/dashboard/app.py" in content

    def test_headless(self, content):
        assert "headless=true" in content

    def test_healthcheck(self, content):
        assert "HEALTHCHECK" in content


# ------------------------------------------------------------------
# Dockerfile.simulator
# ------------------------------------------------------------------


class TestDockerfileSimulator:
    @pytest.fixture
    def content(self):
        return (ROOT / "Dockerfile.simulator").read_text()

    def test_exists(self):
        assert (ROOT / "Dockerfile.simulator").exists()

    def test_multi_stage_build(self, content):
        assert "AS builder" in content

    def test_cmd_python(self, content):
        assert "python" in content
        assert "src.streaming.run_simulator" in content

    def test_copies_src(self, content):
        assert "COPY src/ src/" in content


# ------------------------------------------------------------------
# docker-compose.yml
# ------------------------------------------------------------------


class TestDockerCompose:
    @pytest.fixture
    def config(self):
        return yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    def test_exists(self):
        assert (ROOT / "docker-compose.yml").exists()

    def test_valid_yaml(self):
        data = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        assert isinstance(data, dict)

    def test_has_services(self, config):
        assert "services" in config

    def test_api_service(self, config):
        assert "api" in config["services"]

    def test_dashboard_service(self, config):
        assert "dashboard" in config["services"]

    def test_simulator_service(self, config):
        assert "simulator" in config["services"]

    def test_api_port(self, config):
        api = config["services"]["api"]
        assert "8000:8000" in api["ports"]

    def test_dashboard_port(self, config):
        dash = config["services"]["dashboard"]
        assert "8501:8501" in dash["ports"]

    def test_api_healthcheck(self, config):
        api = config["services"]["api"]
        assert "healthcheck" in api
        hc = api["healthcheck"]
        assert any("/health" in str(t) for t in hc["test"])

    def test_dashboard_depends_on_api(self, config):
        dash = config["services"]["dashboard"]
        assert "api" in dash["depends_on"]

    def test_simulator_depends_on_api(self, config):
        sim = config["services"]["simulator"]
        assert "api" in sim["depends_on"]

    def test_dashboard_api_url_env(self, config):
        dash = config["services"]["dashboard"]
        env = dash.get("environment", [])
        assert any("API_URL" in str(e) for e in env)

    def test_simulator_stream_rate_env(self, config):
        sim = config["services"]["simulator"]
        env = sim.get("environment", [])
        assert any("STREAM_RATE" in str(e) for e in env)

    def test_api_volumes(self, config):
        api = config["services"]["api"]
        volumes = api.get("volumes", [])
        volume_str = " ".join(str(v) for v in volumes)
        assert "data" in volume_str

    def test_dashboard_uses_custom_dockerfile(self, config):
        dash = config["services"]["dashboard"]
        build = dash.get("build", {})
        assert build.get("dockerfile") == "Dockerfile.dashboard"

    def test_simulator_uses_custom_dockerfile(self, config):
        sim = config["services"]["simulator"]
        build = sim.get("build", {})
        assert build.get("dockerfile") == "Dockerfile.simulator"

    def test_dashboard_depends_on_healthy(self, config):
        dash = config["services"]["dashboard"]
        dep = dash["depends_on"]["api"]
        assert dep["condition"] == "service_healthy"

    def test_simulator_depends_on_healthy(self, config):
        sim = config["services"]["simulator"]
        dep = sim["depends_on"]["api"]
        assert dep["condition"] == "service_healthy"


# ------------------------------------------------------------------
# .dockerignore
# ------------------------------------------------------------------


class TestDockerignore:
    def test_exists(self):
        assert (ROOT / ".dockerignore").exists()

    def test_ignores_pycache(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content

    def test_ignores_git(self):
        content = (ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_ignores_tests(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "tests" in content

    def test_ignores_venv(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "venv" in content


# ------------------------------------------------------------------
# Makefile
# ------------------------------------------------------------------


class TestMakefile:
    @pytest.fixture
    def content(self):
        return (ROOT / "Makefile").read_text()

    def test_exists(self):
        assert (ROOT / "Makefile").exists()

    def test_docker_up_target(self, content):
        assert "docker-up:" in content
        assert "docker-compose up" in content

    def test_docker_down_target(self, content):
        assert "docker-down:" in content
        assert "docker-compose down" in content

    def test_docker_logs_target(self, content):
        assert "docker-logs:" in content
        assert "docker-compose logs" in content

    def test_docker_build_target(self, content):
        assert "docker:" in content
        assert "docker build" in content

    def test_phony_includes_docker_targets(self, content):
        phony_line = [l for l in content.split("\n") if ".PHONY" in l][0]
        assert "docker-up" in phony_line
        assert "docker-down" in phony_line
        assert "docker-logs" in phony_line


# ------------------------------------------------------------------
# requirements.txt
# ------------------------------------------------------------------


class TestRequirements:
    @pytest.fixture
    def content(self):
        return (ROOT / "requirements.txt").read_text()

    def test_exists(self):
        assert (ROOT / "requirements.txt").exists()

    def test_fastapi(self, content):
        assert "fastapi" in content

    def test_uvicorn(self, content):
        assert "uvicorn" in content

    def test_streamlit(self, content):
        assert "streamlit" in content

    def test_scikit_learn(self, content):
        assert "scikit-learn" in content

    def test_xgboost(self, content):
        assert "xgboost" in content

    def test_shap(self, content):
        assert "shap" in content

    def test_aiosqlite(self, content):
        assert "aiosqlite" in content

    def test_scipy(self, content):
        assert "scipy" in content


# ------------------------------------------------------------------
# run_simulator entry point
# ------------------------------------------------------------------


class TestRunSimulator:
    def test_module_exists(self):
        assert (ROOT / "src" / "streaming" / "run_simulator.py").exists()

    def test_importable(self):
        from src.streaming.run_simulator import main
        import asyncio
        assert asyncio.iscoroutinefunction(main)
