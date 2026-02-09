"""Tests for CI/CD configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# GitHub Actions CI workflow
# ------------------------------------------------------------------


class TestCIWorkflow:
    @pytest.fixture
    def workflow(self):
        return yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )

    def test_workflow_file_exists(self):
        assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_valid_yaml(self):
        data = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert isinstance(data, dict)

    def test_name(self, workflow):
        assert workflow["name"] == "CI"

    def test_trigger_push(self, workflow):
        # YAML parses bare `on:` as boolean True
        triggers = workflow.get("on") or workflow.get(True)
        assert "push" in triggers

    def test_trigger_pull_request(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        assert "pull_request" in triggers

    def test_push_branches(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        branches = triggers["push"]["branches"]
        assert "main" in branches or "master" in branches

    def test_pr_branches(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        branches = triggers["pull_request"]["branches"]
        assert "main" in branches or "master" in branches

    def test_has_jobs(self, workflow):
        assert "jobs" in workflow
        assert len(workflow["jobs"]) >= 3


class TestLintJob:
    @pytest.fixture
    def job(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        return wf["jobs"]["lint"]

    def test_lint_job_exists(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert "lint" in wf["jobs"]

    def test_runs_on_ubuntu(self, job):
        assert "ubuntu" in job["runs-on"]

    def test_uses_checkout(self, job):
        steps_str = str(job["steps"])
        assert "actions/checkout" in steps_str

    def test_uses_setup_python(self, job):
        steps_str = str(job["steps"])
        assert "actions/setup-python" in steps_str

    def test_python_311(self, job):
        steps_str = str(job["steps"])
        assert "3.11" in steps_str

    def test_installs_ruff(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("pip install ruff" in r for r in run_steps)

    def test_runs_ruff_check(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("ruff check" in r for r in run_steps)

    def test_runs_ruff_format_check(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("ruff format" in r and "--check" in r for r in run_steps)


class TestTestJob:
    @pytest.fixture
    def job(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        return wf["jobs"]["test"]

    def test_test_job_exists(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert "test" in wf["jobs"]

    def test_runs_on_ubuntu(self, job):
        assert "ubuntu" in job["runs-on"]

    def test_needs_lint(self, job):
        assert "lint" in job["needs"]

    def test_uses_checkout(self, job):
        steps_str = str(job["steps"])
        assert "actions/checkout" in steps_str

    def test_uses_setup_python(self, job):
        steps_str = str(job["steps"])
        assert "actions/setup-python" in steps_str

    def test_installs_requirements(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("pip install -r requirements.txt" in r for r in run_steps)

    def test_runs_pytest(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("pytest" in r for r in run_steps)

    def test_pytest_coverage(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("--cov=src" in r for r in run_steps)

    def test_coverage_xml(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("--cov-report=xml" in r for r in run_steps)

    def test_codecov_upload(self, job):
        steps_str = str(job["steps"])
        assert "codecov/codecov-action" in steps_str


class TestDockerJob:
    @pytest.fixture
    def job(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        return wf["jobs"]["docker"]

    def test_docker_job_exists(self):
        wf = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )
        assert "docker" in wf["jobs"]

    def test_runs_on_ubuntu(self, job):
        assert "ubuntu" in job["runs-on"]

    def test_needs_test(self, job):
        assert "test" in job["needs"]

    def test_uses_checkout(self, job):
        steps_str = str(job["steps"])
        assert "actions/checkout" in steps_str

    def test_docker_build(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("docker build" in r for r in run_steps)

    def test_docker_compose_config(self, job):
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        assert any("docker-compose config" in r for r in run_steps)


# ------------------------------------------------------------------
# README badges
# ------------------------------------------------------------------


class TestReadmeBadges:
    @pytest.fixture
    def content(self):
        return (ROOT / "README.md").read_text()

    def test_ci_badge(self, content):
        assert "actions/workflows/ci.yml/badge.svg" in content

    def test_coverage_badge(self, content):
        assert "codecov.io" in content

    def test_badges_are_images(self, content):
        # Badges should be markdown images: ![alt](url)
        assert "![CI]" in content
        assert "![Coverage]" in content


# ------------------------------------------------------------------
# Pre-commit configuration
# ------------------------------------------------------------------


class TestPreCommitConfig:
    @pytest.fixture
    def config(self):
        return yaml.safe_load(
            (ROOT / ".pre-commit-config.yaml").read_text()
        )

    def test_file_exists(self):
        assert (ROOT / ".pre-commit-config.yaml").exists()

    def test_valid_yaml(self):
        data = yaml.safe_load(
            (ROOT / ".pre-commit-config.yaml").read_text()
        )
        assert isinstance(data, dict)

    def test_has_repos(self, config):
        assert "repos" in config
        assert len(config["repos"]) >= 2

    def test_ruff_repo(self, config):
        repos_str = str(config["repos"])
        assert "ruff-pre-commit" in repos_str

    def test_ruff_hook(self, config):
        ruff_repos = [
            r for r in config["repos"]
            if "ruff" in str(r.get("repo", ""))
        ]
        assert len(ruff_repos) >= 1
        hooks = ruff_repos[0]["hooks"]
        hook_ids = [h["id"] for h in hooks]
        assert "ruff" in hook_ids

    def test_ruff_format_hook(self, config):
        ruff_repos = [
            r for r in config["repos"]
            if "ruff" in str(r.get("repo", ""))
        ]
        hooks = ruff_repos[0]["hooks"]
        hook_ids = [h["id"] for h in hooks]
        assert "ruff-format" in hook_ids

    def test_local_pytest_hook(self, config):
        local_repos = [
            r for r in config["repos"]
            if r.get("repo") == "local"
        ]
        assert len(local_repos) >= 1
        hooks = local_repos[0]["hooks"]
        hook_ids = [h["id"] for h in hooks]
        assert "pytest" in hook_ids

    def test_pytest_hook_entry(self, config):
        local_repos = [
            r for r in config["repos"]
            if r.get("repo") == "local"
        ]
        hooks = local_repos[0]["hooks"]
        pytest_hook = [h for h in hooks if h["id"] == "pytest"][0]
        assert "pytest" in pytest_hook["entry"]

    def test_pytest_hook_language(self, config):
        local_repos = [
            r for r in config["repos"]
            if r.get("repo") == "local"
        ]
        hooks = local_repos[0]["hooks"]
        pytest_hook = [h for h in hooks if h["id"] == "pytest"][0]
        assert pytest_hook["language"] == "system"

    def test_ruff_rev_specified(self, config):
        ruff_repos = [
            r for r in config["repos"]
            if "ruff" in str(r.get("repo", ""))
        ]
        assert "rev" in ruff_repos[0]
        assert ruff_repos[0]["rev"]  # non-empty


# ------------------------------------------------------------------
# Job dependency chain
# ------------------------------------------------------------------


class TestJobDependencyChain:
    @pytest.fixture
    def workflow(self):
        return yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        )

    def test_lint_has_no_dependencies(self, workflow):
        lint = workflow["jobs"]["lint"]
        assert "needs" not in lint

    def test_test_depends_on_lint(self, workflow):
        test = workflow["jobs"]["test"]
        needs = test["needs"]
        if isinstance(needs, list):
            assert "lint" in needs
        else:
            assert needs == "lint"

    def test_docker_depends_on_test(self, workflow):
        docker = workflow["jobs"]["docker"]
        needs = docker["needs"]
        if isinstance(needs, list):
            assert "test" in needs
        else:
            assert needs == "test"

    def test_pipeline_is_sequential(self, workflow):
        """Verify the full chain: lint -> test -> docker."""
        jobs = workflow["jobs"]
        assert "needs" not in jobs["lint"]
        assert "lint" in str(jobs["test"]["needs"])
        assert "test" in str(jobs["docker"]["needs"])
