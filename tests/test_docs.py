"""Tests for documentation and docstring coverage."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Documentation files exist
# ------------------------------------------------------------------


class TestDocsExist:
    def test_readme_exists(self):
        assert (ROOT / "README.md").exists()

    def test_architecture_exists(self):
        assert (ROOT / "docs" / "architecture.md").exists()

    def test_api_reference_exists(self):
        assert (ROOT / "docs" / "api_reference.md").exists()

    def test_demo_script_exists(self):
        assert (ROOT / "scripts" / "demo_api.sh").exists()


# ------------------------------------------------------------------
# README content
# ------------------------------------------------------------------


class TestReadmeContent:
    @pytest.fixture
    def content(self):
        return (ROOT / "README.md").read_text()

    def test_title(self, content):
        assert "# Fraud Detection System" in content

    def test_overview_section(self, content):
        assert "## Overview" in content

    def test_architecture_section(self, content):
        assert "## Architecture" in content

    def test_mermaid_diagram(self, content):
        assert "```mermaid" in content

    def test_quick_start_section(self, content):
        assert "## Quick Start" in content

    def test_api_examples_section(self, content):
        assert "## API Examples" in content

    def test_curl_examples(self, content):
        assert "curl" in content

    def test_configuration_section(self, content):
        assert "## Configuration" in content

    def test_project_structure_section(self, content):
        assert "## Project Structure" in content

    def test_technology_stack(self, content):
        assert "## Technology Stack" in content

    def test_license(self, content):
        assert "## License" in content
        assert "MIT" in content

    def test_ci_badge(self, content):
        assert "![CI]" in content

    def test_coverage_badge(self, content):
        assert "![Coverage]" in content

    def test_docker_compose_instructions(self, content):
        assert "docker-up" in content

    def test_features_list(self, content):
        assert "Features" in content


# ------------------------------------------------------------------
# Architecture doc content
# ------------------------------------------------------------------


class TestArchitectureContent:
    @pytest.fixture
    def content(self):
        return (ROOT / "docs" / "architecture.md").read_text()

    def test_title(self, content):
        assert "# System Architecture" in content

    def test_components_section(self, content):
        assert "## Components" in content

    def test_data_pipeline(self, content):
        assert "Data Pipeline" in content

    def test_model_layer(self, content):
        assert "Model Layer" in content

    def test_inference_api(self, content):
        assert "Inference API" in content

    def test_streaming(self, content):
        assert "Streaming" in content

    def test_dashboard(self, content):
        assert "Dashboard" in content

    def test_data_flow(self, content):
        assert "## Data Flow" in content

    def test_technology_decisions(self, content):
        assert "## Technology Decisions" in content

    def test_xgboost_rationale(self, content):
        assert "XGBoost" in content

    def test_shap_rationale(self, content):
        assert "SHAP" in content

    def test_sqlite_rationale(self, content):
        assert "SQLite" in content


# ------------------------------------------------------------------
# API reference content
# ------------------------------------------------------------------


class TestApiReferenceContent:
    @pytest.fixture
    def content(self):
        return (ROOT / "docs" / "api_reference.md").read_text()

    def test_title(self, content):
        assert "# API Reference" in content

    def test_health_endpoint(self, content):
        assert "GET /health" in content

    def test_predict_endpoint(self, content):
        assert "POST /api/v1/predict" in content

    def test_batch_endpoint(self, content):
        assert "/api/v1/predict/batch" in content

    def test_ab_test_endpoint(self, content):
        assert "/api/v1/ab-test/results" in content

    def test_request_body_docs(self, content):
        assert "Request Body" in content

    def test_response_fields_docs(self, content):
        assert "Response" in content

    def test_error_responses(self, content):
        assert "Error Responses" in content

    def test_422_validation_error(self, content):
        assert "422" in content

    def test_curl_examples(self, content):
        assert "curl" in content

    def test_json_examples(self, content):
        assert "```json" in content


# ------------------------------------------------------------------
# Demo script content
# ------------------------------------------------------------------


class TestDemoScript:
    @pytest.fixture
    def content(self):
        return (ROOT / "scripts" / "demo_api.sh").read_text()

    def test_shebang(self, content):
        assert content.startswith("#!/")

    def test_health_check_call(self, content):
        assert "/health" in content

    def test_predict_call(self, content):
        assert "/api/v1/predict" in content

    def test_batch_call(self, content):
        assert "/api/v1/predict/batch" in content

    def test_ab_test_call(self, content):
        assert "/api/v1/ab-test/results" in content

    def test_configurable_base_url(self, content):
        assert "BASE_URL" in content


# ------------------------------------------------------------------
# Internal links in README resolve
# ------------------------------------------------------------------


class TestReadmeLinks:
    @pytest.fixture
    def content(self):
        return (ROOT / "README.md").read_text()

    def test_relative_links_resolve(self, content):
        # Find markdown links like [text](path)
        # Exclude http(s) links and badge image URLs
        link_pattern = re.compile(r"\[.*?\]\(([^)]+)\)")
        for match in link_pattern.finditer(content):
            target = match.group(1)
            if target.startswith("http"):
                continue
            resolved = ROOT / target
            assert resolved.exists(), f"Broken link: {target}"


# ------------------------------------------------------------------
# Internal links in architecture doc resolve
# ------------------------------------------------------------------


class TestArchitectureLinks:
    @pytest.fixture
    def content(self):
        return (ROOT / "docs" / "architecture.md").read_text()

    def test_relative_links_resolve(self, content):
        link_pattern = re.compile(r"\[.*?\]\(([^)]+)\)")
        for match in link_pattern.finditer(content):
            target = match.group(1)
            if target.startswith("http"):
                continue
            resolved = ROOT / "docs" / target
            assert resolved.exists(), f"Broken link: {target}"


# ------------------------------------------------------------------
# Module docstring coverage
# ------------------------------------------------------------------


class TestDocstringCoverage:
    """Every non-__init__ module under src/ must have a module-level docstring."""

    @pytest.fixture
    def src_modules(self):
        return sorted(
            p for p in (ROOT / "src").rglob("*.py") if p.name != "__init__.py"
        )

    def test_all_modules_found(self, src_modules):
        # Sanity check: we should have a reasonable number of modules
        assert len(src_modules) >= 15

    def test_all_modules_have_docstrings(self, src_modules):
        missing = []
        for path in src_modules:
            tree = ast.parse(path.read_text())
            docstring = ast.get_docstring(tree)
            if not docstring:
                missing.append(str(path.relative_to(ROOT)))
        assert missing == [], f"Modules missing docstrings: {missing}"

    def test_docstrings_are_non_trivial(self, src_modules):
        """Docstrings should be more than a single word."""
        short = []
        for path in src_modules:
            tree = ast.parse(path.read_text())
            docstring = ast.get_docstring(tree)
            if docstring and len(docstring.split()) < 3:
                short.append(str(path.relative_to(ROOT)))
        assert short == [], f"Modules with trivially short docstrings: {short}"


# ------------------------------------------------------------------
# Markdown code blocks are well-formed
# ------------------------------------------------------------------


class TestMarkdownCodeBlocks:
    """All fenced code blocks should be properly closed."""

    @pytest.fixture(
        params=["README.md", "docs/architecture.md", "docs/api_reference.md"]
    )
    def md_content(self, request):
        return (ROOT / request.param).read_text(), request.param

    def test_fenced_blocks_balanced(self, md_content):
        content, name = md_content
        count = content.count("```")
        assert count % 2 == 0, (
            f"{name}: unbalanced fenced code blocks ({count} backtick fences)"
        )
