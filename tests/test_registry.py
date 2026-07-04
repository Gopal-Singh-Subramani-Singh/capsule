from __future__ import annotations
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from moto import mock_aws
import boto3


@pytest.fixture
def mock_s3_registry(tmp_path):
    """Registry backed by moto (in-memory S3)."""
    with mock_aws():
        s3 = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        with patch("capsule.registry.boto3.client", return_value=s3):
            from capsule.registry import ModelRegistry
            registry = ModelRegistry(
                endpoint="http://localhost:9000",
                access_key="test",
                secret_key="test",
                bucket="capsule-test",
            )
            yield registry, s3, tmp_path


def test_registry_push_and_pull_image_tag():
    with mock_aws():
        s3 = boto3.client(
            "s3", region_name="us-east-1",
            aws_access_key_id="test", aws_secret_access_key="test",
        )
        with patch("capsule.registry.boto3.client", return_value=s3):
            from capsule.registry import ModelRegistry
            registry = ModelRegistry(bucket="capsule-test")
            registry.push_image_tag("mymodel", "1.0", "localhost:5001/mymodel:1.0")
            tag = registry.get_image_tag("mymodel", "1.0")
            assert tag == "localhost:5001/mymodel:1.0"


def test_registry_missing_tag_returns_none():
    with mock_aws():
        s3 = boto3.client(
            "s3", region_name="us-east-1",
            aws_access_key_id="test", aws_secret_access_key="test",
        )
        with patch("capsule.registry.boto3.client", return_value=s3):
            from capsule.registry import ModelRegistry
            registry = ModelRegistry(bucket="capsule-test")
            tag = registry.get_image_tag("ghost", "9.9")
            assert tag is None


def test_registry_list_versions():
    with mock_aws():
        s3 = boto3.client(
            "s3", region_name="us-east-1",
            aws_access_key_id="test", aws_secret_access_key="test",
        )
        with patch("capsule.registry.boto3.client", return_value=s3):
            from capsule.registry import ModelRegistry
            registry = ModelRegistry(bucket="capsule-test")
            registry.push_image_tag("model", "1.0", "img:1.0")
            registry.push_image_tag("model", "2.0", "img:2.0")
            versions = registry.list_versions("model")
            assert "1.0" in versions or "2.0" in versions
