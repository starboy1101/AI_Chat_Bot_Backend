"""Automotive SRS generation pipeline."""

from srs_generator.extractor import SRSIntelligencePipeline
from srs_generator.models import SRSProject, StructuredRequirement
from srs_generator.template_engine import SrsTemplateRenderer

__all__ = [
    "SRSIntelligencePipeline",
    "SRSProject",
    "StructuredRequirement",
    "SrsTemplateRenderer",
]
