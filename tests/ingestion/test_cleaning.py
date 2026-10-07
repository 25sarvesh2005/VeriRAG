"""Tests for text cleaning and normalization."""

from app.ingestion.cleaning import clean_text


def test_clean_text_normalizes_whitespace():
    raw = "This   is   a   test.\r\n\r\n\r\n\r\nNew   paragraph."
    cleaned = clean_text(raw)
    assert "This is a test." in cleaned
    assert "New paragraph." in cleaned
    # Multiple newlines collapsed to 2
    assert "\n\n\n" not in cleaned


def test_clean_text_normalizes_bullet_points():
    raw = "• First item\n‣ Second item\n⁃ Third item"
    cleaned = clean_text(raw)
    assert "- First item" in cleaned
    assert "- Second item" in cleaned
    assert "- Third item" in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
