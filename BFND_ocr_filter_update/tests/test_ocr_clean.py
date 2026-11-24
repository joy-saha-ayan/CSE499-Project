"""Tests for the OCR box filter in src/ocr.py. Pure logic, no easyocr needed."""
from src.ocr import box_record, clean_ocr_boxes


def _box(text, conf, height, top, left=0.0):
    bbox = [[left, top], [left + 100, top], [left + 100, top + height], [left, top + height]]
    return box_record(bbox, text, conf)


def test_box_record_reads_geometry_from_corner_points():
    b = box_record([[10, 20], [110, 20], [110, 80], [10, 80]], " হেডলাইন ", 0.9)
    assert b["text"] == "হেডলাইন"
    assert b["top"] == 20 and b["height"] == 60 and b["left"] == 10


def test_keeps_the_largest_headline_and_drops_watermark_and_button():
    boxes = [
        _box("বিশ্বকাপে প্রথম ম্যাচ হেরেই চাকরি হারালেন কোচ", 0.95, 60, top=100),
        _box("১৫ জুন ২০২৬", 0.9, 40, top=40),
        _box("বিস্তারিত লিংক কমেন্টে", 0.9, 22, top=400),
        _box("TSPORTS", 0.9, 14, top=10, left=500),
    ]
    text, dropped = clean_ocr_boxes(boxes, min_rel_height=0.6)
    assert "বিশ্বকাপে প্রথম ম্যাচ" in text
    reasons = {d["text"]: d["reason"] for d in dropped}
    assert reasons["বিস্তারিত লিংক কমেন্টে"] == "noise_term"
    assert reasons["TSPORTS"] == "small_text"


def test_low_confidence_junk_is_dropped():
    boxes = [_box("45E01৮০", 0.2, 60, top=0), _box("খবর", 0.9, 60, top=50)]
    text, dropped = clean_ocr_boxes(boxes)
    assert text == "খবর"
    assert dropped[0]["reason"] == "low_conf"


def test_urls_and_digit_only_boxes_are_dropped():
    boxes = [
        _box("https://example.com/post", 0.9, 60, top=0),
        _box("1234 5678", 0.9, 60, top=50),
        _box("সংবাদ", 0.9, 60, top=100),
    ]
    text, dropped = clean_ocr_boxes(boxes)
    assert text == "সংবাদ"
    assert {d["reason"] for d in dropped} == {"url", "no_letters"}


def test_reading_order_is_top_to_bottom_then_left_to_right():
    boxes = [
        _box("দ্বিতীয়", 0.9, 60, top=200, left=0),
        _box("প্রথম", 0.9, 60, top=0, left=0),
        _box("পাশের", 0.9, 60, top=0, left=300),
    ]
    text, _ = clean_ocr_boxes(boxes)
    assert text == "প্রথম পাশের দ্বিতীয়"


def test_empty_input_gives_empty_text():
    assert clean_ocr_boxes([]) == ("", [])


def test_noise_term_match_is_case_insensitive():
    boxes = [_box("Click Here Now", 0.9, 60, top=0), _box("News text", 0.9, 60, top=50)]
    text, dropped = clean_ocr_boxes(boxes)
    assert text == "News text"
    assert dropped[0]["reason"] == "noise_term"
