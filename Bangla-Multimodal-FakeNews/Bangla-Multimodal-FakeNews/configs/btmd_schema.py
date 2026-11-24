"""BTMD column mapping.

Filled in from the inspection of metadata.parquet (6058 rows):
- The label column is 'multi_category', not 'label'. 'label' only holds Fake/Real, and
  Satire and Clickbait both sit inside 'Fake'.
- The text is split across two columns (heading, text); they are joined together.
- image_path values already start with 'images/', so image_root is empty.
- One post can have several images separated by ';'. The first existing image is used.
- group_col = heading (text hash when the heading is empty).
- BTMD has no source URL column, so source_url_col = None. The label-source leakage guard
  therefore cannot run; evidence must come from independent sources.
"""

BTMD_SCHEMA = {
    "table_file": "data/metadata.parquet",  # relative to RAW_DIR
    "id_col": "id",
    "text_col": ["heading", "text"],
    "label_col": "multi_category",          # Real / Misinformation / Satire / Clickbait
    "image_col": "image_path",
    "image_root": "",                       # values already start with "images/"
    "group_col": "heading",
    "source_url_col": None,
}
