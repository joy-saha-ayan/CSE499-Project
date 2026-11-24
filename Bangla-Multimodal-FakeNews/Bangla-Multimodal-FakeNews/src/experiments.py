"""Experiment feature sets for script 04. Pure definitions, no torch or file access."""

BASE_BLOCKS = ["bert", "clip_img", "clip_txt", "sim"]


def experiment_definitions(use_cultural: bool):
    """Return [(name, feature blocks)]. Cultural features enter only when use_cultural is True.

    E5 is E3 + evidence. It adds cultural features only when they are allowed (decision D-2).
    """
    defs = [
        ("E0_has_image_only", ["has_image"]),
        ("E1f_frozen_banglabert_mlp", ["bert"]),
        ("E2_clip_image_only", ["clip_img"]),
        ("E3_banglabert_clip", list(BASE_BLOCKS)),
    ]
    if use_cultural:
        defs.append(("E4_plus_cultural", BASE_BLOCKS + ["cultural"]))
    e5 = BASE_BLOCKS + (["cultural"] if use_cultural else []) + ["evidence"]
    defs.append(("E5_plus_evidence", e5))
    return defs
