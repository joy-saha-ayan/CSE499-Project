from src.experiments import BASE_BLOCKS, experiment_definitions


def _blocks_used(defs):
    return {b for _, blocks in defs for b in blocks}


def test_cultural_off_has_no_e4_and_no_cultural_block():
    defs = experiment_definitions(False)
    names = [n for n, _ in defs]
    assert "E4_plus_cultural" not in names
    assert "cultural" not in _blocks_used(defs)


def test_e5_without_cultural_is_e3_plus_evidence():
    defs = dict(experiment_definitions(False))
    assert defs["E5_plus_evidence"] == BASE_BLOCKS + ["evidence"]


def test_cultural_on_adds_e4_and_cultural_to_e5():
    defs = dict(experiment_definitions(True))
    assert defs["E4_plus_cultural"] == BASE_BLOCKS + ["cultural"]
    assert defs["E5_plus_evidence"] == BASE_BLOCKS + ["cultural", "evidence"]


def test_unimodal_and_e3_definitions_do_not_change():
    for flag in (False, True):
        defs = dict(experiment_definitions(flag))
        assert defs["E0_has_image_only"] == ["has_image"]
        assert defs["E1f_frozen_banglabert_mlp"] == ["bert"]
        assert defs["E2_clip_image_only"] == ["clip_img"]
        assert defs["E3_banglabert_clip"] == BASE_BLOCKS
