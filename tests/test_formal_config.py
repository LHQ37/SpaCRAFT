from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_multislice_invariants():
    config = yaml.safe_load(
        (ROOT / "configs/multi_slice/spacraft_multislice_e11_e18.yaml").read_text()
    )
    assert config["method"] == "SpaCRAFT"
    assert config["lsi"] == {"components": 30, "drop_first": True}
    assert config["graph"]["cross_slice_edges"] is False
    assert config["model"]["encoder_dimensions"] == [29, 50, 10]
    assert config["steps"] == 500
    assert config["optimizer"]["learning_rate"] == 0.00011929202947805012


def test_single_slice_invariants():
    config = yaml.safe_load(
        (ROOT / "configs/single_slice/spacraft_single_slice.yaml").read_text()
    )
    assert config == {
        "method": "SpaCRAFT",
        "latent_dimension": 10,
        "training_seed": 1236,
        "clustering_seed": 1236,
        "training_steps": 500,
        "multi_slice_modules_enabled": False,
    }


def test_formal_entrypoints_and_outputs_exist():
    required = [
        "scripts/single_slice/run_spacraft.py",
        "scripts/multi_slice/run_spacraft.py",
        "figures/multi_slice/plot_multislice_benchmark.py",
        "figures/downstream/plot_single_slice_downstream.py",
        "figures/downstream/plot_multislice_downstream.py",
        "figures/ablation/plot_ablation.py",
        "figures/ablation/Figure_ablation.png",
        "figures/ablation/Figure_ablation.pdf",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_development_model_names_removed_from_text_sources():
    old_space_name = "Space" + "ATAC"
    old_placeholder = "My" + "Model"
    forbidden = re.compile(
        rf"{old_space_name}(?:-CL|_CL)?|{old_placeholder}", re.IGNORECASE
    )
    suffixes = {".py", ".R", ".md", ".yaml", ".yml", ".json", ".csv", ".txt"}
    offenders = []
    for path in ROOT.rglob("*"):
        if path == Path(__file__) or not path.is_file() or path.suffix not in suffixes:
            continue
        else:
            if forbidden.search(path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders
