"""Unit tests for per-layer norm and fusion heads B0/F1/F2/F3/F4/B0+ (fusion.py, T-05a/b)."""

import pytest

torch = pytest.importorskip("torch")

from layer_fusion.contracts import FusionMethod, LayerSubset  # noqa: E402
from layer_fusion.fusion import (  # noqa: E402
    PerLayerNorm,
    build_fusion_head,
    count_trainable_parameters,
    extract_document_layer_weights,
    extract_layer_weights,
    no_weight_decay_parameters,
)
from layer_fusion.layers import GLOBAL_LAYERS, LOCAL_LAYERS  # noqa: E402

HIDDEN = 16  # small hidden size keeps tests fast; heads take it as an argument
NUM_CLASSES = 3
BATCH = 4
TESTED_METHODS = list(FusionMethod)  # B0, F1, F2, F3, F4, B0+
# Slots each head reads; all other slots must not affect its output.
USED_SLOTS = {
    FusionMethod.B0: (22,),
    FusionMethod.B0_PLUS: (22,),
    FusionMethod.F1: tuple(range(23)),
    FusionMethod.F2: tuple(range(23)),
    FusionMethod.F3: (19, 20, 21, 22),
}


def _noise(*shape: int, seed: int) -> "torch.Tensor":
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _vectors(batch: int = BATCH, seed: int = 0) -> "torch.Tensor":
    return _noise(batch, 23, HIDDEN, seed=seed)


def _head(method: FusionMethod, subset: LayerSubset = LayerSubset.ALL, seed: int = 0):
    torch.manual_seed(seed)  # same seed -> same classifier init across heads of equal shape
    return build_fusion_head(method, NUM_CLASSES, subset, hidden_size=HIDDEN)


def _set_f2_logits(head, seed: int = 1) -> None:
    with torch.no_grad():
        head.layer_logits.copy_(_noise(head.layer_logits.numel(), seed=seed) * 2.0)


def _train_f4_step(head, seed: int = 3) -> None:
    """Run public optimizer steps so the zero-initialized gate emits distinct weights."""
    optimizer = torch.optim.Adam(head.parameters(), lr=0.5)
    vectors = _vectors(batch=8, seed=seed)
    targets = torch.arange(8) % NUM_CLASSES
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(head(vectors), targets).backward()
        optimizer.step()


def _rescale_per_layer(vectors: "torch.Tensor") -> "torch.Tensor":
    """a*x+b with a different (a, b) per layer slot; scales 0.5 .. 1000.

    Much smaller scales would push the variance toward LayerNorm's eps (1e-5), where exact
    invariance no longer holds by design.
    """
    scales = torch.logspace(-0.3, 3, 23).view(1, 23, 1)
    shifts = torch.linspace(-5.0, 5.0, 23).view(1, 23, 1)
    return vectors * scales + shifts


def _perturb(vectors: "torch.Tensor", slots, seed: int = 99) -> "torch.Tensor":
    changed = vectors.clone()
    for i, slot in enumerate(slots):
        changed[:, slot] += _noise(vectors.shape[0], HIDDEN, seed=seed + i) * 50.0
    return changed


# --- per-layer norm --------------------------------------------------------------------


def test_per_layer_norm_has_no_parameters_and_normalizes_each_vector() -> None:
    norm = PerLayerNorm(HIDDEN)
    assert sum(p.numel() for p in norm.parameters()) == 0
    out = norm(_vectors() * 7.0 + 3.0)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(BATCH, 23), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(BATCH, 23), atol=1e-3)


@pytest.mark.parametrize("method", TESTED_METHODS)
def test_every_head_is_invariant_to_per_layer_affine_rescaling(method: FusionMethod) -> None:
    """US 28: per-layer norm precedes fusion in every head, so a*x+b per layer changes nothing."""
    head = _head(method)
    if method == FusionMethod.F2:
        _set_f2_logits(head)
    if method == FusionMethod.F4:
        _train_f4_step(head)  # a zero gate would hide what the gate reads
    vectors = _vectors()
    assert torch.allclose(head(vectors), head(_rescale_per_layer(vectors)), atol=1e-4)


# --- parameter counts and shapes -------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (FusionMethod.B0, HIDDEN * NUM_CLASSES + NUM_CLASSES),
        (FusionMethod.F1, HIDDEN * NUM_CLASSES + NUM_CLASSES),
        (FusionMethod.F2, 23 + HIDDEN * NUM_CLASSES + NUM_CLASSES),
        (FusionMethod.F3, 4 * HIDDEN * NUM_CLASSES + NUM_CLASSES),
    ],
)
def test_trainable_parameter_counts_match_definitions(method: FusionMethod, expected: int) -> None:
    assert count_trainable_parameters(_head(method)) == expected


def test_f2_subset_parameter_counts() -> None:
    classifier = HIDDEN * NUM_CLASSES + NUM_CLASSES
    assert count_trainable_parameters(_head(FusionMethod.F2, LayerSubset.GLOBAL_ONLY)) == (
        8 + classifier
    )
    assert count_trainable_parameters(_head(FusionMethod.F2, LayerSubset.LOCAL_ONLY)) == (
        14 + classifier
    )


@pytest.mark.parametrize("method", TESTED_METHODS)
@pytest.mark.parametrize("batch", [1, 5])
def test_logits_shape(method: FusionMethod, batch: int) -> None:
    assert _head(method)(_vectors(batch=batch)).shape == (batch, NUM_CLASSES)


# --- which slots each head reads -------------------------------------------------------


@pytest.mark.parametrize(
    "method", [FusionMethod.B0, FusionMethod.B0_PLUS, FusionMethod.F3]
)  # F1/F2/F4 read every slot
def test_heads_ignore_unused_slots(method: FusionMethod) -> None:
    unused = [s for s in range(23) if s not in USED_SLOTS[method]]
    head = _head(method)
    vectors = _vectors()
    assert torch.allclose(head(vectors), head(_perturb(vectors, unused)), atol=1e-5)


@pytest.mark.parametrize(
    ("method", "slot"),
    [(FusionMethod.B0, 22), (FusionMethod.B0_PLUS, 22)]
    + [(FusionMethod.F3, s) for s in (19, 20, 21, 22)],
)
def test_heads_react_to_each_used_slot(method: FusionMethod, slot: int) -> None:
    head = _head(method)
    vectors = _vectors()
    assert not torch.allclose(head(vectors), head(_perturb(vectors, [slot])), atol=1e-3)


@pytest.mark.parametrize(
    ("subset", "kept"),
    [(LayerSubset.GLOBAL_ONLY, GLOBAL_LAYERS), (LayerSubset.LOCAL_ONLY, LOCAL_LAYERS)],
)
def test_f2_subset_ignores_excluded_slots_and_reacts_to_kept_ones(subset, kept) -> None:
    head = _head(FusionMethod.F2, subset)
    _set_f2_logits(head)
    vectors = _vectors()
    excluded = [s for s in range(23) if s not in kept]
    assert 0 in excluded  # embedding vector is never part of a subset
    assert torch.allclose(head(vectors), head(_perturb(vectors, excluded)), atol=1e-5)
    for slot in kept:
        assert not torch.allclose(head(vectors), head(_perturb(vectors, [slot])), atol=1e-4)


# --- F2 weighting ----------------------------------------------------------------------


def test_f2_starts_equal_to_f1() -> None:
    f1, f2 = _head(FusionMethod.F1), _head(FusionMethod.F2)
    vectors = _vectors()
    assert torch.allclose(f1(vectors), f2(vectors), atol=1e-5)


def test_f2_forward_uses_non_uniform_weights() -> None:
    """Forward equals the hand-computed weighted sum of normalized layer vectors."""
    head = _head(FusionMethod.F2)
    _set_f2_logits(head)
    vectors = _vectors()
    weights = torch.tensor(extract_layer_weights(head))
    assert weights.max() - weights.min() > 0.05  # really non-uniform
    normed = torch.nn.functional.layer_norm(vectors, (HIDDEN,))
    expected = head.classifier(torch.einsum("l,blh->bh", weights, normed))
    assert torch.allclose(head(vectors), expected, atol=1e-5)
    # ...and therefore differs from F1 with the same classifier.
    assert not torch.allclose(head(vectors), _head(FusionMethod.F1)(vectors), atol=1e-3)


def test_f2_has_no_hidden_scale_when_all_layers_agree() -> None:
    """ADR 0005: weights form a convex combination, so identical layers give exactly B0."""
    vector = _noise(BATCH, 1, HIDDEN, seed=5)
    identical = vector.expand(BATCH, 23, HIDDEN).clone()
    for subset in LayerSubset:
        f2 = _head(FusionMethod.F2, subset)
        _set_f2_logits(f2, seed=7)
        b0 = _head(FusionMethod.B0)  # same seed -> same classifier as f2
        assert torch.allclose(f2(identical), b0(identical), atol=1e-5)


def test_f2_weights_are_positive_and_sum_to_one() -> None:
    head = _head(FusionMethod.F2)
    _set_f2_logits(head)
    weights = extract_layer_weights(head)
    assert weights is not None and len(weights) == 23
    assert all(w > 0 for w in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_f2_subsets_expose_8_global_and_14_local_weights() -> None:
    assert len(extract_layer_weights(_head(FusionMethod.F2, LayerSubset.GLOBAL_ONLY))) == 8
    assert len(extract_layer_weights(_head(FusionMethod.F2, LayerSubset.LOCAL_ONLY))) == 14


def test_f2_layer_logits_receive_nonzero_gradient() -> None:
    head = _head(FusionMethod.F2)
    head(_vectors()).pow(2).sum().backward()
    assert head.layer_logits.grad is not None
    assert head.layer_logits.grad.norm() > 0


@pytest.mark.parametrize(
    "method", [FusionMethod.B0, FusionMethod.B0_PLUS, FusionMethod.F1, FusionMethod.F3]
)
def test_heads_without_learned_layer_weights_return_none(method: FusionMethod) -> None:
    assert extract_layer_weights(_head(method)) is None


# --- F4 (per-document gating) ---------------------------------------------------------


def test_f4_parameter_count_is_gate_plus_classifier() -> None:
    gate = HIDDEN * 23 + 23
    classifier = HIDDEN * NUM_CLASSES + NUM_CLASSES
    assert count_trainable_parameters(_head(FusionMethod.F4)) == gate + classifier


def test_f4_starts_equal_to_f1() -> None:
    f1, f4 = _head(FusionMethod.F1), _head(FusionMethod.F4)
    vectors = _vectors()
    assert torch.allclose(f1(vectors), f4(vectors), atol=1e-5)


def test_f4_gate_reads_the_mean_of_the_per_layer_normed_vectors() -> None:
    """SPEC (T-05b): weights = softmax(W · mean_l LN(x_l) + b), per document.

    White-box formula check using the gate's own trained (W, b), so a gate that reads only one
    slot (or sums instead of averaging) fails here.
    """
    head = _head(FusionMethod.F4)
    _train_f4_step(head)
    vectors = _vectors()
    mean_normed = torch.nn.functional.layer_norm(vectors, (HIDDEN,)).mean(dim=1)
    expected = torch.softmax(mean_normed @ head.gate.weight.T + head.gate.bias, dim=-1)
    actual = torch.tensor(extract_document_layer_weights(head, vectors))
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("slot", range(23))
def test_f4_document_weights_react_to_every_slot(slot: int) -> None:
    """US 33: the gate looks at all 23 layer vectors, not only the last one or the embedding."""
    head = _head(FusionMethod.F4)
    _train_f4_step(head)
    vectors = _vectors()
    before = torch.tensor(extract_document_layer_weights(head, vectors))
    after = torch.tensor(extract_document_layer_weights(head, _perturb(vectors, [slot])))
    assert not torch.allclose(before, after, atol=1e-4)


def test_f4_document_weights_are_positive_sum_to_one_and_differ_per_document() -> None:
    head = _head(FusionMethod.F4)
    _train_f4_step(head)
    weights = torch.tensor(extract_document_layer_weights(head, _vectors()))
    assert weights.shape == (BATCH, 23)
    assert (weights > 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(BATCH), atol=1e-6)
    # Input-dependent gating: different documents get different layer weights.
    assert (weights[0] - weights[1]).abs().max() > 1e-3


def test_f4_forward_equals_hand_computed_per_document_weighted_sum() -> None:
    head = _head(FusionMethod.F4)
    _train_f4_step(head)
    vectors = _vectors()
    weights = torch.tensor(extract_document_layer_weights(head, vectors))
    normed = torch.nn.functional.layer_norm(vectors, (HIDDEN,))
    expected = head.classifier(torch.einsum("bl,blh->bh", weights, normed))
    assert torch.allclose(head(vectors), expected, atol=1e-5)
    # ...and differs from F1 with the same classifier.
    assert not torch.allclose(head(vectors), _head(FusionMethod.F1)(vectors), atol=1e-3)


def test_f4_document_weights_are_invariant_to_per_layer_affine_rescaling() -> None:
    """US 28 for F4: the gate sees per-layer-normed vectors, so weights do not follow scale."""
    head = _head(FusionMethod.F4)
    _train_f4_step(head)
    vectors = _vectors()
    rescaled = _rescale_per_layer(vectors)
    assert torch.allclose(
        torch.tensor(extract_document_layer_weights(head, vectors)),
        torch.tensor(extract_document_layer_weights(head, rescaled)),
        atol=1e-5,
    )


def test_f4_training_step_moves_document_weights_away_from_uniform() -> None:
    """Outer behavior: one gradient step moves F4 away from uniform F1 and varies per document."""
    head = _head(FusionMethod.F4)
    vectors = _vectors()
    uniform = torch.full((BATCH, 23), 1.0 / 23.0)
    before = torch.tensor(extract_document_layer_weights(head, vectors))
    assert torch.allclose(before, uniform, atol=1e-6)
    _train_f4_step(head)
    after = torch.tensor(extract_document_layer_weights(head, vectors))
    assert not torch.allclose(after, uniform, atol=1e-3)
    assert (after[0] - after[1]).abs().max() > 1e-3


def test_extract_document_layer_weights_preserves_training_mode() -> None:
    head = _head(FusionMethod.F4)
    head.train()
    assert extract_document_layer_weights(head, _vectors()) is not None
    assert head.training is True
    head.eval()
    assert extract_document_layer_weights(head, _vectors()) is not None
    assert head.training is False


def test_no_weight_decay_parameters_marks_only_layer_weighting_params() -> None:
    for method in (FusionMethod.B0, FusionMethod.F1, FusionMethod.F3, FusionMethod.B0_PLUS):
        assert no_weight_decay_parameters(_head(method)) == ()
    f2_excluded = no_weight_decay_parameters(_head(FusionMethod.F2))
    assert sum(p.numel() for p in f2_excluded) == 23
    f4_excluded = no_weight_decay_parameters(_head(FusionMethod.F4))
    assert sum(p.numel() for p in f4_excluded) == HIDDEN * 23 + 23


def test_f4_exposes_document_weights_but_no_dataset_level_weights() -> None:
    head = _head(FusionMethod.F4)
    assert extract_layer_weights(head) is None  # F2-style dataset-level weights only
    assert len(extract_document_layer_weights(head, _vectors(batch=2))) == 2


@pytest.mark.parametrize(
    "method",
    [FusionMethod.B0, FusionMethod.B0_PLUS, FusionMethod.F1, FusionMethod.F2, FusionMethod.F3],
)
def test_heads_without_gate_have_no_document_weights(method: FusionMethod) -> None:
    assert extract_document_layer_weights(_head(method), _vectors()) is None


# --- B0+ (parameter-matched control: last layer vector -> 2-layer MLP) ------------------

B0_PLUS_HIDDEN_BY_CLASSES = {3: 26, 2: 25}  # ADR 0008 (MNLI 26 / 2 classes 25)


@pytest.mark.parametrize(("num_classes", "mlp_hidden"), sorted(B0_PLUS_HIDDEN_BY_CLASSES.items()))
def test_b0_plus_parameter_count_follows_adr_0008_hidden_size(
    num_classes: int, mlp_hidden: int
) -> None:
    head = build_fusion_head(FusionMethod.B0_PLUS, num_classes, hidden_size=HIDDEN)
    expected = (HIDDEN * mlp_hidden + mlp_hidden) + (mlp_hidden * num_classes + num_classes)
    assert count_trainable_parameters(head) == expected


@pytest.mark.parametrize("num_classes", [1, 4, 5])
def test_b0_plus_hidden_size_is_undecided_for_other_class_counts(num_classes: int) -> None:
    """SPEC leaves the AG News (4-class) hidden size TBD; do not invent one."""
    with pytest.raises(ValueError, match="num_classes"):
        build_fusion_head(FusionMethod.B0_PLUS, num_classes, hidden_size=HIDDEN)


def test_b0_plus_is_a_nonlinear_mlp_not_a_linear_classifier() -> None:
    """Shape differs from B0 on purpose (ADR 0008): f(x) + f(-x) != 2 f(0) for a ReLU MLP."""
    head = _head(FusionMethod.B0_PLUS)
    vectors = _vectors()
    mirrored = vectors.clone()
    mirrored[:, 22] = -mirrored[:, 22]  # per-layer norm is odd-symmetric, so the MLP input flips
    zeros = torch.zeros_like(vectors)  # normalized to 0 -> MLP input 0
    midpoint = (head(vectors) + head(mirrored)) / 2
    assert not torch.allclose(midpoint, head(zeros), atol=1e-3)


def test_b0_plus_every_parameter_receives_gradient() -> None:
    head = _head(FusionMethod.B0_PLUS)
    head(_vectors(batch=5)).pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.norm() > 0 for p in head.parameters())


# --- argument validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [FusionMethod.B0, FusionMethod.F1, FusionMethod.F3, FusionMethod.F4, FusionMethod.B0_PLUS],
)
@pytest.mark.parametrize("subset", [LayerSubset.GLOBAL_ONLY, LayerSubset.LOCAL_ONLY])
def test_layer_subset_is_f2_only(method: FusionMethod, subset: LayerSubset) -> None:
    with pytest.raises(ValueError, match="F2"):
        _head(method, subset)


def test_wrong_input_shape_raises() -> None:
    head = _head(FusionMethod.F1)
    with pytest.raises(ValueError, match="shape"):
        head(torch.zeros(BATCH, 22, HIDDEN))
    with pytest.raises(ValueError, match="shape"):
        head(torch.zeros(BATCH, 23, HIDDEN + 1))


def test_num_classes_below_two_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        build_fusion_head(FusionMethod.B0, 1, hidden_size=HIDDEN)


@pytest.mark.parametrize("hidden_size", [0, -4])
def test_non_positive_hidden_size_raises(hidden_size: int) -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        build_fusion_head(FusionMethod.B0, NUM_CLASSES, hidden_size=hidden_size)
