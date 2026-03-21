"""Tests for MIST arguments handling."""
import argparse
import pytest

# MIST imports.
import mist.cli.args as args_mod


# =============================================================================
# Converters / Validators.
# =============================================================================


@pytest.mark.parametrize("val,expected", [(1, 1), ("2", 2), (123, 123)])
def test_positive_int_success(val, expected):
    """Valid positive integers are accepted as-is or converted from strings."""
    assert args_mod.positive_int(val) == expected


@pytest.mark.parametrize("val", [0, -1, "-5"])
def test_positive_int_rejects_non_positive(val):
    """ Non-positive values raise argparse.ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_int(val)


@pytest.mark.parametrize("val", ["abc", "3.2.1"])
def test_positive_int_rejects_non_numeric_string(val):
    """Non-numeric strings raise ValueError."""
    with pytest.raises(ValueError):
        args_mod.positive_int(val)


@pytest.mark.parametrize("val,expected", [(0.1, 0.1), ("2.5", 2.5), (3, 3.0)])
def test_positive_float_success(val, expected):
    """Valid positive floats are accepted as-is or converted from strings."""
    assert args_mod.positive_float(val) == expected


@pytest.mark.parametrize("val", [0, -0.1, "-2.3"])
def test_positive_float_rejects_non_positive(val):
    """ Non-positive values raise argparse.ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_float(val)


@pytest.mark.parametrize("val", ["abc", "1.2.3"])
def test_positive_float_rejects_non_numeric_string(val):
    """Non-numeric strings raise ValueError."""
    with pytest.raises(ValueError):
        args_mod.positive_float(val)


@pytest.mark.parametrize(
    "val,expected", [(0, 0), ("0", 0), (5, 5), ("10", 10)]
)
def test_non_negative_int_success(val, expected):
    """Valid non-negative integers accepted as-is or converted from strings."""
    assert args_mod.non_negative_int(val) == expected


@pytest.mark.parametrize("val", [-1, "-7"])
def test_non_negative_int_rejects_negative(val):
    """Negative values raise argparse.ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.non_negative_int(val)


@pytest.mark.parametrize("val", ["abc", "1.0"])
def test_non_negative_int_rejects_non_int_string(val):
    """Non-integer strings raise ValueError."""
    with pytest.raises(ValueError):
        args_mod.non_negative_int(val)


@pytest.mark.parametrize(
    "val,expected", [(0, 0.0), ("0", 0.0), (0.5, 0.5), ("1", 1.0), (1.0, 1.0)]
)
def test_float_0_1_success(val, expected):
    """Valid floats in [0.0, 1.0] accepted as-is or converted from strings."""
    assert args_mod.float_0_1(val) == expected


@pytest.mark.parametrize("val", [-0.0001, "1.0001"])
def test_float_0_1_out_of_range(val):
    """Values outside [0.0, 1.0] raise argparse.ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.float_0_1(val)


@pytest.mark.parametrize("val", ["abc", "0..5"])
def test_float_0_1_non_numeric_string(val):
    """Non-numeric strings raise ValueError."""
    with pytest.raises(ValueError):
        args_mod.float_0_1(val)


@pytest.mark.parametrize(
    "val,expected",
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("y", True),
        ("true", True),
        ("t", True),
        ("1", True),
        ("no", False),
        ("n", False),
        ("false", False),
        ("f", False),
        ("0", False),
        ("TrUe", True),
        ("FaLsE", False),
    ],
)
def test_str2bool_success(val, expected):
    """Valid boolean strings (various cases) are converted correctly."""
    assert args_mod.str2bool(val) is expected


def test_str2bool_rejects_unknown():
    """Unknown strings raise argparse.ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.str2bool("maybe")


# =============================================================================
# ArgParser helper methods.
# =============================================================================


def _mk_parser() -> args_mod.ArgParser:
    """Create a bare ArgParser for testing."""
    return args_mod.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def test_argparser_arg_adds_argument_and_parses():
    """arg() adds an argument that can be parsed."""
    parser = _mk_parser()
    parser.arg("--value", type=int, default=7)
    ns = parser.parse_args([])
    assert ns.value == 7
    ns = parser.parse_args(["--value", "42"])
    assert ns.value == 42


def test_argparser_flag_store_true():
    """flag() with store_true creates a boolean flag."""
    parser = _mk_parser()
    parser.flag("--debug")
    ns = parser.parse_args([])
    assert ns.debug is False
    ns = parser.parse_args(["--debug"])
    assert ns.debug is True


def test_argparser_boolean_flag_explicit_and_implicit():
    """boolean_flag() creates a boolean flag with explicit true/false."""
    parser = _mk_parser()
    parser.boolean_flag("--feature", default=False, help="Toggle")

    # Default.
    ns = parser.parse_args([])
    assert ns.feature is False

    # Implicit true (no value).
    ns = parser.parse_args(["--feature"])
    assert ns.feature is True

    # Explicit true/false.
    ns = parser.parse_args(["--feature", "true"])
    assert ns.feature is True
    ns = parser.parse_args(["--feature", "false"])
    assert ns.feature is False

    # Invalid value -> argparse error (SystemExit).
    with pytest.raises(SystemExit):
        parser.parse_args(["--feature", "maybe"])


# =============================================================================
# add_analyzer_args.
# =============================================================================


def test_add_analyzer_args_defaults_and_parsing():
    """add_analyzer_args adds expected args w/ correct defaults and parsing."""
    parser = _mk_parser()
    args_mod.add_analyzer_args(parser)

    # No args specified -> defaults.
    ns = parser.parse_args([])
    assert hasattr(ns, "data")
    assert hasattr(ns, "results")
    assert hasattr(ns, "nfolds")
    assert hasattr(ns, "overwrite")
    assert ns.data is None
    assert ns.results is None
    assert ns.nfolds is None
    assert ns.overwrite is False

    # Explicit values.
    ns = parser.parse_args(
        [
            "--data", "path/to/dataset.json",
            "--results", "out",
            "--nfolds", "3",
            "--overwrite",
        ]
    )
    assert ns.data == "path/to/dataset.json"
    assert ns.results == "out"
    assert ns.nfolds == 3
    assert ns.overwrite is True


# =============================================================================
# add_preprocess_args.
# =============================================================================


def test_add_preprocess_args_defaults_and_parsing():
    """add_preprocess_args adds expected args/correct defaults and parsing."""
    parser = _mk_parser()
    args_mod.add_preprocess_args(parser)

    # Defaults.
    ns = parser.parse_args([])
    assert ns.results is None
    assert ns.numpy is None
    assert ns.no_preprocess is False
    assert ns.compute_dtms is False
    assert ns.overwrite is False

    # Explicit.
    ns = parser.parse_args(
        [
            "--results", "out",
            "--numpy", "npdir",
            "--no-preprocess",
            "--compute-dtms",
            "--overwrite",
        ]
    )
    assert ns.results == "out"
    assert ns.numpy == "npdir"
    assert ns.no_preprocess is True
    assert ns.compute_dtms is True
    assert ns.overwrite is True

    # Explicit boolean_flag false.
    ns = parser.parse_args(["--no-preprocess", "false"])
    assert ns.no_preprocess is False


# =============================================================================
# add_train_args (with deterministic registries).
# =============================================================================


@pytest.fixture
def patched_registries(monkeypatch):
    """Patch the registry listing functions to return deterministic values."""
    monkeypatch.setattr(
        args_mod,
        "list_registered_models",
        lambda: ["mednext", "unet"],
        raising=True,
    )
    monkeypatch.setattr(
        args_mod,
        "list_registered_losses",
        lambda: ["dice", "ce"],
        raising=True,
    )
    monkeypatch.setattr(
        args_mod,
        "list_alpha_schedulers",
        lambda: ["linear", "cosine"],
        raising=True,
    )
    monkeypatch.setattr(
        args_mod, "list_lr_schedulers", lambda: ["cos", "none"], raising=True
    )
    monkeypatch.setattr(
        args_mod, "list_optimizers", lambda: ["adam", "sgd"], raising=True
    )


def test_add_train_args_defaults_and_basic_parse(patched_registries):
    """add_train_args adds expected args/correct defaults and basic parsing."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    # defaults
    ns = parser.parse_args([])
    assert ns.results is None
    assert ns.numpy is None
    assert ns.gpus == [-1]
    assert ns.model is None
    assert ns.patch_size is None
    assert ns.loss is None
    assert ns.use_dtms is False
    assert ns.composite_loss_weighting is None
    assert ns.epochs is None
    assert ns.batch_size_per_gpu is None
    assert ns.learning_rate is None
    assert ns.lr_scheduler is None
    assert ns.optimizer is None
    assert ns.l2_penalty is None
    assert ns.folds is None
    assert ns.overwrite is False


def test_add_train_args_full_parse_success(patched_registries):
    """add_train_args correctly parses a full set of valid arguments."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    argv = [
        "--results", "out",
        "--numpy", "npdir",
        "--gpus", "0", "1",
        "--model", "mednext",
        "--patch-size", "32", "64", "48",
        "--loss", "dice",
        "--use-dtms",
        "--composite-loss-weighting", "linear",
        "--epochs", "10",
        "--batch-size-per-gpu", "2",
        "--learning-rate", "0.001",
        "--lr-scheduler", "cos",
        "--optimizer", "adam",
        "--l2-penalty", "0.0005",
        "--folds", "0", "2", "4",
        "--overwrite",
    ]
    ns = parser.parse_args(argv)

    assert ns.results == "out"
    assert ns.numpy == "npdir"
    assert ns.gpus == [0, 1]
    assert ns.model == "mednext"
    assert ns.patch_size == [32, 64, 48]
    assert ns.loss == "dice"
    assert ns.use_dtms is True
    assert ns.composite_loss_weighting == "linear"
    assert ns.epochs == 10
    assert ns.batch_size_per_gpu == 2
    assert ns.learning_rate == pytest.approx(0.001)
    assert ns.lr_scheduler == "cos"
    assert ns.optimizer == "adam"
    assert ns.l2_penalty == pytest.approx(0.0005)
    assert ns.folds == [0, 2, 4]
    assert ns.overwrite is True


def test_add_train_args_enforces_choices(patched_registries):
    """add_train_args enforces model, loss, schedulers, optimizer."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    # Bad model.
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "badmodel"])

    # Bad loss.
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss", "badloss"])

    # Bad alpha scheduler.
    with pytest.raises(SystemExit):
        parser.parse_args(["--composite-loss-weighting", "badalpha"])

    # Bad lr scheduler.
    with pytest.raises(SystemExit):
        parser.parse_args(["--lr-scheduler", "badlr"])

    # Bad optimizer.
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "badopt"])


def test_add_train_args_patch_size_validation(patched_registries):
    """add_train_args enforces patch-size arity and positivity."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    # invalid (negative) triggers ArgumentTypeError from
    # positive_int -> SystemExit in argparse.
    with pytest.raises(SystemExit):
        parser.parse_args(["--patch-size", "32", "-1", "32"])

    # wrong arity (needs exactly 3)
    with pytest.raises(SystemExit):
        parser.parse_args(["--patch-size", "32", "32"])


def test_add_train_args_numeric_validators(patched_registries):
    """add_train_args enforces numeric argument constraints."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    # epochs must be non-negative int
    with pytest.raises(SystemExit):
        parser.parse_args(["--epochs", "-1"])

    # batch-size-per-gpu must be positive int
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch-size-per-gpu", "0"])

    # learning-rate must be positive float
    with pytest.raises(SystemExit):
        parser.parse_args(["--learning-rate", "0"])

    # l2-penalty must be positive float
    with pytest.raises(SystemExit):
        parser.parse_args(["--l2-penalty", "-0.001"])


def test_add_train_args_boolean_flags_explicit_false(patched_registries):
    """add_train_args boolean_flags can be explicitly set to false."""
    parser = _mk_parser()
    args_mod.add_train_args(parser)

    # use-dtms is a boolean_flag; ensure explicit false works.
    ns = parser.parse_args(["--use-dtms", "false"])
    assert ns.use_dtms is False
