from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

NSL_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "U2R", "sendmail": "R2L", "named": "R2L",
    "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    "normal": "Normal",
}

CIC_EARLY_FILES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
)

CIC_FRIDAY_FILES = (
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
)

# Exact Tomek-link cleaning is expensive in high-dimensional traffic data.
# This fixed cap preserves prevalence; the complete Friday test period is retained.
CIC_TEMPORAL_TRAINING_CAP = 250_000
CIC_CAP_SEED = 20260902


@dataclass
class RawDataset:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    class_names: list[str]
    categorical_columns: list[str]
    normal_class_name: str
    protocol: dict = field(default_factory=dict)


def _stratified_cap(
    X: pd.DataFrame,
    y: np.ndarray,
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) > per_class:
            indices = rng.choice(indices, size=per_class, replace=False)
        selected.append(indices)
    joined = np.concatenate(selected)
    joined = joined[rng.permutation(len(joined))]
    return X.iloc[joined].reset_index(drop=True), y[joined]


def _stratified_cap_total(
    X: pd.DataFrame,
    y: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(y) <= maximum:
        return X.reset_index(drop=True), y.copy()
    indices = np.arange(len(y))
    selected, _ = train_test_split(
        indices,
        train_size=maximum,
        random_state=seed,
        stratify=y,
    )
    return X.iloc[selected].reset_index(drop=True), y[selected]


def _normal_name(class_names: list[str]) -> str:
    for name in class_names:
        if name.strip().casefold() in {"normal", "benign"}:
            return name
    raise ValueError(f"No normal/benign class in {class_names}")


def load_unsw(data_dir: str | Path, smoke: bool = False) -> RawDataset:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv", low_memory=False)
    test = pd.read_csv(data_dir / "UNSW_NB15_testing-set.csv", low_memory=False)
    for frame in (train, test):
        frame.drop(
            columns=[c for c in ("id", "Unnamed: 0") if c in frame.columns],
            inplace=True,
            errors="ignore",
        )

    target = "attack_cat"
    excluded = {target, "label"}
    feature_columns = [c for c in train.columns if c not in excluded]
    categorical = [c for c in feature_columns if train[c].dtype == object]

    train_names = train[target].astype(str).str.strip()
    test_names = test[target].astype(str).str.strip()
    class_names = sorted(train_names.unique().tolist())
    class_index = {name: index for index, name in enumerate(class_names)}
    unseen = sorted(set(test_names.unique()) - set(class_names))
    if unseen:
        raise ValueError(f"UNSW test-only classes: {unseen}")

    y_train = train_names.map(class_index).to_numpy(dtype=np.int64)
    y_test = test_names.map(class_index).to_numpy(dtype=np.int64)
    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    if smoke:
        X_train, y_train = _stratified_cap(X_train, y_train, 250, 1701)
        X_test, y_test = _stratified_cap(X_test, y_test, 120, 1702)

    return RawDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        categorical_columns=categorical,
        normal_class_name=_normal_name(class_names),
        protocol={
            "partition_type": "official train/test split",
            "temporal_holdout": False,
            "training_cap": None,
            "test_cap": None,
        },
    )


def _nsl_targets(labels: pd.Series) -> np.ndarray:
    names = labels.astype(str).str.strip().str.rstrip(".")
    unknown = sorted(set(names.unique()) - set(NSL_CATEGORY))
    if unknown:
        raise ValueError(f"Unmapped NSL-KDD attack names: {unknown}")
    class_index = {name: index for index, name in enumerate(NSL_CLASS_NAMES)}
    return names.map(NSL_CATEGORY).map(class_index).to_numpy(dtype=np.int64)


def load_nslkdd(data_dir: str | Path, smoke: bool = False) -> RawDataset:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "KDDTrain+.txt", names=NSL_COLUMNS)
    test = pd.read_csv(data_dir / "KDDTest+.txt", names=NSL_COLUMNS)
    feature_columns = [
        c for c in NSL_COLUMNS if c not in ("label", "difficulty")
    ]
    categorical = ["protocol_type", "service", "flag"]
    y_train = _nsl_targets(train["label"])
    y_test = _nsl_targets(test["label"])
    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    if smoke:
        X_train, y_train = _stratified_cap(X_train, y_train, 250, 2701)
        X_test, y_test = _stratified_cap(X_test, y_test, 120, 2702)

    return RawDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=list(NSL_CLASS_NAMES),
        categorical_columns=categorical,
        normal_class_name="Normal",
        protocol={
            "partition_type": "official train/test split",
            "temporal_holdout": False,
            "training_cap": None,
            "test_cap": None,
        },
    )


def _read_cic_partition(
    data_dir: Path,
    file_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    feature_frames: list[pd.DataFrame] = []
    label_parts: list[pd.Series] = []
    reference_columns: list[str] | None = None
    for file_name in file_names:
        path = data_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        label_columns = [
            column for column in frame.columns
            if column.strip().casefold() == "label"
        ]
        if len(label_columns) != 1:
            raise ValueError(
                f"Expected one Label column in {path}, got {label_columns}"
            )
        labels = (
            frame.pop(label_columns[0])
            .fillna("__MISSING__")
            .astype(str)
            .str.strip()
            .str.replace("\ufffd", "-", regex=False)
        )
        removable = [
            column for column in frame.columns
            if not column or column.casefold().startswith("unnamed:")
        ]
        if removable:
            frame.drop(columns=removable, inplace=True)
        columns = frame.columns.tolist()
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            missing = sorted(set(reference_columns) - set(columns))
            extra = sorted(set(columns) - set(reference_columns))
            raise ValueError(
                f"CIC schema mismatch in {path}: missing={missing}, extra={extra}"
            )
        feature_frames.append(frame)
        label_parts.append(labels)

    features = pd.concat(feature_frames, ignore_index=True, copy=False)
    labels = pd.concat(label_parts, ignore_index=True)
    counts = {
        str(name): int(value) for name, value in labels.value_counts().items()
    }
    return features, labels, counts


def load_cicids2017(data_dir: str | Path, smoke: bool = False) -> RawDataset:
    data_dir = Path(data_dir)
    X_early_full, early_labels, early_label_counts = _read_cic_partition(
        data_dir, CIC_EARLY_FILES
    )
    X_friday, friday_labels, friday_label_counts = _read_cic_partition(
        data_dir, CIC_FRIDAY_FILES
    )
    if X_early_full.columns.tolist() != X_friday.columns.tolist():
        raise ValueError("CIC early-period and Friday feature schemas differ")

    early_is_benign = early_labels.str.casefold().eq("benign").to_numpy()
    friday_is_benign = friday_labels.str.casefold().eq("benign").to_numpy()
    if early_is_benign.all() or friday_is_benign.all():
        raise ValueError(
            "CIC temporal partitions must each contain benign and attack records"
        )
    y_early_full = (~early_is_benign).astype(np.int64)
    y_friday = (~friday_is_benign).astype(np.int64)
    full_early_rows = len(y_early_full)
    full_friday_rows = len(y_friday)

    if smoke:
        X_early, y_early = _stratified_cap(
            X_early_full, y_early_full, 400, CIC_CAP_SEED
        )
        X_friday, y_friday = _stratified_cap(
            X_friday, y_friday, 250, CIC_CAP_SEED + 1
        )
        training_cap: int | None = 800
        test_cap: int | None = 500
    else:
        X_early, y_early = _stratified_cap_total(
            X_early_full,
            y_early_full,
            CIC_TEMPORAL_TRAINING_CAP,
            CIC_CAP_SEED,
        )
        X_friday = X_friday.reset_index(drop=True)
        training_cap = CIC_TEMPORAL_TRAINING_CAP
        test_cap = None

    early_attacks = {
        name
        for name in early_label_counts
        if name.strip().casefold() != "benign"
    }
    friday_attacks = {
        name
        for name in friday_label_counts
        if name.strip().casefold() != "benign"
    }
    return RawDataset(
        X_train=X_early,
        y_train=y_early,
        X_test=X_friday,
        y_test=y_friday,
        class_names=["BENIGN", "ATTACK"],
        categorical_columns=[],
        normal_class_name="BENIGN",
        protocol={
            "partition_type": "cross-day temporal holdout",
            "temporal_holdout": True,
            "training_period": "Monday through Thursday",
            "test_period": "Friday",
            "training_files": list(CIC_EARLY_FILES),
            "test_files": list(CIC_FRIDAY_FILES),
            "full_training_period_rows_before_cap": full_early_rows,
            "full_test_period_rows": full_friday_rows,
            "training_cap": training_cap,
            "training_cap_seed": CIC_CAP_SEED,
            "training_cap_policy": (
                "fixed stratified sample preserving binary prevalence"
            ),
            "test_cap": test_cap,
            "full_training_label_counts": early_label_counts,
            "full_test_label_counts": friday_label_counts,
            "training_attack_labels": sorted(early_attacks),
            "test_attack_labels": sorted(friday_attacks),
            "test_only_attack_labels": sorted(friday_attacks - early_attacks),
            "binary_task": "BENIGN versus ATTACK",
            "source_label_normalization": (
                "Unicode replacement characters in official Web Attack labels "
                "are normalized to ASCII hyphens; class membership is unchanged"
            ),
        },
    )


LOADERS = {
    "unsw": load_unsw,
    "nslkdd": load_nslkdd,
    "cicids2017": load_cicids2017,
}


class TrainOnlyPreprocessor:
    """Encode and impute using the sub-training partition only."""

    def __init__(self, categorical_columns: list[str]):
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.onehot: OneHotEncoder | None = None
        self.original_feature_names: list[str] = []
        self.expanded_feature_names: list[str] = []
        self.expanded_to_original: np.ndarray | None = None

    @staticmethod
    def _category_text(series: pd.Series) -> pd.Series:
        return series.fillna("__MISSING__").astype(str)

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.numeric_columns = [
            c for c in frame.columns if c not in self.categorical_columns
        ]
        kept_numeric: list[str] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            if float(filled.std()) > 1e-12:
                kept_numeric.append(column)
                self.numeric_medians[column] = median
        self.numeric_columns = kept_numeric

        for column in self.categorical_columns:
            categories = sorted(self._category_text(frame[column]).unique().tolist())
            self.category_maps[column] = {
                value: index for index, value in enumerate(categories)
            }

        self.original_feature_names = self.numeric_columns + self.categorical_columns
        if self.categorical_columns:
            encoded = self.encode(frame)
            cat_values = encoded[:, len(self.numeric_columns):]
            self.onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
            self.onehot.fit(cat_values)
            expanded_cat = self.onehot.get_feature_names_out(
                self.categorical_columns
            ).tolist()
            self.expanded_feature_names = self.numeric_columns + expanded_cat
            group_ids = list(range(len(self.numeric_columns)))
            for offset, categories in enumerate(self.onehot.categories_):
                group_ids.extend(
                    [len(self.numeric_columns) + offset] * len(categories)
                )
        else:
            self.onehot = None
            self.expanded_feature_names = list(self.numeric_columns)
            group_ids = list(range(len(self.numeric_columns)))
        self.expanded_to_original = np.asarray(group_ids, dtype=np.int64)
        return self

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        numeric_parts = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            numeric_parts.append(
                values.fillna(self.numeric_medians[column]).to_numpy(dtype=np.float32)
            )
        numeric = (
            np.column_stack(numeric_parts).astype(np.float32)
            if numeric_parts
            else np.empty((len(frame), 0), dtype=np.float32)
        )

        categorical_parts = []
        for column in self.categorical_columns:
            mapping = self.category_maps[column]
            categorical_parts.append(
                self._category_text(frame[column])
                .map(mapping)
                .fillna(-1)
                .to_numpy(dtype=np.float32)
            )
        categorical = (
            np.column_stack(categorical_parts).astype(np.float32)
            if categorical_parts
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        return np.concatenate([numeric, categorical], axis=1)

    def classifier_matrix(self, encoded: np.ndarray) -> np.ndarray:
        split = len(self.numeric_columns)
        numeric = encoded[:, :split].astype(np.float32, copy=False)
        if not self.categorical_columns:
            return numeric
        if self.onehot is None:
            raise RuntimeError("Preprocessor has not been fitted")
        categorical = self.onehot.transform(encoded[:, split:]).astype(
            np.float32, copy=False
        )
        return np.concatenate([numeric, categorical], axis=1)

    @property
    def categorical_indices(self) -> list[int]:
        start = len(self.numeric_columns)
        return list(range(start, start + len(self.categorical_columns)))

    def audit(self) -> dict:
        if self.expanded_to_original is None:
            raise RuntimeError("Preprocessor has not been fitted")
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "original_feature_names": self.original_feature_names,
            "expanded_feature_names": self.expanded_feature_names,
            "expanded_to_original": self.expanded_to_original.tolist(),
            "category_counts": {
                column: len(mapping)
                for column, mapping in self.category_maps.items()
            },
            "fit_scope": "sub-training partition only",
            "unknown_category_policy": (
                "one-hot all-zero vector"
                if self.categorical_columns
                else "not applicable"
            ),
            "infinite_value_policy": "replace with sub-training median",
        }


def split_official_training(
    dataset: RawDataset,
    seed: int,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return train_test_split(
        dataset.X_train,
        dataset.y_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=dataset.y_train,
    )


def class_counts(y: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = np.bincount(y, minlength=n_classes)
    return {str(index): int(value) for index, value in enumerate(counts)}


def moderate_targets(
    y: np.ndarray,
    n_classes: int,
    expansion_cap: int = 15,
) -> dict[int, int]:
    counts = np.bincount(y, minlength=n_classes)
    positive = counts[counts > 0]
    target = int(np.floor(positive.mean()))
    strategy: dict[int, int] = {}
    for label, count in enumerate(counts):
        if count <= 0 or count >= target:
            continue
        strategy[label] = int(min(target, count * expansion_cap))
    return strategy
