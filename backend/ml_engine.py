from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_SEED = 42

CATEGORY_SYNONYMS = {
    "dining": "restaurants",
    "food": "restaurants",
    "coffee": "restaurants",
    "takeout": "restaurants",
    "ride": "transport",
    "transportation": "transport",
    "fuel": "transport",
    "gas": "transport",
    "bills": "utilities",
    "phone": "utilities",
    "internet": "utilities",
    "streaming": "subscriptions",
    "software": "subscriptions",
    "memberships": "subscriptions",
    "medical": "health",
    "pharmacy": "health",
    "income": "income",
    "salary": "income",
}

TARGET_CATEGORIES = [
    "groceries",
    "restaurants",
    "transport",
    "shopping",
    "utilities",
    "subscriptions",
    "travel",
    "health",
    "income",
    "entertainment",
    "education",
    "insurance",
    "cash",
]

SUBSCRIPTION_LIKE_CATEGORIES = {"subscriptions", "entertainment", "software"}
KNOWN_SUBSCRIPTION_MERCHANTS = {
    "netflix",
    "spotify",
    "icloud",
    "adobe",
    "hulu",
    "apple one",
    "chatgpt",
    "youtube premium",
}

PRIORITY_WEIGHTS = {
    "education": 2.0,
    "health": 1.4,
    "groceries": 1.0,
    "utilities": 0.9,
    "insurance": 0.8,
    "transport": 0.3,
    "income": 0.0,
    "travel": -0.3,
    "cash": -0.2,
    "restaurants": -0.9,
    "subscriptions": -1.0,
    "shopping": -1.0,
    "entertainment": -1.3,
}

DISCRETIONARY_CATEGORIES = {"restaurants", "shopping", "entertainment", "subscriptions"}


@dataclass
class AnalysisResult:
    month: str
    monthly_spending: float
    biggest_category: str
    category_breakdown: list[dict[str, object]]
    recurring_subscriptions: list[dict[str, object]]
    subscription_waste: float
    anomalies: list[dict[str, object]]
    review_queue: list[dict[str, object]]
    categorized_transactions: list[dict[str, object]]
    monthly_trend: list[dict[str, object]]
    month_over_month: dict[str, object]
    financial_score: dict[str, object]
    spending_priorities: list[dict[str, object]]
    forecast: dict[str, object]
    tips: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "month": self.month,
            "monthly_spending": self.monthly_spending,
            "biggest_category": self.biggest_category,
            "category_breakdown": self.category_breakdown,
            "recurring_subscriptions": self.recurring_subscriptions,
            "subscription_waste": self.subscription_waste,
            "anomalies": self.anomalies,
            "review_queue": self.review_queue,
            "categorized_transactions": self.categorized_transactions,
            "monthly_trend": self.monthly_trend,
            "month_over_month": self.month_over_month,
            "financial_score": self.financial_score,
            "spending_priorities": self.spending_priorities,
            "forecast": self.forecast,
            "tips": self.tips,
        }


class PersonalFinanceMLEngine:
    def __init__(self, backend_dir: Path) -> None:
        self.backend_dir = backend_dir
        self.models_dir = backend_dir / "models"
        self.metadata_path = self.models_dir / "category_metadata.json"
        self.category_model_path = self.models_dir / "category_model.joblib"
        self.open_datasets_dir = backend_dir / "datasets" / "open"
        self.sample_path = backend_dir / "data" / "sample_transactions.csv"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.open_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.category_pipeline: Pipeline | None = None
        self.model_metadata: dict[str, object] = {}

    def bootstrap(self) -> None:
        if self.category_model_path.exists() and self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if metadata.get("sklearn_version") == sklearn.__version__:
                    self.category_pipeline = joblib.load(self.category_model_path)
                    self.model_metadata = metadata
                    return
            except Exception:
                pass
        self.train_and_save_category_model()

    def train_and_save_category_model(self) -> dict[str, object]:
        training_frame = self._load_training_frame()
        training_frame["description"] = training_frame["description"].fillna("").astype(str)
        training_frame["merchant"] = training_frame["merchant"].fillna("").astype(str)
        training_frame["category"] = (
            training_frame["category"].fillna("").astype(str).map(self._normalize_category)
        )
        training_frame["amount"] = pd.to_numeric(training_frame["amount"], errors="coerce").fillna(0.0)
        training_frame = training_frame[training_frame["category"].isin(TARGET_CATEGORIES)].copy()
        training_frame["amount_abs"] = training_frame["amount"].abs()
        training_frame["text"] = self._make_text_feature(training_frame)
        training_frame = training_frame[training_frame["text"].str.len() > 0].copy()

        category_counts = training_frame["category"].value_counts()
        valid_categories = category_counts[category_counts >= 8].index.tolist()
        training_frame = training_frame[training_frame["category"].isin(valid_categories)].copy()
        if training_frame["category"].nunique() < 2:
            raise ValueError("Need at least 2 categories with enough rows to train.")

        x = training_frame[["text", "amount_abs"]]
        y = training_frame["category"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        leaderboard: list[dict[str, object]] = []
        best_pipeline: Pipeline | None = None
        best_entry: dict[str, object] | None = None

        for candidate in self._candidate_pipelines():
            name = candidate["name"]
            pipeline = candidate["pipeline"]
            try:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    cv_scores = cross_validate(
                        pipeline,
                        x_train,
                        y_train,
                        cv=cv,
                        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
                        n_jobs=None,
                    )
                    pipeline.fit(x_train, y_train)

                predictions = pipeline.predict(x_test)
                test_macro_f1 = float(f1_score(y_test, predictions, average="macro"))
                test_accuracy = float(accuracy_score(y_test, predictions))
                probabilities = self._estimate_probabilities(pipeline, x_test)
                avg_confidence = float(np.max(probabilities, axis=1).mean())

                row = {
                    "model": name,
                    "cv_macro_f1": float(np.mean(cv_scores["test_macro_f1"])),
                    "cv_accuracy": float(np.mean(cv_scores["test_accuracy"])),
                    "test_macro_f1": test_macro_f1,
                    "test_accuracy": test_accuracy,
                    "test_avg_confidence": avg_confidence,
                }
                leaderboard.append(row)

                if best_entry is None or self._is_better_model(row, best_entry):
                    best_entry = row
                    best_pipeline = pipeline
            except Exception as exc:
                leaderboard.append(
                    {
                        "model": name,
                        "error": str(exc),
                        "cv_macro_f1": 0.0,
                        "cv_accuracy": 0.0,
                        "test_macro_f1": 0.0,
                        "test_accuracy": 0.0,
                        "test_avg_confidence": 0.0,
                    }
                )

        if best_pipeline is None or best_entry is None:
            raise RuntimeError("AutoML could not train any valid category model.")

        self.category_pipeline = best_pipeline
        joblib.dump(best_pipeline, self.category_model_path)

        leaderboard_sorted = sorted(
            leaderboard,
            key=lambda row: (
                float(row.get("test_macro_f1", 0.0)),
                float(row.get("cv_macro_f1", 0.0)),
                float(row.get("test_accuracy", 0.0)),
            ),
            reverse=True,
        )
        metadata = {
            "automl": True,
            "selected_model": best_entry["model"],
            "trained_rows": int(len(training_frame)),
            "category_count": int(training_frame["category"].nunique()),
            "categories": sorted(training_frame["category"].unique().tolist()),
            "metrics": {
                "cv_macro_f1": round(float(best_entry["cv_macro_f1"]), 4),
                "cv_accuracy": round(float(best_entry["cv_accuracy"]), 4),
                "test_macro_f1": round(float(best_entry["test_macro_f1"]), 4),
                "test_accuracy": round(float(best_entry["test_accuracy"]), 4),
                "test_avg_confidence": round(float(best_entry["test_avg_confidence"]), 4),
            },
            "leaderboard": leaderboard_sorted,
            "sklearn_version": sklearn.__version__,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.model_metadata = metadata
        return metadata

    def load_sample_transactions(self) -> list[dict[str, object]]:
        frame = pd.read_csv(self.sample_path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
        return [self._json_clean(row) for row in frame.to_dict(orient="records")]

    def analyze(self, transactions: list[dict[str, object]], month: str | None = None) -> AnalysisResult:
        if self.category_pipeline is None:
            raise RuntimeError("Category model not loaded")

        frame = pd.DataFrame(transactions).copy()
        if frame.empty:
            raise ValueError("At least one transaction is required")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).copy()
        if frame.empty:
            raise ValueError("No valid transaction dates were provided")

        frame["description"] = frame["description"].fillna("").astype(str)
        frame["merchant"] = frame.get("merchant", "").fillna("").astype(str)
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0).astype(float)
        frame["type"] = frame.get("type", "debit").fillna("debit").astype(str).str.lower()
        frame["month"] = frame["date"].dt.to_period("M").astype(str)
        frame["spend"] = frame.apply(self._spend_amount, axis=1)

        categorized = self._categorize_transactions(frame)
        all_months = sorted(categorized["month"].unique().tolist())
        if month is None:
            month = all_months[-1]

        month_frame = categorized[categorized["month"] == month].copy()
        if month_frame.empty:
            raise ValueError(f"No transactions found for requested month: {month}")

        monthly_spending = float(month_frame["spend"].sum())
        category_spend = (
            month_frame.groupby("predicted_category", as_index=False)["spend"]
            .sum()
            .sort_values("spend", ascending=False)
        )
        biggest_category = (
            str(category_spend.iloc[0]["predicted_category"]) if not category_spend.empty else "none"
        )
        category_breakdown = [
            {
                "category": row["predicted_category"],
                "amount": round(float(row["spend"]), 2),
                "share": round(float(row["spend"]) / monthly_spending, 4) if monthly_spending else 0.0,
            }
            for row in category_spend.to_dict(orient="records")
        ]

        recurring_subscriptions = self._detect_subscriptions(categorized, month)
        subscription_waste = round(float(sum(item["waste_estimate"] for item in recurring_subscriptions)), 2)
        anomalies, _ = self._detect_anomaly_signals(categorized, month)
        monthly_trend = self._monthly_trend(categorized)
        month_over_month = self._month_over_month(monthly_trend, month)
        spending_priorities = self._spending_priorities(category_breakdown)
        financial_score = self._financial_score(
            category_breakdown=category_breakdown,
            subscription_waste=subscription_waste,
            monthly_spending=monthly_spending,
            anomaly_count=len(anomalies),
            month_over_month=month_over_month,
        )
        forecast = self._forecast_next_month(
            monthly_trend=monthly_trend,
            category_breakdown=category_breakdown,
            subscription_waste=subscription_waste,
        )
        tips = self._generate_tips(
            month=month,
            month_over_month=month_over_month,
            category_breakdown=category_breakdown,
            subscription_waste=subscription_waste,
            anomalies=anomalies,
            forecast=forecast,
            score=financial_score["score"],
        )

        transaction_view = [
            {
                "date": row["date"].date().isoformat(),
                "description": row["description"],
                "merchant": row["merchant"],
                "amount": round(float(row["amount"]), 2),
                "type": row["type"],
                "category": row["predicted_category"],
                "category_confidence": round(float(row["category_confidence"]), 3),
                "spend": round(float(row["spend"]), 2),
            }
            for row in month_frame.sort_values("date").to_dict(orient="records")
        ]
        review_queue = [
            {
                "date": row["date"].date().isoformat(),
                "description": row["description"],
                "merchant": row["merchant"],
                "amount": round(float(row["spend"]), 2),
                "category": row["predicted_category"],
                "category_confidence": round(float(row["category_confidence"]) * 100.0, 1),
            }
            for row in month_frame[month_frame["category_confidence"] < 0.62]
            .sort_values("category_confidence")
            .head(10)
            .to_dict(orient="records")
        ]

        return AnalysisResult(
            month=month,
            monthly_spending=round(monthly_spending, 2),
            biggest_category=biggest_category,
            category_breakdown=category_breakdown,
            recurring_subscriptions=recurring_subscriptions,
            subscription_waste=subscription_waste,
            anomalies=anomalies,
            review_queue=review_queue,
            categorized_transactions=transaction_view,
            monthly_trend=monthly_trend,
            month_over_month=month_over_month,
            financial_score=financial_score,
            spending_priorities=spending_priorities,
            forecast=forecast,
            tips=tips,
        )

    def _candidate_pipelines(self) -> list[dict[str, object]]:
        return [
            {
                "name": "logistic_regression",
                "pipeline": Pipeline(
                    steps=[
                        ("preprocess", self._base_preprocessor()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=3200,
                                class_weight="balanced",
                                solver="lbfgs",
                            ),
                        ),
                    ]
                ),
            },
            {
                "name": "sgd_logistic",
                "pipeline": Pipeline(
                    steps=[
                        ("preprocess", self._base_preprocessor()),
                        (
                            "classifier",
                            SGDClassifier(
                                loss="log_loss",
                                alpha=1e-5,
                                penalty="l2",
                                class_weight="balanced",
                                max_iter=4000,
                                tol=1e-4,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                ),
            },
            {
                "name": "calibrated_linear_svm",
                "pipeline": Pipeline(
                    steps=[
                        ("preprocess", self._base_preprocessor()),
                        (
                            "classifier",
                            CalibratedClassifierCV(
                                estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_SEED),
                                method="sigmoid",
                                cv=3,
                            ),
                        ),
                    ]
                ),
            },
            {
                "name": "complement_nb",
                "pipeline": Pipeline(
                    steps=[
                        ("preprocess", self._base_preprocessor()),
                        ("classifier", ComplementNB(alpha=0.7)),
                    ]
                ),
            },
        ]

    def _base_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), "text"),
                ("amount", "passthrough", ["amount_abs"]),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

    def _is_better_model(self, candidate: dict[str, object], best: dict[str, object]) -> bool:
        return (
            float(candidate["test_macro_f1"]),
            float(candidate["cv_macro_f1"]),
            float(candidate["test_accuracy"]),
            float(candidate["test_avg_confidence"]),
        ) > (
            float(best["test_macro_f1"]),
            float(best["cv_macro_f1"]),
            float(best["test_accuracy"]),
            float(best["test_avg_confidence"]),
        )

    def _load_training_frame(self) -> pd.DataFrame:
        frames = [self._seed_training_frame()]
        for csv_path in sorted(self.open_datasets_dir.glob("*.csv")):
            parsed = self._parse_open_dataset(csv_path)
            if parsed is not None and not parsed.empty:
                frames.append(parsed)
        merged = pd.concat(frames, ignore_index=True)
        return merged.dropna(subset=["description", "category"]).copy()

    def _parse_open_dataset(self, csv_path: Path) -> pd.DataFrame | None:
        try:
            raw = pd.read_csv(csv_path)
        except Exception:
            return None

        columns = {column.lower(): column for column in raw.columns}
        if {"description", "amount", "category"}.issubset(columns):
            merchant_col = columns.get("merchant")
            merchant_series = raw[merchant_col] if merchant_col else raw[columns["description"]]
            return pd.DataFrame(
                {
                    "description": raw[columns["description"]],
                    "merchant": merchant_series,
                    "amount": raw[columns["amount"]],
                    "category": raw[columns["category"]],
                }
            )

        if {"transaction_description", "amount", "category"}.issubset(columns):
            merchant_col = columns.get("merchant_name")
            merchant_series = raw[merchant_col] if merchant_col else raw[columns["transaction_description"]]
            return pd.DataFrame(
                {
                    "description": raw[columns["transaction_description"]],
                    "merchant": merchant_series,
                    "amount": raw[columns["amount"]],
                    "category": raw[columns["category"]],
                }
            )
        return None

    def _seed_training_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        rng = np.random.default_rng(RANDOM_SEED)
        category_templates: dict[str, list[tuple[str, str, tuple[float, float]]]] = {
            "groceries": [
                ("Whole Foods", "grocery store purchase", (38, 170)),
                ("Trader Joe's", "weekly groceries", (35, 140)),
                ("Costco", "bulk groceries", (55, 240)),
            ],
            "restaurants": [
                ("Sweetgreen", "lunch order", (14, 36)),
                ("DoorDash", "takeout dinner", (22, 72)),
                ("Blue Bottle", "coffee and pastry", (6, 20)),
                ("Chipotle", "restaurant meal", (11, 33)),
            ],
            "transport": [
                ("Uber", "rideshare trip", (9, 52)),
                ("Lyft", "rideshare fare", (8, 45)),
                ("Chevron", "fuel station payment", (35, 110)),
                ("City Metro", "metro card reload", (15, 65)),
            ],
            "shopping": [
                ("Amazon", "online shopping order", (22, 280)),
                ("Target", "retail purchase", (16, 170)),
                ("Apple Store", "electronics purchase", (90, 1100)),
            ],
            "utilities": [
                ("Comcast", "internet bill payment", (52, 130)),
                ("AT&T", "phone bill", (45, 95)),
                ("ConEdison", "electric utility bill", (70, 170)),
            ],
            "subscriptions": [
                ("Netflix", "monthly streaming subscription", (15, 24)),
                ("Spotify", "music subscription charge", (9, 18)),
                ("Adobe", "creative cloud subscription", (35, 68)),
            ],
            "travel": [
                ("Delta", "flight booking", (140, 850)),
                ("Marriott", "hotel reservation", (120, 520)),
                ("Hertz", "rental car charge", (65, 260)),
            ],
            "health": [
                ("CVS Pharmacy", "pharmacy purchase", (12, 58)),
                ("City Clinic", "medical copay", (25, 140)),
                ("Optum", "healthcare bill", (55, 210)),
            ],
            "income": [
                ("Employer Payroll", "salary deposit", (2200, 6500)),
                ("Freelance Client", "freelance income payment", (350, 2600)),
                ("Tax Return", "tax refund deposit", (300, 1800)),
            ],
            "entertainment": [
                ("AMC", "movie tickets", (18, 54)),
                ("Ticketmaster", "concert tickets", (45, 260)),
                ("Steam", "gaming purchase", (12, 88)),
            ],
            "education": [
                ("Coursera", "course enrollment", (29, 129)),
                ("Udemy", "online class purchase", (15, 98)),
                ("Bookstore", "study books", (18, 130)),
            ],
            "insurance": [
                ("State Farm", "insurance premium", (85, 240)),
                ("Geico", "auto insurance", (75, 210)),
                ("Lemonade", "renters insurance", (11, 35)),
            ],
            "cash": [
                ("ATM Withdrawal", "cash withdrawal", (40, 240)),
                ("Venmo", "cash transfer to friend", (18, 120)),
                ("CashApp", "p2p cash transfer", (16, 115)),
            ],
        }
        for category, templates in category_templates.items():
            for merchant, description, (low, high) in templates:
                for _ in range(24):
                    value = float(np.round(rng.uniform(low, high), 2))
                    noise = float(np.round(rng.normal(0, max(value * 0.03, 1.0)), 2))
                    rows.append(
                        {
                            "description": description,
                            "merchant": merchant,
                            "amount": max(value + noise, 1.0),
                            "category": category,
                        }
                    )
        frame = pd.DataFrame(rows)
        # Inject small label noise so synthetic seed data does not look unrealistically perfect.
        noise_count = max(int(len(frame) * 0.08), 1)
        noise_indices = rng.choice(frame.index.to_numpy(), size=noise_count, replace=False)
        for index in noise_indices:
            original = str(frame.at[index, "category"])
            alternatives = [category for category in TARGET_CATEGORIES if category != original]
            frame.at[index, "category"] = rng.choice(alternatives)
        return frame

    def _make_text_feature(self, frame: pd.DataFrame) -> pd.Series:
        amount_bucket = pd.cut(
            frame["amount"].abs(),
            bins=[-1, 15, 40, 100, 300, 1000, np.inf],
            labels=["tiny", "small", "medium", "large", "xlarge", "xxlarge"],
        ).astype(str)
        return (
            frame["description"].str.lower().fillna("")
            + " "
            + frame["merchant"].str.lower().fillna("")
            + " amount_"
            + amount_bucket
        )

    def _normalize_category(self, category: str) -> str:
        normalized = category.strip().lower().replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())
        return CATEGORY_SYNONYMS.get(normalized, normalized)

    def _categorize_transactions(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["amount_abs"] = prepared["amount"].abs()
        prepared["text"] = self._make_text_feature(prepared)
        probabilities = self._estimate_probabilities(self.category_pipeline, prepared[["text", "amount_abs"]])
        classes = self._classifier_classes(self.category_pipeline)
        top_indices = probabilities.argmax(axis=1)
        predicted = classes[top_indices]
        confidence = probabilities[np.arange(len(prepared)), top_indices]
        prepared["predicted_category"] = predicted
        prepared["category_confidence"] = confidence

        for idx, row in prepared.iterrows():
            if float(row["category_confidence"]) >= 0.52:
                continue
            fallback = self._keyword_fallback(str(row["description"]), str(row["merchant"]), float(row["amount"]))
            if fallback is not None:
                prepared.at[idx, "predicted_category"] = fallback
                prepared.at[idx, "category_confidence"] = max(float(row["category_confidence"]), 0.55)
        return prepared

    def _keyword_fallback(self, description: str, merchant: str, amount: float) -> str | None:
        text = f"{description} {merchant}".lower()
        keywords = {
            "netflix": "subscriptions",
            "spotify": "subscriptions",
            "adobe": "subscriptions",
            "coursera": "education",
            "udemy": "education",
            "linkedin learning": "education",
            "uber": "transport",
            "lyft": "transport",
            "whole foods": "groceries",
            "trader joe": "groceries",
            "doordash": "restaurants",
            "restaurant": "restaurants",
            "salary": "income",
            "payroll": "income",
            "insurance": "insurance",
            "pharmacy": "health",
            "clinic": "health",
            "movie": "entertainment",
            "concert": "entertainment",
            "atm": "cash",
        }
        for keyword, category in keywords.items():
            if keyword in text:
                return category
        if amount > 1800:
            return "shopping"
        return None

    def _spend_amount(self, row: pd.Series) -> float:
        amount = float(row["amount"])
        tx_type = str(row["type"]).strip().lower()
        if tx_type == "credit":
            return 0.0
        if amount < 0:
            return abs(amount)
        return amount

    def _detect_subscriptions(self, frame: pd.DataFrame, month: str) -> list[dict[str, object]]:
        expenses = frame[frame["spend"] > 0].copy()
        if expenses.empty:
            return []

        expenses["merchant_key"] = expenses["merchant"].fillna("").str.lower().str.strip().replace("", "unknown")
        expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
        current_month = expenses[expenses["month"] == month]
        results: list[dict[str, object]] = []

        for merchant, group in expenses.groupby("merchant_key"):
            group = group.sort_values("date")
            current_rows = current_month[current_month["merchant_key"] == merchant]
            if current_rows.empty:
                continue

            recurring = False
            if len(group) >= 2:
                diffs = group["date"].diff().dropna().dt.days
                recurring = bool(((diffs >= 25) & (diffs <= 35)).any())
            if not recurring and merchant not in KNOWN_SUBSCRIPTION_MERCHANTS:
                continue

            monthly_estimate = float(current_rows["spend"].mean())
            category = str(current_rows["predicted_category"].mode().iloc[0])
            if category in {"education", "health"}:
                waste_weight = 0.25
            elif category in SUBSCRIPTION_LIKE_CATEGORIES:
                waste_weight = 1.0
            else:
                waste_weight = 0.55
            results.append(
                {
                    "merchant": str(current_rows.iloc[0]["merchant"] or merchant),
                    "category": category,
                    "monthly_estimate": round(monthly_estimate, 2),
                    "waste_estimate": round(monthly_estimate * waste_weight, 2),
                    "priority": "keep" if category in {"education", "health"} else "review",
                }
            )
        return sorted(results, key=lambda row: row["waste_estimate"], reverse=True)

    def _detect_anomaly_signals(
        self,
        frame: pd.DataFrame,
        month: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        expenses = frame[frame["spend"] > 0].copy()
        if len(expenses) < 8:
            return [], []

        expenses["merchant_key"] = expenses["merchant"].fillna("").str.lower().str.strip()
        merchant_freq = expenses["merchant_key"].value_counts(normalize=True)
        category_freq = expenses["predicted_category"].value_counts(normalize=True)
        expenses["merchant_freq"] = expenses["merchant_key"].map(merchant_freq).fillna(0.01)
        expenses["category_freq"] = expenses["predicted_category"].map(category_freq).fillna(0.01)
        expenses["day"] = expenses["date"].dt.day.astype(float)
        expenses["weekday"] = expenses["date"].dt.weekday.astype(float)
        expenses["amount_log"] = np.log1p(expenses["spend"])

        model = IsolationForest(
            n_estimators=300,
            contamination=min(0.15, 4 / len(expenses)),
            random_state=RANDOM_SEED,
        )
        feature_columns = ["amount_log", "day", "weekday", "merchant_freq", "category_freq", "category_confidence"]
        model.fit(expenses[feature_columns])
        expenses["iso_label"] = model.predict(expenses[feature_columns])
        expenses["iso_score"] = -model.score_samples(expenses[feature_columns])

        baseline = expenses.groupby("predicted_category")["spend"].agg(["median", "std"]).reset_index()
        baseline_map = {
            row["predicted_category"]: (float(row["median"]), max(float(row["std"]), 1.0))
            for row in baseline.to_dict(orient="records")
        }
        z_scores = []
        for row in expenses.to_dict(orient="records"):
            med, std = baseline_map.get(row["predicted_category"], (0.0, 1.0))
            z_scores.append((float(row["spend"]) - med) / std)
        expenses["category_z"] = z_scores
        expenses["combined"] = 0.65 * self._minmax(expenses["iso_score"].to_numpy()) + 0.35 * self._minmax(
            np.abs(expenses["category_z"].to_numpy())
        )

        month_expenses = expenses[expenses["month"] == month].copy()
        if month_expenses.empty:
            return [], []

        threshold = max(float(month_expenses["combined"].quantile(0.85)), 0.6)
        flagged = month_expenses[(month_expenses["combined"] >= threshold) | (month_expenses["iso_label"] == -1)]
        if flagged.empty:
            flagged = month_expenses.sort_values("combined", ascending=False).head(3)

        median_spend = float(month_expenses["spend"].median())
        month_q95 = float(month_expenses["spend"].quantile(0.95))

        anomalies: list[dict[str, object]] = []
        for row in flagged.sort_values("combined", ascending=False).head(8).to_dict(orient="records"):
            raw_combined = float(row.get("combined", 0.0) or 0.0)
            if not np.isfinite(raw_combined):
                raw_combined = 0.0
            confidence = round(float(np.clip(raw_combined * 100.0, 0.0, 100.0)), 1)
            severity = "high" if raw_combined > 0.8 else "medium"
            anomalies.append(
                {
                    "date": row["date"].date().isoformat(),
                    "merchant": row["merchant"],
                    "description": row["description"],
                    "amount": round(float(row["spend"]), 2),
                    "category": row["predicted_category"],
                    "anomaly_score": round(raw_combined, 3),
                    "anomaly_confidence": confidence,
                    "severity": severity,
                    "reasons": self._anomaly_reasons(row, month_q95, median_spend),
                }
            )

        return anomalies, []

    def _anomaly_reasons(self, row: dict[str, object], month_q95: float, median_spend: float) -> list[str]:
        reasons = []
        spend = float(row.get("spend", 0.0) or 0.0)
        merchant_freq = float(row.get("merchant_freq", 0.0) or 0.0)
        category_confidence = float(row.get("category_confidence", 0.0) or 0.0)
        if spend > month_q95:
            reasons.append("top 5% amount for the month")
        if spend > 2.4 * max(median_spend, 1.0):
            reasons.append("much higher than your median transaction")
        if merchant_freq < 0.05:
            reasons.append("merchant appears rarely in your history")
        if category_confidence < 0.45:
            reasons.append("description pattern is uncommon for this category")
        if not reasons:
            reasons.append("overall spending pattern is atypical")
        return reasons

    def _monthly_trend(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        grouped = frame.groupby("month", as_index=False)["spend"].sum().sort_values("month")
        return [{"month": row["month"], "amount": round(float(row["spend"]), 2)} for row in grouped.to_dict("records")]

    def _month_over_month(self, monthly_trend: list[dict[str, object]], month: str) -> dict[str, object]:
        month_to_amount = {row["month"]: float(row["amount"]) for row in monthly_trend}
        sorted_months = sorted(month_to_amount.keys())
        if month not in month_to_amount:
            return {"previous_month": None, "previous_spending": 0.0, "delta_amount": 0.0, "delta_pct": 0.0}

        idx = sorted_months.index(month)
        if idx == 0:
            return {
                "previous_month": None,
                "previous_spending": 0.0,
                "delta_amount": 0.0,
                "delta_pct": 0.0,
            }

        previous_month = sorted_months[idx - 1]
        previous_spending = month_to_amount[previous_month]
        delta_amount = month_to_amount[month] - previous_spending
        delta_pct = (delta_amount / previous_spending) if previous_spending > 0 else 0.0
        return {
            "previous_month": previous_month,
            "previous_spending": round(previous_spending, 2),
            "delta_amount": round(delta_amount, 2),
            "delta_pct": round(delta_pct, 4),
            "direction": "up" if delta_amount > 0 else "down" if delta_amount < 0 else "flat",
        }

    def _spending_priorities(self, category_breakdown: list[dict[str, object]]) -> list[dict[str, object]]:
        priorities = []
        for row in category_breakdown:
            category = str(row["category"])
            share = float(row["share"])
            weight = PRIORITY_WEIGHTS.get(category, -0.2)
            points = round(share * 100 * weight, 2)
            signal = "good" if points >= 0 else "reduce"
            priorities.append(
                {
                    "category": category,
                    "share": round(share, 4),
                    "points": points,
                    "signal": signal,
                }
            )
        return sorted(priorities, key=lambda item: item["points"], reverse=True)

    def _financial_score(
        self,
        category_breakdown: list[dict[str, object]],
        subscription_waste: float,
        monthly_spending: float,
        anomaly_count: int,
        month_over_month: dict[str, object],
    ) -> dict[str, object]:
        shares = {row["category"]: float(row["share"]) for row in category_breakdown}
        positive = (
            shares.get("education", 0) * 18
            + shares.get("health", 0) * 12
            + shares.get("groceries", 0) * 8
            + shares.get("utilities", 0) * 5
        )
        discretionary_share = sum(shares.get(category, 0) for category in DISCRETIONARY_CATEGORIES)
        waste_ratio = subscription_waste / max(monthly_spending, 1.0)
        trend_penalty = max(float(month_over_month.get("delta_pct", 0.0)), 0.0) * 14
        score = 70 + positive - discretionary_share * 35 - waste_ratio * 42 - anomaly_count * 2.0 - trend_penalty
        score = max(0, min(100, score))
        if score >= 85:
            grade = "A"
        elif score >= 72:
            grade = "B"
        elif score >= 58:
            grade = "C"
        elif score >= 45:
            grade = "D"
        else:
            grade = "E"

        return {
            "score": round(score, 1),
            "grade": grade,
            "discretionary_share": round(discretionary_share, 4),
            "waste_ratio": round(waste_ratio, 4),
        }

    def _forecast_next_month(
        self,
        monthly_trend: list[dict[str, object]],
        category_breakdown: list[dict[str, object]],
        subscription_waste: float,
    ) -> dict[str, object]:
        amounts = [float(row["amount"]) for row in sorted(monthly_trend, key=lambda row: row["month"])]
        if len(amounts) >= 2:
            indices = np.arange(len(amounts), dtype=float)
            slope, intercept = np.polyfit(indices, np.array(amounts, dtype=float), 1)
            baseline = float(intercept + slope * len(amounts))
        else:
            baseline = amounts[-1] if amounts else 0.0

        current_spending = amounts[-1] if amounts else 0.0
        shares = {row["category"]: float(row["share"]) for row in category_breakdown}
        discretionary_spending = current_spending * sum(shares.get(cat, 0) for cat in DISCRETIONARY_CATEGORIES)
        optimized = max(baseline - subscription_waste * 0.65 - discretionary_spending * 0.1, 0.0)
        potential_saving = baseline - optimized

        return {
            "baseline_next_month": round(baseline, 2),
            "optimized_next_month": round(optimized, 2),
            "potential_saving": round(potential_saving, 2),
        }

    def _generate_tips(
        self,
        month: str,
        month_over_month: dict[str, object],
        category_breakdown: list[dict[str, object]],
        subscription_waste: float,
        anomalies: list[dict[str, object]],
        forecast: dict[str, object],
        score: float,
    ) -> list[str]:
        tips: list[str] = []
        delta_pct = float(month_over_month.get("delta_pct", 0.0))
        if month_over_month.get("previous_month"):
            if delta_pct > 0.08:
                tips.append(
                    f"In {month}, spending rose {(delta_pct * 100):.1f}% versus {month_over_month['previous_month']}. "
                    "Start by capping discretionary categories this week."
                )
            elif delta_pct < -0.05:
                tips.append(
                    f"Great work. Spending fell {abs(delta_pct) * 100:.1f}% from {month_over_month['previous_month']}."
                )

        if subscription_waste > 0:
            tips.append(
                f"Estimated subscription waste is ${subscription_waste:.0f}/month. Cancel or downgrade one low-value plan first."
            )

        top_categories = category_breakdown[:2]
        if top_categories:
            category_text = ", ".join(f"{row['category']} ({row['share'] * 100:.1f}%)" for row in top_categories)
            tips.append(f"Your top spend concentration is {category_text}.")

        if anomalies:
            first = anomalies[0]
            tips.append(
                f"Review anomaly: {first['merchant']} on {first['date']} for ${first['amount']:.0f}. "
                "Confirm whether this was expected."
            )

        if score < 60:
            tips.append(
                "Your current financial behavior score is low. Focus on reducing entertainment, shopping, and unused subscriptions."
            )
        elif score >= 80:
            tips.append(
                "Your score is strong. Keep prioritizing education, health, and essential categories to sustain progress."
            )

        tips.append(
            f"If you follow the optimized plan, forecasted next-month savings are about ${forecast['potential_saving']:.0f}."
        )
        return tips[:6]

    def _estimate_probabilities(self, pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(x)
                return self._safe_probability_matrix(np.asarray(probs))

            decision = pipeline.decision_function(x)
            decision_array = np.asarray(decision)
            if decision_array.ndim == 1:
                positive = 1.0 / (1.0 + np.exp(-decision_array))
                probs = np.column_stack([1.0 - positive, positive])
                return self._safe_probability_matrix(probs)
            shifted = decision_array - np.max(decision_array, axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return self._safe_probability_matrix(probs)

    def _classifier_classes(self, pipeline: Pipeline) -> np.ndarray:
        classifier = pipeline.named_steps["classifier"]
        classes = getattr(classifier, "classes_", None)
        if classes is None and hasattr(pipeline, "classes_"):
            classes = getattr(pipeline, "classes_")
        if classes is None:
            return np.array(TARGET_CATEGORIES)
        return np.asarray(classes)

    def _safe_probability_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim == 1:
            matrix = np.column_stack([1.0 - matrix, matrix])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        cleaned = np.asarray(values, dtype=float)
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        minimum = float(np.min(cleaned))
        maximum = float(np.max(cleaned))
        if maximum - minimum < 1e-9:
            return np.zeros_like(cleaned, dtype=float)
        return (cleaned - minimum) / (maximum - minimum)

    def _json_clean(self, payload: dict[str, object]) -> dict[str, object]:
        cleaned: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, np.generic):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        return cleaned
