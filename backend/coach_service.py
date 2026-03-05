from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


GENERIC_TIPS = [
    {
        "tip": "Build a monthly budget and compare planned vs actual spend at month-end.",
        "source": "Consumer.gov budgeting guidance",
    },
    {
        "tip": "Keep an emergency fund target so surprise expenses do not become debt.",
        "source": "Fidelity emergency fund guidance",
    },
    {
        "tip": "Automate savings as a recurring transfer right after payday.",
        "source": "Consumer.gov savings guidance",
    },
    {
        "tip": "Use a weekly cap for discretionary categories like dining and entertainment.",
        "source": "CFPB spending control practices",
    },
    {
        "tip": "Review subscriptions monthly and cancel anything unused for two billing cycles.",
        "source": "Consumer finance subscription hygiene practices",
    },
    {
        "tip": "For debt payoff, prioritize highest-interest balances first.",
        "source": "Fidelity debt reduction guidance",
    },
    {
        "tip": "Use a 24-hour delay before non-essential purchases to reduce impulse spending.",
        "source": "Behavioral personal finance best practice",
    },
    {
        "tip": "Track your top 3 merchants monthly and set reduction targets on one of them.",
        "source": "Consumer finance spend-tracking practice",
    },
    {
        "tip": "Set transaction alerts for large charges to catch spikes quickly.",
        "source": "FDIC account monitoring guidance",
    },
    {
        "tip": "Reconcile your transactions weekly to keep forecast and reality aligned.",
        "source": "Budget operations best practice",
    },
    {
        "tip": "Protect savings goals by separating essentials and lifestyle categories.",
        "source": "CFPB budgeting framework",
    },
    {
        "tip": "Review recurring annual subscriptions before renewal and downgrade if needed.",
        "source": "Consumer software subscription management practice",
    },
]

GENERIC_FACTS = [
    {
        "fact": "Emergency funds are commonly structured to cover 3-6 months of essential expenses.",
        "source": "CFPB emergency savings guidance",
    },
    {
        "fact": "Automating savings transfers can improve consistency because the transfer happens before discretionary spend.",
        "source": "Consumer.gov savings guidance",
    },
    {
        "fact": "Separating fixed expenses from variable expenses makes monthly forecasts more accurate.",
        "source": "Federal Reserve financial education resources",
    },
    {
        "fact": "Paying credit cards in full each cycle avoids revolving interest charges.",
        "source": "CFPB credit card basics",
    },
    {
        "fact": "Tracking recurring subscriptions monthly can reduce unnoticed leakage over time.",
        "source": "Consumer subscription management best practice",
    },
    {
        "fact": "A spending plan is easier to maintain when reviewed weekly instead of only at month-end.",
        "source": "Behavioral budgeting practice",
    },
    {
        "fact": "Transaction alerts can help detect outlier charges earlier and reduce statement surprises.",
        "source": "FDIC account monitoring guidance",
    },
    {
        "fact": "Short-term spending caps are more effective when set by category and reviewed against actuals.",
        "source": "CFPB budgeting framework",
    },
]


@dataclass
class CoachReply:
    text: str
    powered_by: str
    source: str | None = None


class FinancialCoachService:
    def __init__(self) -> None:
        self.provider = os.getenv("FINANCE_COACH_PROVIDER", "auto").strip().lower()
        self.model = os.getenv("FINANCE_COACH_MODEL", "gpt-4o-mini").strip()
        self._llm = None
        self._prompt = None
        self._langchain_ready = False
        self._tip_cursor = 0
        self._fact_cursor = 0
        self._last_tip_text = ""
        self._bootstrap_langchain()

    def _bootstrap_langchain(self) -> None:
        if self.provider == "heuristic":
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except Exception:
            return

        self._llm = ChatOpenAI(model=self.model, temperature=0.25, api_key=api_key)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a personal finance coach. Give practical guidance only. "
                    "Use a friendly, human support tone. "
                    "Answer the user's exact question without canned templates. "
                    "Keep responses concise, clear, and actionable. "
                    "Do not use headers like 'Direct answer', 'Reason', or 'Next step'. "
                    "If the user indicates they are done (for example: 'that's it', 'for now', 'no worries'), "
                    "reply briefly, politely close, and do not add extra advice.",
                ),
                (
                    "human",
                    "User question: {question}\n\n"
                    "Recent chat context:\n{history}\n\n"
                    "Current analysis context:\n{context}\n\n"
                    "Respond naturally like a helpful finance tutor. "
                    "If useful, include one practical next action in plain language.",
                ),
            ]
        )
        self._langchain_ready = True

    def tip(self, analysis: dict[str, Any] | None = None) -> CoachReply:
        del analysis
        tip_text, source = self._general_tip()
        return CoachReply(text=tip_text, powered_by="heuristic", source=source)

    def fact(self) -> CoachReply:
        fact_text, source = self._general_fact()
        return CoachReply(text=fact_text, powered_by="heuristic", source=source)

    def coach(
        self,
        message: str,
        analysis: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> CoachReply:
        if self._is_close_intent(message):
            return CoachReply(
                text="Understood. We can stop here. If you want to continue later, I’m here.",
                powered_by="heuristic",
            )

        payload = analysis or {}
        turns = history or []
        if self._langchain_ready:
            reply = self._ask_langchain(question=message, analysis=payload, history=turns)
            if reply:
                return reply
        return CoachReply(
            text=self._heuristic_reply(message, payload, turns),
            powered_by="heuristic",
        )

    def _ask_langchain(
        self,
        question: str,
        analysis: dict[str, Any],
        history: list[dict[str, str]],
    ) -> CoachReply | None:
        if not self._langchain_ready or self._llm is None or self._prompt is None:
            return None
        try:
            messages = self._prompt.format_messages(
                question=question,
                history=self._history_text(history),
                context=self._analysis_context_text(analysis),
            )
            result = self._llm.invoke(messages)
            content = getattr(result, "content", None)
            text = str(content).strip() if content is not None else str(result).strip()
            if text:
                return CoachReply(text=text, powered_by="langchain")
        except Exception:
            return None
        return None

    def _history_text(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "none"
        lines = []
        for turn in history[-8:]:
            role = str(turn.get("role", "user"))
            message = str(turn.get("message", "")).strip()
            if message:
                lines.append(f"{role}: {message}")
        return "\n".join(lines) if lines else "none"

    def _analysis_context_text(self, analysis: dict[str, Any]) -> str:
        monthly_spending = float(analysis.get("monthly_spending", 0.0) or 0.0)
        subscription_waste = float(analysis.get("subscription_waste", 0.0) or 0.0)
        score = float((analysis.get("financial_score") or {}).get("score", 0.0) or 0.0)
        month = str(analysis.get("month", "unknown"))
        mom = analysis.get("month_over_month") or {}
        delta_pct = float(mom.get("delta_pct", 0.0) or 0.0) * 100
        top_categories = analysis.get("category_breakdown") or []
        top_label = "none"
        if top_categories:
            first = top_categories[0]
            top_label = (
                f"{first.get('category', 'unknown')} "
                f"({float(first.get('share', 0.0) or 0.0) * 100:.1f}% share)"
            )

        return (
            f"Month: {month}\n"
            f"Monthly spending: ${monthly_spending:,.0f}\n"
            f"MoM change: {delta_pct:+.1f}%\n"
            f"Subscription waste: ${subscription_waste:,.0f}\n"
            f"Financial score: {score:.1f}/100\n"
            f"Top category: {top_label}"
        )

    def _dynamic_tip(self, analysis: dict[str, Any]) -> tuple[str, str]:
        monthly_spending = float(analysis.get("monthly_spending", 0.0) or 0.0)
        subscription_waste = float(analysis.get("subscription_waste", 0.0) or 0.0)
        top_categories = analysis.get("category_breakdown") or []
        tip_pool: list[tuple[str, str]] = []

        if subscription_waste >= 40:
            tip_pool.append(
                (
                    f"Estimated subscription waste is ${subscription_waste:,.0f}/month. "
                    "Cancel or downgrade one low-value plan this week.",
                    "Behavioral analysis from your uploaded transactions",
                )
            )
            tip_pool.append(
                (
                    f"Subscription spend may have ${subscription_waste:,.0f}/month in waste. "
                    "Pause one non-essential renewal and review usage after 14 days.",
                    "Behavioral analysis from your uploaded transactions",
                )
            )

        if top_categories:
            top = top_categories[0]
            category = str(top.get("category", "discretionary spending"))
            share = float(top.get("share", 0.0) or 0.0) * 100
            tip_pool.append(
                (
                    f"Your top category is {category} at {share:.1f}% of spend. "
                    "Try a weekly cap in this category for the next 4 weeks.",
                    "Behavioral analysis from your uploaded transactions",
                )
            )

        for offset in range(min(8, len(GENERIC_TIPS))):
            entry = GENERIC_TIPS[(self._tip_cursor + offset) % len(GENERIC_TIPS)]
            text = str(entry["tip"])
            if monthly_spending > 0:
                text = f"{text} Current month spend is ${monthly_spending:,.0f}."
            tip_pool.append((text, str(entry["source"])))

        if not tip_pool:
            tip_pool.append(
                (
                    "Set one realistic spending target for this week and check it daily.",
                    "General personal finance best practice",
                )
            )

        pick = self._tip_cursor % len(tip_pool)
        tip_text, source = tip_pool[pick]
        if tip_text == self._last_tip_text and len(tip_pool) > 1:
            pick = (pick + 1) % len(tip_pool)
            tip_text, source = tip_pool[pick]

        self._tip_cursor += 1
        self._last_tip_text = tip_text
        return tip_text, source

    def _general_tip(self) -> tuple[str, str]:
        if not GENERIC_TIPS:
            return (
                "Set one realistic spending target for this week and review it daily.",
                "General personal finance best practice",
            )
        index = self._tip_cursor % len(GENERIC_TIPS)
        self._tip_cursor += 1
        entry = GENERIC_TIPS[index]
        return str(entry["tip"]), str(entry["source"])

    def _general_fact(self) -> tuple[str, str]:
        if not GENERIC_FACTS:
            return (
                "Consistent tracking improves spending decisions over time.",
                "General financial literacy best practice",
            )
        index = self._fact_cursor % len(GENERIC_FACTS)
        self._fact_cursor += 1
        entry = GENERIC_FACTS[index]
        return str(entry["fact"]), str(entry["source"])

    def _heuristic_reply(
        self,
        question: str,
        analysis: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        query = question.lower().strip()
        query_words = set(re.findall(r"[a-z0-9]+", query))
        monthly_spending = float(analysis.get("monthly_spending", 0.0) or 0.0)
        subscription_waste = float(analysis.get("subscription_waste", 0.0) or 0.0)
        score = float((analysis.get("financial_score") or {}).get("score", 0.0) or 0.0) / 10
        month_over_month = analysis.get("month_over_month") or {}
        delta_pct = float(month_over_month.get("delta_pct", 0.0) or 0.0) * 100
        forecast = analysis.get("forecast") or {}
        baseline = float(forecast.get("baseline_next_month", 0.0) or 0.0)
        optimized = float(forecast.get("optimized_next_month", 0.0) or 0.0)
        potential_saving = float(forecast.get("potential_saving", 0.0) or 0.0)
        top_categories = analysis.get("category_breakdown") or []
        anomalies = analysis.get("anomalies") or []
        assistant_turns = sum(1 for turn in history if str(turn.get("role", "")).lower() == "assistant")

        def has_any(*terms: str) -> bool:
            return any(term in query_words for term in terms)

        if self._is_close_intent(query):
            return "Got it. Happy to help. Reach out anytime you want to review your spending again."

        if has_any("thanks", "thank", "thx", "appreciate"):
            return "You’re welcome. If you want, I can help you pick one category to reduce next month for the biggest impact."

        if has_any("hello", "hi", "hey"):
            return "Hi. I can help with saving strategies, subscription cleanup, and next-month planning based on your transactions."

        if "anomaly" in query or "fraud" in query or "unusual" in query:
            if anomalies:
                top = anomalies[0]
                return (
                    f"The most unusual transaction is {top.get('merchant', 'unknown merchant')} on "
                    f"{top.get('date', 'unknown date')} with about "
                    f"{float(top.get('anomaly_confidence', 0.0) or 0.0):.1f}% confidence. "
                    "If that charge looks unfamiliar, confirm it and add an alert for similar amounts."
                )
            return "I’m not seeing high-confidence anomalies this month. Your recent transactions look within normal range."

        if "tip" in query or "advice" in query:
            tip_text, _ = self._dynamic_tip(analysis)
            return f"{tip_text} Try it for one week, then we can check the impact together."

        if "other" in query or "another" in query or "more" in query or "different" in query or "else" in query:
            alt = self._alternate_coach_idea(analysis, assistant_turns)
            return alt

        if "subscription" in query or "waste" in query:
            return (
                f"You’re likely losing around ${subscription_waste:,.0f}/month to low-value subscriptions. "
                "Start by canceling or downgrading one non-essential plan this week."
            )

        if "score" in query or "grade" in query:
            return (
                f"Your financial score is {score:.1f}/10 right now. "
                "The fastest way to improve it is to trim one low-priority discretionary category next month."
            )

        if "forecast" in query or "next month" in query or "plan" in query:
            return (
                f"Next month, your baseline is about ${baseline:,.0f}, and your optimized plan is about ${optimized:,.0f}. "
                f"That’s a potential improvement of roughly ${potential_saving:,.0f}. "
                "A weekly cap on your top discretionary category is the best first step."
            )

        if "month" in query or "trend" in query or "compare" in query:
            return (
                f"Month over month, spending changed by {delta_pct:+.1f}% and is currently around ${monthly_spending:,.0f}. "
                "If you want, I can break down which category drove that move most."
            )

        if "education" in query or "entertainment" in query or "priority" in query:
            return (
                "In this scoring system, education and health are treated as higher-value spend, while entertainment is lower-priority. "
                "So it’s better to trim entertainment first before cutting development-focused spend."
            )

        if has_any("save", "saving", "reduce", "cut", "budget", "control", "optimize"):
            if top_categories:
                top = top_categories[0]
                top_name = str(top.get("category", "top category"))
                top_share = float(top.get("share", 0.0) or 0.0) * 100
                return (
                    f"The fastest way to save is to reduce {top_name}, which is about {top_share:.1f}% of your spending. "
                    f"A practical target is to trim around ${max(monthly_spending * 0.05, 25):,.0f} this month."
                )
            return (
                "A weekly budget works better than a monthly-only budget for most people. "
                "Pick one flexible category and reduce it by about 10%."
            )

        if has_any("sorry", "what", "explain", "help"):
            return (
                "I can help with category cuts, subscription cleanup, forecast planning, or unusual transactions. "
                "If you want, ask me something specific like 'which category should I cut first?'"
            )

        mentioned_merchant = self._find_merchant_in_query(query, analysis)
        if mentioned_merchant:
            merchant_name, merchant_spend = mentioned_merchant
            return (
                f"You spent about ${merchant_spend:,.0f} at {merchant_name}. "
                "If this is discretionary spend, a merchant-specific cap for the next two billing cycles can help."
            )

        if top_categories:
            return self._generic_context_reply(
                query=query,
                top_categories=top_categories,
                monthly_spending=monthly_spending,
                subscription_waste=subscription_waste,
            )

        if history:
            return (
                "You’re asking good questions. I’d keep it simple: make one change this week, track it, and then adjust."
            )

        return (
            "A good starting point is reducing one discretionary category by 10% this month, then reviewing progress weekly."
        )

    def _is_close_intent(self, text: str) -> bool:
        query = str(text or "").lower().strip()
        normalized = re.sub(r"[^a-z0-9\s]", " ", query)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        close_phrases = (
            "that is it",
            "thats it",
            "that s it",
            "for now",
            "no worries",
            "no worry",
            "all good",
            "all set",
            "i am done",
            "im done",
            "done for now",
            "nothing else",
            "thats all",
            "that s all",
            "thats enough",
            "that s enough",
            "no thanks",
            "no thank you",
            "no need",
            "no need done",
            "end it",
            "stop here",
            "stop now",
            "goodbye",
            "bye",
        )
        if any(phrase in normalized for phrase in close_phrases):
            return True

        if re.search(r"\b(no|nah|nope)\s+thanks?\b", normalized):
            return True

        return False

    def _find_merchant_in_query(
        self,
        query: str,
        analysis: dict[str, Any],
    ) -> tuple[str, float] | None:
        categorized = analysis.get("categorized_transactions") or []
        merchant_totals: dict[str, float] = {}
        for row in categorized:
            merchant = str(row.get("merchant", "")).strip()
            if not merchant:
                continue
            spend = float(row.get("spend", 0.0) or 0.0)
            merchant_totals[merchant] = merchant_totals.get(merchant, 0.0) + spend

        for merchant, total in merchant_totals.items():
            if merchant.lower() in query:
                return merchant, total
        return None

    def _alternate_coach_idea(self, analysis: dict[str, Any], assistant_turns: int) -> str:
        monthly_spending = float(analysis.get("monthly_spending", 0.0) or 0.0)
        subscription_waste = float(analysis.get("subscription_waste", 0.0) or 0.0)
        top_categories = analysis.get("category_breakdown") or []
        top_name = str(top_categories[0].get("category", "top category")) if top_categories else "top category"
        top_share = float(top_categories[0].get("share", 0.0) or 0.0) * 100 if top_categories else 0.0

        ideas = [
            (
                f"Try a 7-day spending sprint: cut {top_name} by 15% this week and log daily totals. "
                "This gives a fast signal on what is actually controllable."
            ),
            (
                f"Use merchant caps: pick your top two discretionary merchants and set a hard monthly limit. "
                f"Your current top category is {top_name} ({top_share:.1f}%)."
            ),
            (
                f"Shift recurring expenses first: reduce subscription waste by ${max(subscription_waste * 0.25, 15):,.0f} "
                "before cutting variable categories."
            ),
            (
                f"Use a weekly review rhythm: compare actual vs plan every Sunday on ${monthly_spending:,.0f} monthly spend. "
                "Only change one category per week to keep adherence high."
            ),
            (
                "Set a rule-based guardrail: any non-essential spend above a set amount requires a 24-hour delay. "
                "This reduces impulse leakage without strict deprivation."
            ),
        ]
        return ideas[assistant_turns % len(ideas)]

    def _generic_context_reply(
        self,
        query: str,
        top_categories: list[dict[str, Any]],
        monthly_spending: float,
        subscription_waste: float,
    ) -> str:
        first = top_categories[0]
        second = top_categories[1] if len(top_categories) > 1 else None
        first_name = str(first.get("category", "top category"))
        first_share = float(first.get("share", 0.0) or 0.0) * 100
        selector = sum(ord(ch) for ch in query) % 3
        if selector == 0 and second:
            second_name = str(second.get("category", "next category"))
            return (
                f"Your biggest categories are {first_name} ({first_share:.1f}%) and {second_name}. "
                f"If you reduce {first_name} first, you’ll usually see the fastest result."
            )
        if selector == 1:
            return (
                f"Monthly spending is about ${monthly_spending:,.0f}. "
                f"Start with {first_name}, and cap it to trim at least ${max(monthly_spending * 0.04, 20):,.0f}."
            )
        return (
            f"Subscription leakage may be around ${subscription_waste:,.0f}/month. "
            "Pausing one non-essential subscription for two weeks is a good first experiment."
        )
