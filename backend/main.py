from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .coach_service import FinancialCoachService
from .ml_engine import PersonalFinanceMLEngine
from .schemas import AnalyzeRequest, CoachRequest, TipRequest
from .statement_ingest import StatementIngestor


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Personal Finance Insights API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PersonalFinanceMLEngine(BASE_DIR)
statement_ingestor = StatementIngestor()
coach_service = FinancialCoachService()


@app.on_event("startup")
def startup() -> None:
    engine.bootstrap()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": engine.category_pipeline is not None,
        "metadata": engine.model_metadata,
    }


@app.get("/api/sample-transactions")
def sample_transactions() -> dict[str, object]:
    return {"transactions": engine.load_sample_transactions()}


@app.post("/api/ingest")
async def ingest_statement(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename for upload.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        transactions, meta = statement_ingestor.ingest(file.filename, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"transactions": transactions, "meta": meta}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, object]:
    try:
        result = engine.analyze(
            transactions=[transaction.model_dump() for transaction in payload.transactions],
            month=payload.month,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/api/retrain")
def retrain() -> dict[str, object]:
    metadata = engine.train_and_save_category_model()
    return {"status": "retrained", "metadata": metadata}


@app.post("/api/tip")
def tip(payload: TipRequest) -> dict[str, object]:
    reply = coach_service.tip(payload.analysis)
    return {"tip": reply.text, "source": reply.source, "powered_by": reply.powered_by}


@app.get("/api/fact")
def fact() -> dict[str, object]:
    reply = coach_service.fact()
    return {"fact": reply.text, "source": reply.source, "powered_by": reply.powered_by}


@app.post("/api/coach")
def coach(payload: CoachRequest) -> dict[str, object]:
    history = []
    if payload.history:
        history = [{"role": item.role, "message": item.message} for item in payload.history]
    reply = coach_service.coach(payload.message, payload.analysis, history=history)
    return {"response": reply.text, "powered_by": reply.powered_by}


app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
