import torch
from fastapi import FastAPI
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from config_loader import env_int, load_env_file, require_env
from detector_arch import DesklibAIDetectionModel
from schemas import DetectRequest, DetectResponse


load_env_file("detector")

MODEL_ID = require_env("DETECTOR_MODEL_NAME")
HOST = require_env("DETECTOR_HOST")
PORT = env_int("DETECTOR_PORT", 8002)
BATCH_SIZE = env_int("DETECTOR_BATCH_SIZE", 32)
MAX_LENGTH = env_int("DETECTOR_MAX_LENGTH", 512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Oculus AI Detector API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = DesklibAIDetectionModel.from_pretrained(MODEL_ID).to(device)
model.eval()


def batch_detect_logits(texts: list[str]) -> list[float]:
    """Return raw detector logits for a list of texts."""
    logits_out: list[float] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="detect"):
        chunk = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs)["logits"].squeeze(-1)

        logits_out.extend(logits.detach().cpu().numpy().tolist())

    return logits_out


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": str(device)}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    """Score texts with the multilingual AI detector."""
    logits = batch_detect_logits(request.texts)
    return DetectResponse(logits=logits)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
