import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_loader import env_int, load_env_file, require_env
from schemas import GenerateRequest, GenerateResponse


load_env_file("generator")

MODEL_NAME = require_env("GENERATOR_MODEL_NAME")
HOST = require_env("GENERATOR_HOST")
PORT = env_int("GENERATOR_PORT", 8001)
BATCH_SIZE = env_int("GENERATOR_BATCH_SIZE", 16)
MAX_NEW_TOKENS_DEFAULT = env_int("GENERATOR_MAX_NEW_TOKENS", 300)

app = FastAPI(title="Qwen Essay Generator API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def batch_generate(system_prompt: str, user_prompts: list[str], max_new_tokens: int) -> list[str]:
    """Generate one essay per user prompt with a shared system prompt."""
    all_messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in user_prompts
    ]

    texts = tokenizer.apply_chat_template(
        all_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    model_inputs["attention_mask"] = model_inputs["attention_mask"].bool().int()

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(trimmed, skip_special_tokens=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate Spanish essays for a batch of literary topics."""
    user_prompts = [f"Escribe un ensayo sobre: {topic}" for topic in request.topics]
    max_new_tokens = request.max_new_tokens or MAX_NEW_TOKENS_DEFAULT

    essays: list[str] = []
    for start in range(0, len(user_prompts), BATCH_SIZE):
        chunk = user_prompts[start : start + BATCH_SIZE]
        essays.extend(batch_generate(request.system_prompt, chunk, max_new_tokens))

    return GenerateResponse(essays=essays, topics=request.topics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
