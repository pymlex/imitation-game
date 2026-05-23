from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    system_prompt: str
    topics: list[str] = Field(min_length=1)
    max_new_tokens: int = 300


class GenerateResponse(BaseModel):
    essays: list[str]
    topics: list[str]


class DetectRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class DetectResponse(BaseModel):
    logits: list[float]
