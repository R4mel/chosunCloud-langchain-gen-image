import uuid
from fastapi import APIRouter
from datetime import datetime

from service.ai_service import AIService
from schema.image_schema import PromptRequest, PromptResponse

router = APIRouter(prefix="/api")


@router.post("/image", response_model=PromptResponse)
async def gen_image(req: PromptRequest):
    print(f"Human Prompt: {req}")
    unique_id = str(uuid.uuid4())
    print(f"Generated Unique ID: {unique_id}")

    curret_time = datetime.now()  # 이미지 생성날짜
    ai_service = AIService()

    graph = ai_service.gen_graph(req.prompt)
    state = graph.invoke(
        {
            "id": unique_id,
            "prompt": req.prompt,
            "image_url": "",
        }
    )
    return {"url": state["image_url"]}
