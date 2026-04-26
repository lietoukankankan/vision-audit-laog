"""
词根视频视觉审核中间件
调用老G Gemini API，部署到Render
"""

import os
import json
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# 老G的Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}"

# 老G的审核Prompt
AUDIT_PROMPT = """你是词根视频的视觉审核专家（老G角色），我有大量历史沟通记录，了解课程节奏。

请对这张分镜图进行像素级审核：

核心视觉规范：
- 极致留白（#FFFFFF背景）
- 无边框、无线条
- EB Garamond（英文衬线）+ Noto Sans CJK Bold（中文无衬线）
- 字体的古典与现代碰撞感

审核要点：
1. 背景是否纯白#FFFFFF？有无杂色或边框？
2. 字体是否正确？EB Garamond英文 + Noto Sans CJK Bold中文？
3. 中英文字体大小是否匹配？是否有势均力敌的力量感？
4. 排版是否有重叠、错位、模糊？
5. 整体是否达到"极简降维打击"的质感？

请用JSON格式返回审核结果：
{
    "passed": true/false,
    "score": 0-100,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "detail": "详细分析"
}"""

app = FastAPI(title="词根视频视觉审核 - 老G版")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuditRequest(BaseModel):
    image_base64: str
    context: str = ""  # 可选的课程上下文

class AuditResponse(BaseModel):
    passed: bool
    score: int
    issues: list[str]
    suggestions: list[str]
    detail: str

@app.get("/health")
async def health():
    return {"status": "ok", "service": "vision-audit-laog"}

@app.post("/audit", response_model=AuditResponse)
async def audit_image(request: AuditRequest):
    """调用老G审核图片"""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")
    
    # 清理base64
    image_data = request.image_base64
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    # 构建Gemini请求
    prompt = AUDIT_PROMPT
    if request.context:
        prompt += f"\n\n课程上下文：{request.context}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2000
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code != 200:
            raise HTTPException(502, f"Gemini API error: {response.text}")
        
        result = response.json()
        
        try:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            return AuditResponse(**parsed)
        except Exception as e:
            # 解析失败，返回原始内容
            return AuditResponse(
                passed=False,
                score=0,
                issues=["解析审核结果失败"],
                suggestions=[],
                detail=content if 'content' in dir() else str(result)
            )

class AskRequest(BaseModel):
    message: str

class AskResponse(BaseModel):
    reply: str

@app.post("/ask", response_model=AskResponse)
async def ask_laog(request: AskRequest):
    """和老G文字对话"""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")
    
    payload = {
        "contents": [{
            "parts": [{"text": request.message}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2000
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code != 200:
            raise HTTPException(502, f"Gemini API error: {response.text}")
        
        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        
        return AskResponse(reply=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
