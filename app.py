import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from bot import Bot
    
bot = Bot("phi3")

app = FastAPI()

@app.get("/")
async def index_page():
    return FileResponse("index.html", 200, media_type="text/html")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    return StreamingResponse(bot.chat(data.get('messages', []), stream=True), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app)