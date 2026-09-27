from fastapi import FastAPI
from src.api.routes import query, ingest, evaluate

def create_app() -> FastAPI:
    app = FastAPI(
        title="Production RAG API",
        description="A robust RAG pipeline demonstrating Dave Ebbelaar's architectural patterns",
        version="1.0.0"
    )
    
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(evaluate.router, prefix="/api/v1")
    
    @app.get("/health")
    def health_check():
        return {"status": "ok"}
        
    return app

app = create_app()
