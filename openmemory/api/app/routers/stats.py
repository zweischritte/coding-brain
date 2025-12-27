from app.database import get_db
from app.models import App, Memory, MemoryState, User
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/stats", tags=["stats"])

@router.get("/")
async def get_profile(
    principal: Principal = Depends(require_scopes(Scope.STATS_READ)),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get total number of memories
    total_memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted).count()

    # Get total number of apps
    apps = db.query(App).filter(App.owner == user)
    total_apps = apps.count()

    return {
        "total_memories": total_memories,
        "total_apps": total_apps,
        "apps": apps.all()
    }

