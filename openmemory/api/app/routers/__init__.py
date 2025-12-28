from .apps import router as apps_router
from .backup import router as backup_router
from .config import router as config_router
from .entities import router as entities_router
from .experiments import router as experiments_router
from .feedback import router as feedback_router
from .gdpr import router as gdpr_router
from .graph import router as graph_router
from .health import router as health_router
from .memories import router as memories_router
from .search import router as search_router
from .stats import router as stats_router

__all__ = [
    "memories_router",
    "apps_router",
    "stats_router",
    "config_router",
    "backup_router",
    "entities_router",
    "graph_router",
    "health_router",
    "feedback_router",
    "experiments_router",
    "search_router",
    "gdpr_router",
]
