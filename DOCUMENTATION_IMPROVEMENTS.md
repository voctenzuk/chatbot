# README.md Documentation Improvements

## Suggested New README.md Structure

Below are the proposed additions and modifications to README.md, presented in a diff-like format.

---

## 1. QUICKSTART SECTION (Expanded)

### Current:
```bash
# 1) Create .env from .env.example:
cp .env.example .env
# Fill in: TELEGRAM_BOT_TOKEN, LLM_API_KEY (OpenRouter)

# 2) Start infrastructure:
docker compose up -d  # Redis

# 3) Install dependencies:
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,images]"

# 4) Run bot:
python -m bot
```

### Proposed:
```bash
# 1) Clone and enter directory
git clone <repo>
cd chatbot

# 2) Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3) Install dependencies
uv pip install -e ".[dev,images]"

# 4) Configure environment
cp .env.example .env
# Edit .env with required variables (see Environment Variables section)

# 5) Start infrastructure (Redis)
docker compose up -d

# 6) Setup Supabase (optional but recommended)
# See "Database Setup" section below

# 7) Run the bot
python -m bot
# or: make run
```

---

## 2. NEW: ENVIRONMENT VARIABLES SECTION

```markdown
## Environment Variables

Copy `.env.example` to `.env` and fill in the following variables:

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Get from @BotFather | `123456:ABC-DEF...` |
| `LLM_API_KEY` | OpenRouter API key | `sk-or-v1-...` |

### Optional but Recommended
| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | (empty) |
| `SUPABASE_KEY` | Supabase anon key | (empty) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | (empty) |
| `MEM0_API_KEY` | mem0.ai API key for memory | (empty) |
| `MEM0_PROJECT_ID` | mem0 project ID | (empty) |
| `STRIPE_SECRET_KEY` | Stripe secret key | (empty) |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook secret | (empty) |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key | (empty) |
| `PADDLE_API_KEY` | Paddle API key | (empty) |
| `PADDLE_WEBHOOK_SECRET` | Paddle webhook secret | (empty) |

### Feature-Specific
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | Text model via OpenRouter | `x-ai/grok-4.1-fast` |
| `VISION_MODEL` | Vision model for image analysis | `x-ai/grok-2-vision-1212` |
| `IMAGE_MODEL` | Image generation model | `x-ai/grok-imagine-image` |
| `IMAGE_PROVIDER` | Image provider (`openrouter` or `openai`) | `openrouter` |
| `OPENAI_API_KEY` | Fallback for OpenAI image gen | (empty) |
| `FREE_MESSAGES_LIMIT` | Daily message limit (free tier) | `10` |
| `FREE_IMAGES_LIMIT` | Daily image limit (free tier) | `3` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
```

---

## 3. NEW: DATABASE SETUP SECTION

```markdown
## Database Setup (Supabase)

The bot uses Supabase for user management, subscriptions, and usage tracking.

### Option 1: New Supabase Project

1. Create a project at [supabase.com](https://supabase.com)
2. Go to SQL Editor → New Query
3. Run the migration from `supabase_migrations/migrations/001_initial_schema.sql`
4. Get your credentials from Project Settings → API:
   - `SUPABASE_URL`: Project URL
   - `SUPABASE_KEY`: `anon public` key
   - `SUPABASE_SERVICE_ROLE_KEY`: `service_role` key (keep secret!)

### Option 2: Local Development (Optional)

For local development without Supabase:
- The bot will use in-memory storage with limited functionality
- Subscriptions and usage tracking won't persist
- Run only Redis: `docker compose up -d redis`

### Database Schema Overview

- **users**: Telegram user mapping
- **subscription_plans**: Available plans (Free, Pro)
- **user_subscriptions**: Active subscriptions
- **usage_tracking**: Daily usage counters
- **payments**: Payment history
- **webhook_events**: Audit log for payment webhooks
```

---

## 4. NEW: PAYMENT WEBHOOKS SECTION

```markdown
## Payment Webhooks (Stripe/Paddle)

To handle subscription payments, configure webhooks:

### Stripe Setup

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://your-domain.com/webhooks/stripe`
3. Select events:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy the Webhook signing secret to `STRIPE_WEBHOOK_SECRET`

### Local Webhook Testing (Stripe)

Use Stripe CLI for local development:

```bash
# Install Stripe CLI first
stripe login
stripe listen --forward-to localhost:8000/webhooks/stripe
# This gives you a webhook secret for .env
```

### Paddle Setup (Optional)

1. Go to Paddle Dashboard → Developer Tools → Notifications
2. Add endpoint: `https://your-domain.com/webhooks/paddle`
3. Select events:
   - `subscription.created`
   - `subscription.updated`
   - `subscription.canceled`
   - `transaction.completed`
4. Copy webhook secret to `PADDLE_WEBHOOK_SECRET`

### Webhook Health Check

```bash
curl http://localhost:8000/webhooks/health
```
```

---

## 5. NEW: MEMORY (MEM0) SETUP SECTION

```markdown
## Memory Setup (mem0)

The bot supports long-term memory via [mem0.ai](https://mem0.ai):

1. Sign up at mem0.ai and create a project
2. Get API key from Dashboard
3. Set in `.env`:
   ```
   MEM0_API_KEY=your_key_here
   MEM0_PROJECT_ID=your_project_id
   ```

Without mem0 configured, the bot still works with short-term (Redis) memory only.

### Memory Architecture

- **Short-term**: Redis (conversation context, rate limits)
- **Long-term**: mem0 (user facts, preferences, image descriptions)
- **Emotional**: Tracked via EnhancedMem0MemoryService
```

---

## 6. NEW: RUNNING TESTS SECTION

```markdown
## Running Tests

### Prerequisites
Ensure you have installed dev dependencies:
```bash
uv pip install -e ".[dev]"
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov
# HTML report: htmlcov/index.html
```

### Run Specific Test File
```bash
pytest tests/test_llm_service.py -v
pytest tests/test_subscription_service.py -v
pytest tests/test_handlers.py -v
```

### Code Quality
```bash
make fmt    # Format with ruff
make lint   # Lint with ruff
pyright     # Type checking
```

### Test Environment Variables
For testing, you can use a separate `.env.test`:
```bash
TESTING=true
REDIS_URL=redis://localhost:6379/1  # Use different DB
```
```

---

## 7. NEW: DEPLOYMENT NOTES SECTION

```markdown
## Deployment Notes

### Production Checklist

- [ ] Set `TELEGRAM_BOT_TOKEN` (production bot)
- [ ] Set `LLM_API_KEY` with production API key
- [ ] Configure `SUPABASE_*` keys for production database
- [ ] Set `STRIPE_*` keys for production payments
- [ ] Configure Redis (Upstash, Redis Cloud, or self-hosted)
- [ ] Set up webhook endpoints with HTTPS
- [ ] Configure monitoring (Sentry, Logtail, etc.)

### Recommended Hosting

#### Railway / Render / Fly.io
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e ".[images]"

CMD ["python", "-m", "bot"]
```

#### Docker Compose (Self-hosted)
```yaml
version: '3.8'
services:
  bot:
    build: .
    env_file: .env
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Environment-Specific Config

| Environment | Supabase Project | Stripe Mode | Notes |
|-------------|-----------------|-------------|-------|
| Development | Local or staging | Test mode | Use test API keys |
| Staging | Staging project | Test mode | Mirror production |
| Production | Production project | Live mode | Real payments |

### Webhook URL Format

```
https://<your-domain>/webhooks/stripe
https://<your-domain>/webhooks/paddle
```

### Health Check Endpoint

The bot exposes a health check at:
```
GET /webhooks/health
```

Response:
```json
{
  "status": "healthy",
  "stripe_available": true,
  "paddle_available": false,
  "stripe_configured": true,
  "paddle_configured": false
}
```
```

---

## 8. ARCHITECTURE DOCUMENTATION (New File: ARCHITECTURE.md)

```markdown
# Chatbot Architecture

## Overview

Telegram bot built with:
- **aiogram 3.x**: Async Telegram Bot API framework
- **LangChain/LangGraph**: LLM orchestration (currently minimal, expandable)
- **mem0**: Long-term memory service
- **Supabase**: PostgreSQL database + Auth
- **Redis**: Short-term cache and state
- **OpenRouter**: Unified LLM API (Grok models)

## Service Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Telegram User                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  aiogram Dispatcher (src/bot/app.py)                        │
│  └── Router (src/bot/handlers.py)                           │
│      ├── Command handlers (/start, /status, /upgrade)       │
│      ├── Photo handler (vision)                             │
│      └── Message handler (text + image generation)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  LLMService  │ │ImageService│ │VisionService │
│  (Grok 4.1)  │ │(Generation)│ │ (Analysis)   │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│MemoryService │ │ Supabase │ │Subscription  │
│  (mem0.ai)   │ │ Manager  │ │   Service    │
└──────────────┘ └──────────┘ └──────────────┘
```

## Key Services

### LLMService (src/bot/services/llm_service.py)
- OpenRouter API client
- Supports streaming responses
- Configurable model (default: Grok 4.1 Fast)

### ImageService (src/bot/services/image_service.py)
- Image generation via OpenRouter or OpenAI
- Prompt refinement

### VisionService (src/bot/services/vision_service.py)
- Image analysis using vision models
- Generates descriptions and tags
- Stores descriptions in memory

### MemoryService (src/bot/services/mem0_memory_service.py)
- Long-term memory via mem0.ai
- User facts, preferences, conversation history
- Image descriptions with metadata

### EnhancedMem0MemoryService (src/bot/services/enhanced_mem0_memory_service.py)
- Emotional memory tracking
- User profile building
- Proactive message triggers
- Importance scoring and memory decay

### SupabaseManager (src/bot/services/supabase_manager.py)
- User management
- Subscription tracking
- Usage tracking with rate limiting
- Connection pooling via asyncpg

### SubscriptionService (src/bot/services/subscription_service.py)
- In-memory quota tracking (fallback)
- Free/Premium tier management

## Webhook Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Stripe/    │────▶│  FastAPI Router  │────▶│  Supabase    │
│   Paddle     │     │  (webhooks.py)   │     │  (persist)   │
└──────────────┘     └──────────────────┘     └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Bot Logic   │
                       │ (notify user)│
                       └──────────────┘
```

## Data Flow

### Text Message Flow
1. User sends message
2. Handler checks rate limit (Supabase)
3. Memory context retrieved (mem0)
4. LLM generates response (OpenRouter)
5. Response stored in memory
6. Usage incremented (Supabase)

### Photo Analysis Flow
1. User sends photo
2. Handler checks rate limit
3. VisionService analyzes image
4. Description stored in memory
5. Response sent to user

### Image Generation Flow
1. User sends "draw/image" request
2. Handler checks rate limit
3. ImageService generates image
4. URL/description stored in memory
5. Response sent with image info

## Security Considerations

- All user data isolated by `user_id`
- Supabase RLS policies enforce access control
- Webhook signatures verified (Stripe/Paddle)
- Service role keys never exposed to clients
- Rate limiting at database level
```

---

## 9. BEST PRACTICES NOTES (Based on Code Review)

### aiogram 3.x Best Practices (Verified)

Current implementation follows good practices:
- ✅ Uses `Dispatcher` with routers
- ✅ Proper async handlers
- ✅ Type hints on handlers
- ✅ `F.photo` filter for images
- ✅ Command filters (`CommandStart`, `Command`)

Suggested improvements:
- Consider using `middleware` for rate limiting instead of per-handler checks
- Use `aiogram.fsm` (Finite State Machine) for multi-step conversations
- Add error handler with `@router.errors()`

Example middleware approach:
```python
# Suggested for future refactoring
class RateLimitMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        # Rate limit check here
        return await handler(event, data)
```

### LangChain/LangGraph Integration (Observations)

Current state:
- Dependencies installed (`langchain-core`, `langgraph`)
- Minimal actual integration in current code
- LLMService uses raw httpx instead of LangChain

Suggested approach for future:
```python
# If fully integrating LangGraph:
from langgraph.graph import StateGraph

class BotState(TypedDict):
    messages: list
    memory_context: str
    user_id: int

graph = StateGraph(BotState)
graph.add_node("retrieve_memory", retrieve_memory_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve_memory")
```

Current architecture is simpler and works well for the use case.
```

---

## Summary of Recommended Changes

### Files to Create:
1. `ARCHITECTURE.md` - Detailed architecture documentation
2. `DEPLOYMENT.md` - Deployment guide (optional, can be in README)

### Files to Modify:
1. `README.md` - Add sections: Environment Variables, Database Setup, Payment Webhooks, Memory Setup, Running Tests, Deployment Notes
2. `.env.example` - Add comments explaining each variable

### Minor Code Improvements (Optional):
1. Add docstring to `src/bot/app.py` explaining the app structure
2. Add error handler in handlers.py
3. Consider FSM for multi-step flows
