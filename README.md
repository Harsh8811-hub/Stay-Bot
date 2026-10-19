# Stay Bot

Stay Bot is a hotel booking chatbot. Guests talk to it on WhatsApp or in a local chat screen. It answers questions, finds rooms, quotes a price, takes a booking, and follows the stay through payment, check-in, and checkout.

The bot is a FastAPI service. Conversation flow is a LangGraph workflow backed by Gemini. Hotel and guest data live in MongoDB. Hotel knowledge for answers is stored in ChromaDB.

## What it handles

- Guest chat, room search, quotes, and booking confirmation
- WhatsApp messages, including buttons and media
- Payments, receipts, and payment-proof checks
- Staff takeover, tasks, and escalations
- Hotel amenities, policies, and knowledge-base answers
- Per-hotel settings so one deployment can serve more than one property

## Run it

Python 3.11 is the version the container uses. From this folder:

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
# Create a .env in this folder with the values listed below, then:
.\start.ps1
```

`start.ps1` starts the API on port 8001 and the Streamlit chat UI.

Docker:

```powershell
docker compose up --build
```

The API listens on port 8000 inside Docker. Health check: `GET /health`.

## Configuration

Settings come from a `.env` file, which is not part of this repo. Required values:

- `MONGODB_URI` — MongoDB connection string
- `GEMINI_API_KEY` — Gemini API key

Common optional values: `MONGODB_DATABASE`, `HOTEL_ID`, `HOTEL_NAME`, `GEMINI_MODEL`, `APP_PORT`.

## Layout

- `app/` — API, workflow, services, and database code
- `chat_ui.py` — local Streamlit chat screen
- `requirements.txt` — Python dependencies
- `Dockerfile` and `docker-compose.yml` — container run
