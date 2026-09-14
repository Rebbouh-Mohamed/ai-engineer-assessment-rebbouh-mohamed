# AI Engineer Assessment — Superhero api & Movies dataset

A production-grade FastAPI service providing a single endpoint `POST /ask` that accepts natural language questions, intelligently routes queries across a local movie catalog (~9,000 films) and the Superhero API (or synthesizes from both), and provides strict source attribution on every response.



## Setup & Running

### 1. Prerequisites
- Python 3.10+
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/))
- Superhero API Token ([Superhero API](https://superheroapi.com/) via GitHub sign-in)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Rebbouh-Mohamed/ai-engineer-assessment-rebbouh-mohamed.git
cd ai-engineer-assessment-rebbouh-mohamed

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```ini
GEMINI_API_KEY=your_gemini_api_key
SUPER_HERO_API_KEY=your_superhero_api_token
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite
MOVIES_CSV_PATH=data/movies.csv
```

### 4. Run the Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger docs are available at `http://localhost:8000/docs`.

---

## Example Usage

### 1. Mixed Query (Both Sources)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who directed The Dark Knight, and what are Batman'\''s power stats?"}'
```
**Response:**
```json
{
  "answer": "From the movie dataset, information about the director of 'The Dark Knight' is not available. According to the Superhero API, Batman's power stats are: intelligence: 100, strength: 26, speed: 27, durability: 50, power: 47, and combat: 100.",
  "sources": [
    {
      "type": "dataset",
      "mode": "lookup",
      "detail": "movies dataset, title: 'The Dark Knight' (id 155, match_score: 100.0)"
    },
    {
      "type": "superhero_api",
      "mode": "live_api",
      "detail": "superhero database, queried: 'Batman', resolved: 'Batman' (id 70)"
    }
  ],
  "routing": {
    "needs_dataset": true,
    "dataset_mode": "lookup",
    "movie_titles": ["The Dark Knight"],
    "needs_superhero_api": true,
    "hero_names": ["Batman"]
  }
}
```

### 2. Structured Movie Query
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 3 most popular movies released in 2019?"}'
```

### 3. Pure Superhero Query
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Spider-Man'\''s powerstats and alter ego?"}'
```

---

## Running Automated Tests

A comprehensive pytest suite covers data hygiene, fuzzy search, BM25 keyword matching, ranking vote floors, Superhero API caching and retries, LLM fallback failover, and endpoint validation:

```bash
pytest tests/ -v
```

