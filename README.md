# Simple & Scalable LLM Council

A lightweight FastAPI app where multiple LLM agents answer the same prompt, then democratically vote on response quality to choose a winner.

This gives you:
- Diverse model perspectives on one question
- A real voting process instead of single-model output
- A clear winning response with transparent score breakdowns

## How It Works

1. You send a prompt to `/api/council`.
2. All enabled agents generate responses in parallel.
3. Each agent acts as a judge and ranks the *other* agents' responses (anonymized during judging).
4. Votes are aggregated with Borda-style scoring.
5. The highest total score wins.

## Voting Mechanism (Borda-Style)

If there are `N` agents:
- Each evaluator ranks `N - 1` other responses from best to worst.
- Points are assigned by rank:
  - 1st gets `N - 1` points
  - 2nd gets `N - 2` points
  - ...
  - last gets `1` point
- Scores are summed across all evaluators.
- The top total is the winner.

The API also returns detailed per-vote data so you can inspect how each result was decided.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

- App: `http://localhost:8000`
- Main endpoint: `POST /api/council`

## Config

Agents are defined in `config.json` with:
- `name`
- `endpoint`
- `model` (optional; empty lets server-side default model be used)
- `enabled`

Add or remove agents by editing `config.json`.
