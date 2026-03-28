import json
import asyncio
import re
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import requests

app = FastAPI()
config = json.load(open("config.json"))
lifetime_scores: Dict[str, Dict[str, Any]] = {}

class PromptRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    name: str
    response: str

class ScoreResult(BaseModel):
    agent_name: str
    correctness: float
    helpfulness: float
    clarity: float
    reasoning: float
    total: float

class CouncilResult(BaseModel):
    winner: str
    winner_response: str
    all_responses: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]

def load_config():
    global config
    config = json.load(open("config.json"))

def get_enabled_agents():
    return [a for a in config["agents"] if a.get("enabled", True)]

def init_lifetime_scores() -> None:
    for agent in get_enabled_agents():
        name = agent["name"]
        if name not in lifetime_scores:
            lifetime_scores[name] = {
                "total_points": 0,
                "first_place": 0,
                "second_place": 0,
                "third_place": 0,
                "votes": 0,
                "rounds": 0,
                "last_response_latency_ms": None,
                "last_response_error": None,
            }

async def call_llm(agent: Dict, messages: List[Dict], purpose: str) -> str:
    payload = {
        "messages": messages,
        "temperature": 0.7,
    }
    model = agent.get("model")
    if model:
        payload["model"] = model

    def do_request() -> str:
        start = time.perf_counter()
        try:
            resp = requests.post(agent["endpoint"], json=payload, timeout=120)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                preview = resp.text[:300] if resp.text else ""
                raise ValueError(f"Non-JSON response: {preview}")
            content = data["choices"][0]["message"]["content"]
            latency_ms = int((time.perf_counter() - start) * 1000)
            init_lifetime_scores()
            if purpose == "response":
                lifetime_scores[agent["name"]]["last_response_latency_ms"] = latency_ms
                lifetime_scores[agent["name"]]["last_response_error"] = None
            return content
        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            init_lifetime_scores()
            if purpose == "response":
                lifetime_scores[agent["name"]]["last_response_latency_ms"] = latency_ms
                lifetime_scores[agent["name"]]["last_response_error"] = str(e)
            return f"Error: {str(e)}"

    return await asyncio.to_thread(do_request)

async def get_rankings(evaluator_agent: Dict, prompt: str, responses: List[Dict]) -> Dict[str, Any]:
    ranking_prompt = f"""You are a judge in a democratic council. Rank all responses by overall quality.

User's original question: {prompt}

Responses to rank (numbered list, anonymized):
"""
    candidates: List[Dict[str, Any]] = []
    for idx, r in enumerate(responses):
        if r["name"] == evaluator_agent["name"]:
            continue
        local_idx = len(candidates)
        label = chr(ord("A") + local_idx)
        candidates.append({"local_idx": local_idx, "global_idx": idx, "name": r["name"], "label": label})
        ranking_prompt += f"\n[{local_idx}] Response {label}:\n{r['response']}\n"

    ranking_prompt += """

Rank these responses from BEST to WORST. Return ONLY a JSON array of the numbered indices from the list above (0-based), ordered from best to worst.
Example: [1, 0, 2]

Return ONLY the JSON array, no other text.
"""

    messages = [
        {"role": "system", "content": config["scoring_prompt"]},
        {"role": "user", "content": ranking_prompt}
    ]

    result = await call_llm(evaluator_agent, messages, "ranking")
    
    def parse_rankings(raw: str) -> List[Any]:
        try:
            return json.loads(raw)
        except Exception:
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    return []
        return []

    rankings_raw = parse_rankings(result)
    if not isinstance(rankings_raw, list):
        return {
            "rankings": [],
            "debug": {
                "evaluator": evaluator_agent["name"],
                "raw": result,
                "parsed": rankings_raw,
                "mapped": [],
                "candidates": candidates,
            },
        }

    # Map local indices (or names) back to global indices.
    mapped: List[int] = []
    name_to_global = {c["name"].casefold(): c["global_idx"] for c in candidates}
    label_to_global = {c["label"].casefold(): c["global_idx"] for c in candidates}
    for item in rankings_raw:
        if isinstance(item, int):
            if 0 <= item < len(candidates):
                mapped.append(candidates[item]["global_idx"])
        elif isinstance(item, str):
            if item.isdigit():
                local_idx = int(item)
                if 0 <= local_idx < len(candidates):
                    mapped.append(candidates[local_idx]["global_idx"])
            else:
                key = item.casefold()
                if key in name_to_global:
                    mapped.append(name_to_global[key])
                elif key in label_to_global:
                    mapped.append(label_to_global[key])

    # Preserve order, remove duplicates
    seen = set()
    deduped: List[int] = []
    for idx in mapped:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return {
        "rankings": deduped,
        "debug": {
            "evaluator": evaluator_agent["name"],
            "raw": result,
            "parsed": rankings_raw,
            "mapped": deduped,
            "candidates": candidates,
        },
    }

async def generate_all_responses(prompt: str) -> List[Dict]:
    agents = get_enabled_agents()
    system_msg = {"role": "system", "content": config["system_prompt"]}
    user_msg = {"role": "user", "content": prompt}
    messages = [system_msg, user_msg]

    async def timed_call(agent: Dict) -> Dict[str, Any]:
        start = time.perf_counter()
        response = await call_llm(agent, messages, "response")
        latency_ms = int((time.perf_counter() - start) * 1000)
        error = response if response.startswith("Error:") else None
        return {"name": agent["name"], "response": response, "latency_ms": latency_ms, "error": error}

    tasks = [timed_call(agent) for agent in agents]
    results = await asyncio.gather(*tasks)

    responses = []
    for item in results:
        responses.append({
            "name": item["name"],
            "response": item["response"],
            "latency_ms": item["latency_ms"],
            "error": item["error"],
        })
    return responses

async def score_responses(prompt: str, responses: List[Dict]) -> Dict[str, Any]:
    agents = get_enabled_agents()
    num_agents = len(responses)
    points_per_vote = num_agents - 1

    all_votes = []

    ranking_tasks = [get_rankings(evaluator, prompt, responses) for evaluator in agents]
    rankings_list = await asyncio.gather(*ranking_tasks)
    debug_rankings = [r["debug"] for r in rankings_list]

    for evaluator, rankings in zip(agents, rankings_list):
        for rank_position, idx in enumerate(rankings["rankings"]):
            if idx < 0 or idx >= len(responses):
                continue
            if responses[idx]["name"] == evaluator["name"]:
                continue

            points = points_per_vote - rank_position

            all_votes.append({
                "evaluator": evaluator["name"],
                "evaluated": responses[idx]["name"],
                "rank": rank_position + 1,
                "points": points,
            })

    return {"votes": all_votes, "debug_rankings": debug_rankings}

def aggregate_scores(votes: List[Dict], responses: List[Dict]) -> List[Dict]:
    num_agents = len(responses)
    max_points = num_agents - 1
    
    agent_totals = {r["name"]: {"points": 0, "first_place": 0, "second_place": 0, "third_place": 0, "votes": 0} for r in responses}
    
    for v in votes:
        evaluated = v["evaluated"]
        agent_totals[evaluated]["points"] += v["points"]
        agent_totals[evaluated]["votes"] += 1
        if v["rank"] == 1:
            agent_totals[evaluated]["first_place"] += 1
        elif v["rank"] == 2:
            agent_totals[evaluated]["second_place"] += 1
        else:
            agent_totals[evaluated]["third_place"] += 1
    
    final_scores = []
    for name, data in agent_totals.items():
        total = data["points"]
        final_scores.append({
            "agent_name": name,
            "total": total,
            "first_place": data["first_place"],
            "second_place": data["second_place"],
            "third_place": data["third_place"]
        })
    
    return sorted(final_scores, key=lambda x: x["total"], reverse=True)

def update_lifetime_scores(votes: List[Dict], final_scores: List[Dict]) -> None:
    init_lifetime_scores()
    for score in final_scores:
        name = score["agent_name"]
        if name not in lifetime_scores:
            lifetime_scores[name] = {
                "total_points": 0,
                "first_place": 0,
                "second_place": 0,
                "third_place": 0,
                "votes": 0,
                "rounds": 0,
            }
        lifetime_scores[name]["total_points"] += score["total"]
        lifetime_scores[name]["first_place"] += score["first_place"]
        lifetime_scores[name]["second_place"] += score["second_place"]
        lifetime_scores[name]["third_place"] += score["third_place"]
        lifetime_scores[name]["rounds"] += 1

    for v in votes:
        evaluated = v["evaluated"]
        if evaluated in lifetime_scores:
            lifetime_scores[evaluated]["votes"] += 1

@app.post("/api/council")
async def council(request: PromptRequest):
    try:
        prompt = request.message

        responses = await generate_all_responses(prompt)

        score_result = await score_responses(prompt, responses)
        votes = score_result["votes"]

        final_scores = aggregate_scores(votes, responses)
        update_lifetime_scores(votes, final_scores)

        if not final_scores:
            return {"error": "No valid scores generated"}

        winner = final_scores[0]
        winner_response = next(r["response"] for r in responses if r["name"] == winner["agent_name"])

        return {
            "winner": winner["agent_name"],
            "winner_response": winner_response,
            "all_responses": responses,
            "scores": final_scores,
            "detailed_votes": votes,
            "debug_rankings": score_result["debug_rankings"],
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"{type(e).__name__}: {str(e)}",
                "trace": traceback.format_exc(limit=5),
            },
        )

@app.get("/api/config")
async def get_config():
    return {"agents": config["agents"]}

@app.get("/api/scores")
async def get_scores():
    init_lifetime_scores()
    scores = []
    for name, data in lifetime_scores.items():
        scores.append({
            "agent_name": name,
            "total_points": data["total_points"],
            "first_place": data["first_place"],
            "second_place": data["second_place"],
            "third_place": data["third_place"],
            "votes": data["votes"],
            "rounds": data["rounds"],
            "last_response_latency_ms": data["last_response_latency_ms"],
            "last_response_error": data["last_response_error"],
        })
    return {"scores": sorted(scores, key=lambda x: x["total_points"], reverse=True)}

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Council AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
            background: radial-gradient(1200px 600px at 20% 0%, #1f2937 0%, #0b0d10 55%, #07080a 100%);
            color: #e7e7ea;
            min-height: 100vh;
        }
        .container { max-width: 1500px; margin: 0 auto; padding: 24px; }
        .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
        .brand { font-size: 14px; letter-spacing: 3px; text-transform: uppercase; color: #9ca3af; }
        .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 24px; }

        .chat-container {
            background: linear-gradient(180deg, rgba(24, 24, 28, 0.9), rgba(15, 15, 18, 0.9));
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 20px;
            min-height: 420px;
        }
        .message { padding: 14px 16px; margin-bottom: 12px; border-radius: 10px; line-height: 1.6; }
        .message.user { background: #212635; margin-left: 80px; border: 1px solid #2f3446; }
        .message.assistant { background: #141821; margin-right: 80px; border: 1px solid #1f2433; }
        .message.winner { border: 2px solid #4ade80; background: #142018; }
        .message.assistant.non-winner { opacity: 0.74; font-size: 0.9em; }
        .agent-label { font-size: 12px; color: #9ca3af; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
        .winner-badge { display: inline-block; background: #4ade80; color: #0b0d10; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }

        .message .content table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        .message .content th, .message .content td { border: 1px solid #2a3242; padding: 8px 10px; text-align: left; }
        .message .content th { background: #1b2232; color: #e5e7eb; }
        .message .content code { background: #0f141f; border: 1px solid #232b3a; padding: 2px 6px; border-radius: 4px; white-space: pre; }
        .message .content pre { background: #0f141f; border: 1px solid #232b3a; padding: 12px; border-radius: 8px; overflow-x: auto; }
        .message .content pre code { white-space: pre; }

        .input-container { display: flex; gap: 10px; margin-top: 14px; }
        input { flex: 1; padding: 14px; border: 1px solid #2a3242; border-radius: 10px; background: #0f131b; color: #fff; font-size: 16px; }
        input:focus { outline: 2px solid #4ade80; }
        button { padding: 14px 28px; background: #4ade80; color: #0b0d10; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #22c55e; }
        button:disabled { background: #475569; cursor: not-allowed; }

        .sidebar { display: flex; flex-direction: column; gap: 16px; }
        .panel { background: #12161f; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }
        .panel h2 { font-size: 14px; letter-spacing: 1px; text-transform: uppercase; color: #a1a1aa; margin-bottom: 12px; }

        .scores-panel { display: none; }
        .score-row { display: grid; grid-template-columns: 1fr 70px; align-items: center; padding: 10px; background: #151b27; border-radius: 8px; margin-bottom: 8px; }
        .score-row.winner { border: 1px solid #4ade80; }
        .agent-name { font-weight: 600; color: #e5e7eb; }
        .score-value { text-align: right; font-weight: 700; color: #4ade80; }
        .score-details { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; font-size: 11px; color: #94a3b8; }
        .detail-item { text-align: center; padding: 6px; background: #0f141f; border-radius: 4px; }
        .score-meta { margin-top: 8px; font-size: 11px; color: #9ca3af; }

        .lifetime-row { padding: 10px; background: #151b27; border-radius: 8px; margin-bottom: 8px; }
        .lifetime-top { display: flex; justify-content: space-between; align-items: center; }
        .lifetime-points { color: #f59e0b; font-weight: 700; }
        .lifetime-details { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; font-size: 11px; color: #94a3b8; }
        .lifetime-meta { margin-top: 8px; font-size: 11px; color: #9ca3af; }
        .error-badge { color: #fca5a5; background: #3b0a0a; border: 1px solid #7f1d1d; padding: 4px 6px; border-radius: 6px; display: inline-block; margin-top: 6px; }

        .loading { text-align: center; padding: 40px; color: #9ca3af; }
        .thinking { display: flex; gap: 8px; justify-content: center; }
        .dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out; }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

        @media (max-width: 980px) {
            .layout { grid-template-columns: 1fr; }
            .message.user { margin-left: 0; }
            .message.assistant { margin-right: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="topbar">
            <div class="brand">Council AI</div>
        </div>

        <div class="layout">
            <div>
                <div class="chat-container" id="chat"></div>

                <div class="input-container">
                    <input type="text" id="prompt" placeholder="Ask your question..." onkeypress="if(event.key==='Enter')send()">
                    <button onclick="send()" id="sendBtn">Send</button>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel scores-panel" id="scoresPanel">
                    <h2>Round Scores</h2>
                    <div id="scores"></div>
                </div>
                <div class="panel" id="lifetimePanel">
                    <h2>Lifetime Scores</h2>
                    <div id="lifetimeScores"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let chatHistory = [];

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function formatInline(text) {
            let safe = escapeHtml(text);
            safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
            safe = safe.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
            safe = safe.replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
            return safe;
        }

        function renderTable(lines, startIndex) {
            const rows = [];
            let i = startIndex;
            for (; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line.includes("|")) break;
                rows.push(line);
            }
            if (rows.length < 2) {
                return { html: formatInline(lines[startIndex]) + "<br>", nextIndex: startIndex + 1 };
            }
            const header = rows[0];
            const bodyRows = rows.slice(2);
            const parseRow = (row) => row.replace(/^\\|/, "").replace(/\\|$/, "").split("|").map(c => c.trim());
            const headCells = parseRow(header);
            const bodyCells = bodyRows.map(parseRow);
            let html = "<table><thead><tr>";
            headCells.forEach(c => { html += `<th>${formatInline(c)}</th>`; });
            html += "</tr></thead><tbody>";
            bodyCells.forEach(r => {
                html += "<tr>";
                r.forEach(c => { html += `<td>${formatInline(c)}</td>`; });
                html += "</tr>";
            });
            html += "</tbody></table>";
            return { html, nextIndex: i };
        }

        function renderMarkdown(md) {
            const parts = md.split(/```/);
            let html = "";
            for (let i = 0; i < parts.length; i++) {
                if (i % 2 === 1) {
                    let codeBlock = parts[i];
                    let lines = codeBlock.split("\\n");
                    if (lines.length > 1 && /^[a-z0-9-]+$/i.test(lines[0].trim())) {
                        lines.shift();
                    }
                    codeBlock = lines.join("\\n");
                    html += `<pre><code>${escapeHtml(codeBlock)}</code></pre>`;
                } else {
                    const lines = parts[i].split("\\n");
                    for (let j = 0; j < lines.length; j++) {
                        const line = lines[j];
                        const nextLine = lines[j + 1] || "";
                        if (line.includes("|") && /^\\s*\\|?[-:\\s|]+\\|?\\s*$/.test(nextLine)) {
                            const rendered = renderTable(lines, j);
                            html += rendered.html;
                            j = rendered.nextIndex - 1;
                            continue;
                        }
                        html += formatInline(line) + "<br>";
                    }
                }
            }
            return html;
        }

        function addMessage(content, role, agent = null, isWinner = false) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            let cls = `message ${role}${isWinner ? ' winner' : ''}`;
            if (role === 'assistant' && !isWinner) {
                cls += ' non-winner';
            }
            div.className = cls;
            
            let html = '';
            if (agent) {
                html += `<div class="agent-label">${agent}${isWinner ? '<span class="winner-badge">WINNER</span>' : ''}</div>`;
            }
            html += `<div class="content">${renderMarkdown(content)}</div>`;
            div.innerHTML = html;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }
        
        function showScores(scores, responseMeta) {
            const panel = document.getElementById('scoresPanel');
            const container = document.getElementById('scores');
            panel.style.display = 'block';
            container.innerHTML = '';

            scores.forEach((s, i) => {
                const meta = responseMeta[s.agent_name] || {};
                const respLatency = meta.latency_ms != null ? `${meta.latency_ms} ms` : '—';
                const err = meta.error || '';
                const row = document.createElement('div');
                row.className = `score-row${i === 0 ? ' winner' : ''}`;
                row.innerHTML = `
                    <div class="agent-name">${s.agent_name}</div>
                    <div class="score-value">${s.total}</div>
                    <div class="score-details">
                        <div class="detail-item" style="color:#fbbf24">1st: ${s.first_place}</div>
                        <div class="detail-item" style="color:#94a3b8">2nd: ${s.second_place}</div>
                        <div class="detail-item" style="color:#ef4444">3rd: ${s.third_place}</div>
                    </div>
                    <div class="score-meta">Resp time: ${respLatency}</div>
                    ${err ? `<div class="error-badge">Last error: ${err}</div>` : ''}
                `;
                container.appendChild(row);
            });
        }

        function showLifetimeScores(scores) {
            const container = document.getElementById('lifetimeScores');
            container.innerHTML = '';
            scores.forEach((s) => {
                const row = document.createElement('div');
                row.className = 'lifetime-row';
                const respLatency = s.last_response_latency_ms != null ? `${s.last_response_latency_ms} ms` : '—';
                const err = s.last_response_error || '';
                row.innerHTML = `
                    <div class="lifetime-top">
                        <div class="agent-name">${s.agent_name}</div>
                        <div class="lifetime-points">${s.total_points}</div>
                    </div>
                    <div class="lifetime-details">
                        <div class="detail-item">1st: ${s.first_place}</div>
                        <div class="detail-item">2nd: ${s.second_place}</div>
                        <div class="detail-item">3rd: ${s.third_place}</div>
                    </div>
                    <div class="lifetime-meta">Resp time: ${respLatency}</div>
                    ${err ? `<div class="error-badge">Last error: ${err}</div>` : ''}
                `;
                container.appendChild(row);
            });
        }

        async function refreshLifetimeScores() {
            try {
                const resp = await fetch('/api/scores');
                const data = await resp.json();
                showLifetimeScores(data.scores || []);
            } catch (e) {
                console.error('Failed to load lifetime scores', e);
            }
        }
        
        async function send() {
            const input = document.getElementById('prompt');
            const btn = document.getElementById('sendBtn');
            const prompt = input.value.trim();
            if (!prompt) return;
            
            addMessage(prompt, 'user');
            chatHistory.push({role: 'user', content: prompt});
            input.value = '';
            
            btn.disabled = true;
            input.disabled = true;
            
            const loading = document.createElement('div');
            loading.id = 'loading';
            loading.className = 'loading';
            loading.innerHTML = '<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><p>Council is deliberating...</p>';
            document.getElementById('chat').appendChild(loading);
            
            try {
                const resp = await fetch('/api/council', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: prompt})
                });
                const data = await resp.json();
                
                document.getElementById('loading').remove();
                
                if (data.error) {
                    let errText = 'Error: ' + data.error;
                    if (data.trace) {
                        errText += '\\n\\n' + data.trace;
                    }
                    addMessage(errText, 'assistant');
                } else {
                    data.all_responses.forEach(r => {
                        const isWinner = r.name === data.winner;
                        addMessage(r.response, 'assistant', r.name, isWinner);
                    });
                    
                    const responseMeta = {};
                    data.all_responses.forEach(r => {
                        responseMeta[r.name] = { latency_ms: r.latency_ms, error: r.error };
                    });
                    showScores(data.scores, responseMeta);
                    refreshLifetimeScores();
                }
            } catch (e) {
                document.getElementById('loading').remove();
                addMessage('Error: ' + e.message, 'assistant');
            }
            
            btn.disabled = false;
            input.disabled = false;
            input.focus();
        }

        refreshLifetimeScores();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
