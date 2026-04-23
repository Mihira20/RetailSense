import os
import json
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from datetime import datetime, timedelta

load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_external_signals(date_str, category="retail"):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    month_year = date.strftime("%B %Y")
    
    queries = [
        f"retail sales events {month_year}",
        f"weather disasters US {month_year}",
        f"economic news consumer spending {month_year}",
        f"holidays shopping {month_year}"
    ]
    
    all_results = []
    for query in queries:
        try:
            results = tavily_client.search(
                query=query,
                max_results=3,
                search_depth="basic"
            )
            for r in results.get('results', []):
                all_results.append({
                    'title': r.get('title', ''),
                    'content': r.get('content', '')[:200]
                })
        except Exception as e:
            print(f"Search error: {e}")
            continue
    
    return all_results

def score_external_signals(date_str, category="grocery", search_results=None):
    if search_results is None:
        search_results = search_external_signals(date_str, category)
    
    news_text = "\n".join([
        f"- {r['title']}: {r['content']}"
        for r in search_results[:8]
    ])
    
    prompt = f"""You are a retail demand forecasting assistant.
    
Analyze the following news and events for the week of {date_str} and estimate 
their impact on retail sales for the {category} category at Walmart stores.

NEWS AND EVENTS:
{news_text}

Based on this information, provide a JSON response with exactly this structure:
{{
    "demand_boost": <float between -0.5 and 0.5, where 0 is no impact, 
                    positive means demand increase, negative means decrease>,
    "confidence": <float between 0 and 1>,
    "reason": "<one sentence explanation with no trailing comma>",
    "key_events": <list of 1-3 key events identified>
}}

Respond with ONLY the JSON, no other text."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    raw = response.choices[0].message.content.strip()
    print("RAW LLM RESPONSE:")
    print(raw)
    print("="*50)
    
   # Clean common LLM JSON quirks
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        signal = json.loads(raw)
    except json.JSONDecodeError:
        import re
        # Fix trailing commas before } or ]
        raw_fixed = re.sub(r',(\s*[}\]])', r'\1', raw)
        try:
            signal = json.loads(raw_fixed)
        except json.JSONDecodeError:
            # Last resort — try extracting values manually
            try:
                demand = float(re.search(r'"demand_boost":\s*([-\d.]+)', raw).group(1))
                confidence = float(re.search(r'"confidence":\s*([-\d.]+)', raw).group(1))
                reason = re.search(r'"reason":\s*"([^"]+)"', raw).group(1)
                signal = {
                    "demand_boost": demand,
                    "confidence": confidence,
                    "reason": reason,
                    "key_events": []
                }
            except:
                signal = {
                    "demand_boost": 0.0,
                    "confidence": 0.0,
                    "reason": "Could not parse LLM response",
                    "key_events": []
                }
    
    signal['date'] = date_str
    signal['category'] = category
    return signal

def get_signal_for_date_range(start_date, end_date, category="grocery"):
    signals = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Processing {date_str}...")
        signal = score_external_signals(date_str, category)
        signals.append(signal)
        current += timedelta(weeks=1)
    
    return signals

if __name__ == "__main__":
    print("Testing LLM Signal Agent...")
    print("="*50)
    
    signal = score_external_signals("2011-08-27", category="grocery")
    print(json.dumps(signal, indent=2))