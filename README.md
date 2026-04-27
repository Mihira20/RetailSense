# RetailSense 
**AI-Augmented Demand Forecasting with External Signal Intelligence**

End-to-end demand forecasting pipeline for Walmart retail data, augmented with an LLM layer that reads real-world news to inject external signals into forecasts.

---

## The Project

Built a forecasting pipeline on the Walmart Store Sales dataset (421,570 records, 45 stores, 81 departments, 2010-2012) and tested three models of increasing sophistication:

| Model | MAE | WMAE |
|-------|-----|------|
| Naive Baseline | $3,413 | $2,984 |
| ARIMA | $3,023 | $3,178 |
| Prophet | $1,703 | $1,998 |
| **LightGBM** | **$1,458** | **$1,532** |

LightGBM won on both metrics, validated with 5-fold time series cross validation (mean MAE: $4,319 ± $489).

---

## The LLM Experiment - What I Tried and Why It Failed

The core hypothesis: *news headlines contain demand signals that structured data misses.* A hurricane causes panic buying. An economic shock suppresses spending. A viral trend spikes demand overnight. None of this appears in historical sales data.

We built an LLM agent (Groq/Llama 3 + Tavily search) that:
1. Takes a date + product category as input
2. Searches for relevant news headlines
3. Outputs a structured demand signal: `{"demand_boost": 0.15, "reason": "..."}`
4. Injects that signal into the forecast as a feature

**Why it failed:** Tavily returns live/recent news, not 2010-2012 archives. Correlating 2022 news sentiment with 2012 sales is meaningless. Additionally, the LLM outputs coarse discrete values (-0.2, -0.1, 0.1, 0.2) that lack the granularity needed to capture sales variance. Correlation analysis confirmed this: r=0.04.

**The deeper finding:** Feature importance analysis showed department identity (score: 7,316) dominates all other signals. Structural factors — what you're selling, store size, location — matter far more than temporal or external signals. This explains why even holiday flags scored near zero in importance.

---

## What Would Make the LLM Layer Work

1. **Historical news API** - access to 2010-2012 news archives (e.g. GDELT, NewsAPI historical)
2. **Department-specific signals** - a hurricane signal means different things for Electronics vs Grocery
3. **Granular scoring** - fine-tuned LLM that outputs continuous scores, not rounded values
4. **Longer evaluation window** - test on periods with known disruptions (Hurricane Sandy, 2012)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data | Kaggle Walmart Sales Dataset |
| Processing | Pandas |
| Forecasting | ARIMA, Prophet, LightGBM |
| LLM | Groq API (Llama 3.3) |
| LLM Orchestration | LangChain |
| Web Search | Tavily API |
| Visualization | Matplotlib, Seaborn |

---

## Data

Download from [Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) and place in `data/raw/`.

---

## Run It

```bash
conda create -n retailsense python=3.11
conda activate retailsense
pip install -r requirements.txt
python src/data_pipeline.py
```

Add your API keys to `.env`:
```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```
