"""Smoke-test evaluation. For trustworthy architecture decisions, expand and manually label questions."""
import json, time, statistics, argparse
from pathlib import Path
from app.config import DATA_DIR
from app.pipeline import answer
from app.retrieval import hybrid_search, dense_search

def reciprocal_rank(rows, relevant):
    for i, row in enumerate(rows, 1):
        if any(s.lower() in str(row.get("metadata",{}).get("source","")).lower() for s in relevant):
            return 1/i
    return 0.0

def run_evaluation(question_path=None):
    path = Path(question_path) if question_path else DATA_DIR / "eval_questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    report = {"note":"Small smoke-test dataset; not a statistically valid benchmark.", "results":[], "aggregate":{}}
    latencies, rrs = [], []
    for item in questions:
        q = item["question"]
        t = time.perf_counter()
        retrieval = hybrid_search(q, 10)
        retrieval_ms = (time.perf_counter()-t)*1000
        relevant = item.get("relevant_sources", [])
        rr = reciprocal_rank(retrieval, relevant) if relevant != ["none"] else 0.0
        answer_data = None
        try: answer_data = answer(q, mode="hybrid")
        except Exception as exc: answer_data = {"error":str(exc)}
        latencies.append(retrieval_ms); rrs.append(rr)
        report["results"].append({"question":q, "retrieval_ms":round(retrieval_ms,2),
            "reciprocal_rank":rr, "retrieved_sources":[r.get("metadata",{}).get("source") for r in retrieval[:5]],
            "answer_latency_ms":answer_data.get("latency_ms"), "answer_error":answer_data.get("error")})
    report["aggregate"] = {"questions":len(questions), "mean_retrieval_ms":round(statistics.mean(latencies),2) if latencies else 0,
        "p95_retrieval_ms":round(sorted(latencies)[max(0, int(.95*len(latencies))-1)],2) if latencies else 0,
        "mrr":round(statistics.mean(rrs),4) if rrs else 0}
    out = DATA_DIR / "eval_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--questions", default=None)
    args=parser.parse_args(); print(json.dumps(run_evaluation(args.questions), indent=2))
