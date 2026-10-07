"""Temperature ablation — GPT→Gemini→GLM, Protocols A-E, 2 cases, T=0.3/0.7."""
import csv, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from domains import CASES, MAX_OUTPUT_TOKENS
from protocols import build_prompt, build_initial_prompt, make_system_prompt

OUTPUT = SCRIPT_DIR / "output" / "ablation_temperature.csv"
CASES_SELECTED = ["math_short", "geo_short"]
PROTOS = ["A", "B", "C", "D", "E"]
TEMPS = [0.3, 0.7]
RUNS = 3
WORKERS = 6
WRITE_LOCK = threading.Lock()

def get_client(model):
    if model == "gpt":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-5.4-mini"
    elif model == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"]), "gemini-3.1-flash-lite"
    elif model == "glm":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["ZHIPUAI_API_KEY"], base_url="https://open.bigmodel.cn/api/paas/v4"), "glm-4.5-air"

def call_api(client, model, prompt, temp, model_id):
    if model == "gpt":
        r = client.chat.completions.create(model=model_id, messages=[{"role":"user","content":prompt}], max_completion_tokens=MAX_OUTPUT_TOKENS, temperature=temp)
        return r.choices[0].message.content or ""
    elif model == "gemini":
        r = client.models.generate_content(model="models/"+model_id, contents=prompt, config={"max_output_tokens":MAX_OUTPUT_TOKENS,"temperature":temp})
        return r.text or ""
    elif model == "glm":
        r = client.chat.completions.create(model=model_id, messages=[{"role":"user","content":prompt}], max_tokens=MAX_OUTPUT_TOKENS, temperature=temp)
        return r.choices[0].message.content or ""

def build_baseline(model_key, case_id, run_num):
    """Build turns 1-4 at T=0. Returns real_responses dict."""
    case = CASES[case_id]
    client, model_id = get_client(model_key)
    sys_prompt = make_system_prompt(case)
    real_responses = {}
    
    for turn in range(1, 5):
        if turn == 1:
            prompt = build_initial_prompt(case, "A", sys_prompt)
        else:
            prompt, _ = build_prompt(case, "A", turn, real_responses)
        
        for attempt in range(2):
            try:
                out = call_api(client, model_key, prompt, 0.0, model_id)
                real_responses[turn] = out
                break
            except Exception as e:
                if attempt == 1:
                    real_responses[turn] = f"[ERROR: {e}]"
                time.sleep(3)
    
    return real_responses

def run_session(args):
    """One (case, run) session: build baseline once, test T5 at each (proto, temp)."""
    model_key, case_id, run_num = args
    case = CASES[case_id]
    client, model_id = get_client(model_key)
    
    # Build baseline once
    real_responses = build_baseline(model_key, case_id, run_num)
    
    # Test T5 at each (protocol, temperature)
    results = []
    for proto in PROTOS:
        prompt, _ = build_prompt(case, proto, 5, real_responses)
        for temp in TEMPS:
            try:
                start = time.time()
                out = call_api(client, model_key, prompt, temp, model_id)
                elapsed = round(time.time() - start, 2)
                results.append({"model":model_key,"caseId":case_id,"protocol":proto,"run":run_num,"temperature":temp,"turn":5,"elapsed_s":elapsed,"prompt":prompt[:2000],"rawOutput":out[:2000]})
            except Exception as e:
                results.append({"model":model_key,"caseId":case_id,"protocol":proto,"run":run_num,"temperature":temp,"turn":5,"elapsed_s":0,"prompt":"ERROR","rawOutput":f"ERROR:{e}"})
                time.sleep(3)
    
    return results

if __name__ == "__main__":
    MODELS = [
        ("gpt", "GPT-5.4 Mini", "OPENAI_API_KEY"),
        ("gemini", "Gemini-3.1 Flash-Lite", "GEMINI_API_KEY"),
        ("glm", "GLM-4.5-Air", "ZHIPUAI_API_KEY"),
    ]
    
    all_rows = []
    
    for model_key, model_name, env_key in MODELS:
        if not os.environ.get(env_key):
            print(f"SKIP {model_name}: {env_key} not set")
            continue
        
        tasks = [(model_key, case, run) for case in CASES_SELECTED for run in range(1, RUNS + 1)]
        
        print(f"\n{'='*50}")
        print(f" {model_name} — {len(tasks)} sessions, {WORKERS} workers")
        print(f"{'='*50}")
        
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(run_session, t): t for t in tasks}
            for f in as_completed(futures):
                rows = f.result()
                with WRITE_LOCK:
                    all_rows.extend(rows)
                case_id, run_num = futures[f][1], futures[f][2]
                print(f"  ✓ {case_id} run={run_num} ({len(rows)} tests)", flush=True)
        
        print(f"  {model_name} done")
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with open(OUTPUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model","caseId","protocol","run","temperature","turn","elapsed_s","prompt","rawOutput"])
            w.writeheader()
            w.writerows(all_rows)
    
    print(f"\n✓ All done. {len(all_rows)} rows → {OUTPUT}")
