#!/usr/bin/env python3
"""Numerical Intelligence Transfer -- single entry point reproducing the whole study.

    python main.py                       # full experiment (downloads models on first run)
    python main.py --download-only       # pre-fetch models + STS-B (e.g. on an HPC login node)
    python main.py --smoke               # offline end-to-end check: tiny random encoder + MOCK teacher

Stages: environment -> dataset -> teacher -> baselines (Exp1, Exp2) -> training (Exp3, 3b, 4a, 4b, 6, abs,
ablations) -> evaluation (+Exp5) -> tables -> figures -> reports. Finished training runs are reused
(resume) unless --no-resume is given.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import yaml


# ------------------------------------------------------------------ utilities

class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a):
        msg = f"[{dt.datetime.now().strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def set_seed(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def detect_hardware(cfg):
    import torch
    hw = {"cpu_count": os.cpu_count(), "platform": platform.platform()}
    try:
        with open("/proc/meminfo") as f:
            hw["ram_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except Exception:
        hw["ram_gb"] = None
    hw["cuda"] = torch.cuda.is_available()
    if hw["cuda"]:
        p = torch.cuda.get_device_properties(0)
        hw.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1), bf16=torch.cuda.is_bf16_supported(),
                  n_gpus=torch.cuda.device_count())
    else:
        hw.update(gpu=None, gpu_mem_gb=0, bf16=False)
    dev = cfg["device"]
    hw["device"] = ("cuda" if hw["cuda"] else "cpu") if dev == "auto" else dev
    hw["summary"] = f"{hw.get('gpu') or 'CPU only'} ({hw.get('gpu_mem_gb')} GB), {hw['cpu_count']} CPUs, {hw['ram_gb']} GB RAM"
    return hw


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            if k in ("per_dimension", "cosine_profile"):
                continue
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def build_tiny_encoder(sentences, path):
    """Smoke test only: a randomly initialised 2-layer BERT with a WordPiece vocab trained on the data."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel
    from sentence_transformers import SentenceTransformer, models as stm
    raw = os.path.join(path, "_hf")
    os.makedirs(raw, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(sentences, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers import processors
    from transformers import PreTrainedTokenizerFast
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", mask_token="[MASK]").save_pretrained(raw)
    vocab = tok.get_vocab()
    import torch
    torch.manual_seed(42)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128, max_position_embeddings=128)).save_pretrained(raw)
    tr = stm.Transformer(raw, max_seq_length=64)
    st = SentenceTransformer(modules=[tr, stm.Pooling(tr.get_word_embedding_dimension(), "mean")])
    st.save(path)
    return path


def load_sts(cfg, log):
    try:
        from datasets import load_dataset
        name = cfg["semantic_corpus"]["stsb_dataset"]
        test = load_dataset(name, split="test").to_pandas()
        train = load_dataset(name, split="train").to_pandas()
        gen = list(dict.fromkeys(list(train.sentence1) + list(train.sentence2)))
        rng = random.Random(cfg["seed"]); rng.shuffle(gen)
        log(f"STS-B loaded: test {len(test)} pairs, {len(gen)} unique train sentences")
        return test[["sentence1", "sentence2", "score"]], gen[: cfg["semantic_corpus"]["max_general_sentences"]]
    except Exception as e:  # noqa
        log(f"WARNING: STS-B unavailable ({e}); semantic evaluation skipped, L_sem uses masked numeric sentences")
        return None, None


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    ap.add_argument("--out", default=None, help="output root (default: project dir; smoke: ./smoke_run)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override training.steps_per_epoch")
    ap.add_argument("--lr", type=float, default=None, help="override training.lr")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.smoke:
        cfg["seeds"] = [42]
        cfg["dataset"]["scale"] = 0.25
        cfg["training"].update(epochs=2, steps_per_epoch=6, val_batches=3, batch_size=16)
        cfg["evaluation"]["bootstrap"] = 50
        cfg["teacher"]["max_pairs"] = 300
    if args.steps:
        cfg["training"]["steps_per_epoch"] = args.steps
    if args.lr:
        cfg["training"]["lr"] = args.lr
    out = os.path.abspath(args.out or (os.path.join(ROOT, "smoke_run") if args.smoke else ROOT))
    P = {k: os.path.join(out, v) for k, v in dict(data="data", models="models", results="results", fig="results/figures",
                                                   tables="results/tables", preds="results/predictions", reports="reports").items()}
    for p in P.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(P["models"], "baseline"), exist_ok=True)
    os.makedirs(os.path.join(P["models"], "numerical_transfer"), exist_ok=True)
    log = Logger(os.path.join(P["results"], "run_log.txt"))
    set_seed(cfg["seed"])

    import torch
    import sentence_transformers
    import transformers
    hw = detect_hardware(cfg)
    dev = hw["device"]
    log("Hardware:", hw["summary"], "| device:", dev)

    from teacher_supervision import select_teacher
    tcfg = select_teacher(cfg, hw)

    if args.download_only:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        SentenceTransformer(cfg["models"]["student"])
        for m in {tcfg["model"], cfg["teacher"]["preferred_model"], cfg["teacher"]["fallback_model"]}:
            AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
        load_sts(cfg, log)
        log("Download complete.")
        return

    env = dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
               torch=torch.__version__, sentence_transformers=sentence_transformers.__version__,
               transformers=transformers.__version__, hardware=hw, hardware_summary=hw["summary"],
               student=cfg["models"]["student"] if not args.smoke else "SMOKE tiny random BERT",
               teacher_config=tcfg, smoke=args.smoke)
    T0 = time.time()

    # ---------------------------------------------------------------- 1. dataset
    from dataset_generator import generate_all
    audit = generate_all(cfg, P["data"])
    log("Dataset:", json.dumps(audit))
    rd = lambda n: pd.read_csv(os.path.join(P["data"], n))
    train, val, test = rd("train.csv"), rd("validation.csv"), rd("test.csv")
    tr_pairs, val_pairs, test_pairs = rd("train_pairs.csv"), rd("validation_pairs.csv"), rd("test_pairs.csv")
    tr_trip, val_trip, test_trip = rd("train_triplets.csv"), rd("validation_triplets.csv"), rd("test_triplets.csv")
    stress = rd("stress_words_test.csv")

    # ---------------------------------------------------------------- 2. base student
    from sentence_transformers import SentenceTransformer
    base_path = os.path.join(P["models"], "baseline")
    if args.smoke:
        build_tiny_encoder(list(pd.concat([train, val, test]).sentence), base_path)
        sts, gen = None, None
    else:
        st = SentenceTransformer(cfg["models"]["student"], device=dev)
        st.save(base_path)
        env["student_revision"] = getattr(st[0].auto_model.config, "_commit_hash", None)
        del st
        sts, gen = load_sts(cfg, log)
    if gen is None:
        import re
        gen = list(dict.fromkeys(re.sub(r"[-\d.,e]+(?=\s)", "some", s) for s in train.sentence))[:2000]
        env["semantic_corpus"] = "fallback: training sentences with numbers masked"
    else:
        env["semantic_corpus"] = "STS-B train sentences (labels unused)"

    # ---------------------------------------------------------------- 3. teacher
    from teacher_supervision import LLMTeacher, MockTeacher, run_teacher, sample_teacher_pairs
    teacher_acc, t_pairs, t_g = {}, None, None
    tp_path = os.path.join(P["data"], "teacher_supervision_train.csv")
    if not args.skip_teacher:
        try:
            if os.path.exists(tp_path) and not args.no_resume and not args.smoke:
                t_pairs = pd.read_csv(tp_path)
                t_g = pd.read_csv(os.path.join(P["data"], "teacher_compatibility_train.csv"))
                teacher_acc = json.load(open(os.path.join(P["tables"], "teacher_accuracy.json")))
                log("Teacher supervision reused from disk")
            else:
                budget = sample_teacher_pairs(tr_pairs, tcfg["max_pairs"], cfg["seed"])
                if args.smoke:
                    teacher = MockTeacher(noise=0.2, seed=cfg["seed"])
                else:
                    dtype = tcfg["dtype"] if tcfg["dtype"] != "auto" else ("float16" if dev == "cuda" else "float32")
                    log(f"Loading teacher {tcfg['model']} ({dtype}) on {dev}; budget {len(budget)} pairs")
                    teacher = LLMTeacher(tcfg["model"], dtype, dev, tcfg["batch_size"], tcfg["max_new_tokens"])
                t_pairs, t_g, teacher_acc = run_teacher(teacher, budget, train, tcfg["ratio_fraction"], cfg["seed"])
                t_pairs.to_csv(tp_path, index=False)
                t_g.to_csv(os.path.join(P["data"], "teacher_compatibility_train.csv"), index=False)
                json.dump(teacher_acc, open(os.path.join(P["tables"], "teacher_accuracy.json"), "w"), indent=1)
                del teacher
                if dev == "cuda":
                    torch.cuda.empty_cache()
            log("Teacher accuracy:", json.dumps({k: v for k, v in teacher_acc.items() if k != "by_pair_type"}))
        except Exception as e:
            log(f"TEACHER FAILED: {e}\n{traceback.format_exc()}")
            teacher_acc = {"error": str(e)}

    # ---------------------------------------------------------------- 4. training runs
    from numerical_distillation import encode, math_supervision, teacher_supervision, train_student
    L = cfg["training"]["lambdas"]
    sup_full = math_supervision(tr_pairs, tr_trip, train, cfg)
    val_sup = math_supervision(val_pairs, val_trip, val, cfg)
    runs = []  # (config_name, seed, sup, lambdas, proj_dim, cfg_used)
    for s in cfg["seeds"]:
        runs.append(("exp3_math_full", s, sup_full, L, 0, cfg))
        if t_pairs is not None:
            budget_pairs = tr_pairs.merge(t_pairs[["a_id", "b_id"]], on=["a_id", "b_id"])
            runs.append(("exp3b_math_budget", s, math_supervision(budget_pairs, None, train, cfg), L, 0, cfg))
            runs.append(("exp4a_teacher_raw", s, teacher_supervision(t_pairs, t_g, cfg, verified=False), L, 0, cfg))
            runs.append(("exp4b_teacher_verified", s, teacher_supervision(t_pairs, t_g, cfg, verified=True), L, 0, cfg))
        if cfg["experiments"]["run_projection_head"]:
            runs.append(("exp6_learned_projection", s, sup_full, L, cfg["training"]["proj_dim"], cfg))
    s0 = cfg["seed"]
    if cfg["experiments"]["run_distance_mode_abs"]:
        cabs = copy.deepcopy(cfg); cabs["training"]["distance"]["mode"] = "abs"
        runs.append(("exp3_math_full_absdist", s0, math_supervision(tr_pairs, tr_trip, train, cabs), L, 0, cabs))
    abl_def = {"A_semantic_only": ["sem"], "B_+equivalence": ["sem", "eq"], "C_+ordering": ["sem", "eq", "ord"],
               "D_+distance": ["sem", "eq", "ord", "dist"]}
    if cfg["experiments"]["run_ablation"]:
        for name, keys in abl_def.items():
            runs.append((f"abl_{name}", s0, sup_full, {k: (L[k] if k in keys else 0.0) for k in L}, 0, cfg))

    train_logs = {}
    for name, s, sup, lam, proj, c in runs:
        rdir = os.path.join(P["models"], "numerical_transfer", f"{name}_s{s}")
        lp = os.path.join(rdir, "training_log.json")
        if os.path.exists(lp) and not args.no_resume and not args.smoke:
            train_logs[f"{name}_s{s}"] = json.load(open(lp))
            log(f"[{name}_s{s}] reused checkpoint")
            continue
        try:
            set_seed(s)
            vs = math_supervision(val_pairs, val_trip, val, c)
            train_logs[f"{name}_s{s}"] = train_student(f"{name}_s{s}", base_path, sup, vs, gen, c, lam, rdir, s,
                                                       proj_dim=proj, device=dev, log=log)
        except Exception as e:
            log(f"RUN FAILED {name}_s{s}: {e}\n{traceback.format_exc()}")
            train_logs[f"{name}_s{s}"] = {"error": str(e)}
    json.dump(train_logs, open(os.path.join(P["results"], "training_logs.json"), "w"), indent=1)

    # ---------------------------------------------------------------- 5. evaluation
    from baseline_evaluation import baseline_cosine_table, run_baseline_configs
    from evaluation import Space, evaluate_space
    sents = list(dict.fromkeys(list(train.sentence) + list(val.sentence) + list(test.sentence) + list(stress.sentence)
                               + ([] if sts is None else list(sts.sentence1) + list(sts.sentence2))))
    data = dict(train=train, val=val, test=test, val_pairs=val_pairs, test_pairs=test_pairs,
                test_triplets=test_trip, stress=stress, sts=sts)
    metrics, preds42, emb42 = {}, {}, {}

    def record(cfg_name, seed, res, pred, E=None):
        metrics.setdefault(cfg_name, {})[f"seed_{seed}"] = res
        if seed == s0:
            preds42[cfg_name] = pred["pairs"]
            pred["pairs"].to_csv(os.path.join(P["preds"], f"{cfg_name}_test_pairs.csv"), index=False)
            if E is not None:
                emb42[cfg_name] = E

    E0, _ = encode(base_path, sents, dev)
    log("Evaluating Exp1 / Exp2 / parser-only")
    for name, (m, p, sp) in run_baseline_configs(sents, E0, data, cfg).items():
        for s in cfg["seeds"]:  # deterministic configurations: same result for every seed
            metrics.setdefault(name, {})[f"seed_{s}"] = m
        record(name, s0, m, p, E0 if name == "exp1_original_minilm" else None)
        if name == "exp1_original_minilm":
            baseline_cosine_table(sp, test_pairs).to_csv(os.path.join(P["tables"], "baseline_cosine_analysis.csv"), index=False)

    for name, s, sup, lam, proj, c in runs:
        rdir = os.path.join(P["models"], "numerical_transfer", f"{name}_s{s}")
        if "error" in train_logs.get(f"{name}_s{s}", {}):
            continue
        log(f"Evaluating {name}_s{s}")
        E, Z = encode(rdir, sents, dev)
        mode = "hybrid_proj" if proj else "cos"
        res, pr = evaluate_space(Space(sents, E, Z, mode, c), data, c)
        record(name, s, res, pr, E)
        if name == "exp3_math_full":  # Exp5: fine-tuned embedding + parser channel
            res5, pr5 = evaluate_space(Space(sents, E, None, "hybrid_parser", c), data, c)
            record("exp5_hybrid_finetuned_plus_parser", s, res5, pr5)
    json.dump(metrics, open(os.path.join(P["results"], "metrics.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- 6. tables
    rows = {}
    for cname, by_seed in metrics.items():
        flat = pd.DataFrame([flatten(v) for v in by_seed.values()])
        row = flat.mean().to_dict()
        if len(flat) > 1:
            row.update({k + "__std": v for k, v in flat.std(ddof=1).to_dict().items()})
        row["n_seeds"] = len(flat)
        rows[cname] = row
    agg = pd.DataFrame(rows).T
    agg.to_csv(os.path.join(P["tables"], "all_configurations_mean_std.csv"))
    main_cfgs = [c for c in agg.index if not c.startswith("abl_")]
    agg.loc[main_cfgs].to_csv(os.path.join(P["tables"], "main_comparison.csv"))
    abl = None
    if cfg["experiments"]["run_ablation"]:
        order = [f"abl_{k}" for k in abl_def] + ["exp3_math_full"]
        rows_abl = {}
        for k in order:
            if k in metrics and f"seed_{s0}" in metrics[k]:
                rows_abl[k.replace("exp3_math_full", "E_full (+context)").replace("abl_", "")] = flatten(metrics[k][f"seed_{s0}"])
        abl = pd.DataFrame(rows_abl).T
        abl.to_csv(os.path.join(P["tables"], "ablation_seed42.csv"))
    for cname in metrics:
        pdim = metrics[cname].get(f"seed_{s0}", {}).get("ordering", {}).get("per_dimension")
        if pdim:
            pd.DataFrame(pdim).T.to_csv(os.path.join(P["tables"], f"ordering_per_dimension_{cname}.csv"))

    # ---------------------------------------------------------------- 7. figures
    import visualisation as V
    try:
        sel = {k: preds42[k] for k in ("exp1_original_minilm", "exp3_math_full", "exp4b_teacher_verified") if k in preds42}
        V.fig_cosine_by_pair_type(sel, P["fig"])
        V.fig_distance_scatter({k: preds42[k] for k in ("exp1_original_minilm", "exp3_math_full") if k in preds42}, P["fig"])
        show = [c for c in ["exp1_original_minilm", "exp2_parser_plus_original", "exp3_math_full", "exp3b_math_budget",
                            "exp4a_teacher_raw", "exp4b_teacher_verified", "exp5_hybrid_finetuned_plus_parser",
                            "exp6_learned_projection"] if c in agg.index]
        V.fig_retrieval(agg.loc[show], P["fig"])
        V.fig_ordering(agg.loc[show], P["fig"])
        V.fig_semantic(agg.loc[show], P["fig"])
        V.fig_pca({k: (sents, emb42[k]) for k in ("exp1_original_minilm", "exp3_math_full") if k in emb42}, test, P["fig"])
        if teacher_acc and "error" not in teacher_acc:
            V.fig_teacher(teacher_acc, P["fig"])
        if abl is not None and len(abl):
            V.fig_ablation(abl, P["fig"])
        V.fig_training_curves({k: v["history"] for k, v in train_logs.items() if "history" in v and k.endswith(f"_s{s0}")}, P["fig"])
    except Exception as e:
        log(f"FIGURE FAILED: {e}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- 8. reports
    from report_writer import write_methodology, write_report
    env["total_seconds"] = round(time.time() - T0, 1)
    json.dump(env, open(os.path.join(P["results"], "environment.json"), "w"), indent=1, default=str)
    write_methodology(cfg, env, audit, os.path.join(P["reports"], "methodology.md"))
    write_report(metrics, agg.copy(), abl, teacher_acc if "error" not in teacher_acc else {}, cfg, env, audit,
                 os.path.join(P["reports"], "experimental_report.md"), mock=args.smoke)
    log(f"Done in {env['total_seconds']} s. Report: {os.path.join(P['reports'], 'experimental_report.md')}")


if __name__ == "__main__":
    main()
