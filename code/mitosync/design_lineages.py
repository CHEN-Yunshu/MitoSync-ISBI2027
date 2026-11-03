#!/usr/bin/env python
"""design_lineages.py -- M1 lineages of ONE MitoSync design corpus (pre-registration amendment 4 D3).  CPU only, PYC.

    $PYC design_lineages.py --dataset PhC-C2DL-PSC --corpus i1_k1 --interval 1 --kappa 1 \
        --densities 1.0:67:410,1.6:67:411,2.4:66:412 --phase_lock 1,2 --root $N/data/mitosync/design/PhC-C2DL-PSC/i1_k1

Programmer (amendment 4 D1 = $N/eval/mitosync/m1_version_choice.json, head only): interval 1 -> choice[ds].interval1,
interval 8 -> choice[ds].x8 (script + flags exactly as recorded there).  Fixed settings (D3): 17 frames, cond_v2, entries 1,
overlap none (D2: no overlap in the tracker design), head theta / library / root, event budget kappa, phase lock K* (DP3).
Density mixture: one simulator run per density d (density_scale d x the head mean cells/frame) with n_d clips and its own
--seed s_d; the simulator seeds clip i of that run with s_d * 100000 + i (lineage_sim_v3 rule), so every clip index of the
corpus has a fixed, recorded seed.  The runs are merged in density order into <root>/lineages/clip_%03d (+ <root>/
lineages_ctc/clip_%03d_{GT,ST}); meta.json gets the corpus clip name (original name, run and density are recorded under
'design').  Writes <root>/lineage_manifest.json (per clip: density, run, sim clip, seed; per run: simulator summary) and runs
verify_sim_v3.py on the merged set (detached; verify_lineages.json).  Refuses to overwrite a finished root (lineage_manifest.json present).
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time

N = os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace"))
PYC = os.environ.get("CELLPOSE_PYTHON", sys.executable)
S = f"{N}/scripts/mitosync"


def log(*a):
    print(time.strftime("[%F %T]"), "[lineages]", *a, flush=True)


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["PhC-C2DL-PSC", "Fluo-N2DL-HeLa"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--interval", type=int, required=True, choices=[1, 8])
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--densities", required=True, help="d:n:seed,... in merge order")
    ap.add_argument("--phase_lock", required=True, help="K* from dp3_decision.json, e.g. 1,2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ds = a.dataset
    root = os.path.realpath(a.root)
    assert root.startswith(f"{N}/data/mitosync/design/"), root
    if os.path.exists(f"{root}/lineage_manifest.json"):
        log(f"exists: {root}/lineage_manifest.json -- nothing to do"); return
    ch = json.load(open(f"{N}/eval/mitosync/m1_version_choice.json"))["choice"][ds]["interval1" if a.interval == 1 else "x8"]
    prog = f"{N}/{ch['programmer']}"; flags = ch["flags"].split()
    lin, lctc, sub = f"{root}/lineages", f"{root}/lineages_ctc", f"{root}/_sub"
    for d in (lin, lctc):
        if os.path.exists(d):
            shutil.rmtree(d)   # unfinished earlier attempt (no manifest): rebuild deterministically
    os.makedirs(lin); os.makedirs(lctc); os.makedirs(sub, exist_ok=True)
    runs, clips, k = [], {}, 0
    for j, spec in enumerate(a.densities.split(",")):
        d, n, s = spec.split(":"); d, n, s = float(d), int(n), int(s)
        out = f"{sub}/r{j}_d{d:g}"
        args = ["--dataset", ds, "--theta", f"{N}/eval/mitosync/theta_{ds}.json", "--root", f"{N}/data/mitosync/kin_head",
                "--cache", f"{N}/data/mitosync/lib_{ds}_head.pkl", "--num_clips", str(n), "--num_frames", str(a.num_frames),
                "--density_scale", f"{d:g}", "--event_budget", f"{a.kappa:g}", "--interval", str(a.interval), "--phase_lock", a.phase_lock,
                "--overlap_mode", "none", "--entries", "1", "--cond_version", "2", "--seed", str(s), "--workers", str(min(a.workers, n))] + flags
        if not os.path.exists(f"{out}/summary.json"):
            for x in (out, out + "_ctc"):
                if os.path.exists(x):
                    shutil.rmtree(x)
            log(f"run {j}: density {d:g} x head, {n} clips, seed {s}: {os.path.basename(prog)} {' '.join(flags)}")
            t0 = time.time()
            with open(f"{root}/lineages.log", "a") as f:
                f.write(time.strftime("\n=== %F %T ") + " ".join([PYC, prog] + args + ["--out", out]) + "\n"); f.flush()
                rc = subprocess.run(["nice", "-n", "10", PYC, prog] + args + ["--out", out], stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not os.path.exists(f"{out}/summary.json"):
                sys.exit(f"simulator failed (rc {rc}) for run {j}; see {root}/lineages.log")
            log(f"  done in {time.time() - t0:.0f}s")
        sm = json.load(open(f"{out}/summary.json"))
        runs.append(dict(run=j, density_scale=d, n_clips=n, seed=s, args=args, out=out, summary={x: sm.get(x) for x in (
            "generator", "mean_cells_per_frame", "divisions_total", "divisions_per_clip_mean", "div_per_1000_atrisk", "theta_div_per_1000",
            "entry_per_1000_atrisk", "exit_per_1000_atrisk", "touching_fraction", "lineage_violations", "phase_lock_violations", "one_cc_violations",
            "calibration", "theta_sha", "wall_seconds")}))
        for i in range(n):
            src = f"clip_{i:03d}"; dst = f"clip_{k:03d}"
            m = json.load(open(f"{out}/{src}/meta.json"))
            os.rename(f"{out}/{src}", f"{lin}/{dst}")
            for suf in ("_GT", "_ST"):
                os.rename(f"{out}_ctc/{src}{suf}", f"{lctc}/{dst}{suf}")
            m["design"] = dict(corpus=f"{ds}/{a.corpus}", original_clip=src, run=j, run_dir=out, density_scale=d, run_seed=s)
            m["clip"] = dst
            json.dump(m, open(f"{lin}/{dst}/meta.json", "w"), indent=1)
            clips[dst] = dict(run=j, original_clip=src, density_scale=d, run_seed=s, sim_seed=m.get("seed"), n_div=m.get("n_div"),
                              mean_cells_per_frame=m.get("mean_cells_per_frame"), ref_frame=(m.get("ref") or {}).get("frame"),
                              lineage_violations=m.get("lineage_violations"), phase_lock_violations=m.get("phase_lock_violations"))
            k += 1
    vj = f"{root}/verify_lineages.json"
    vl = f"{root}/verify_lineages.log"
    # verify_sim_v3.py is single-threaded (~20 min per 200 clips): run it detached so that certification can start;
    # its result (ERRORS count) lands in verify_lineages.json and is read by design_corpus_summary.py
    vp = subprocess.Popen(f"nice -n 10 {PYC} {S}/verify_sim_v3.py {lin} --json {vj} > {vl} 2>&1", shell=True, start_new_session=True,
                          stdin=subprocess.DEVNULL)
    man = dict(dataset=ds, corpus=a.corpus, created=time.strftime("%F %T %Z"), script=os.path.abspath(__file__),
               amendment="PREREGISTRATION_AMENDMENT_4.md D3", programmer=dict(file=prog, sha256=sha(prog), version=ch["version"], flags=ch["flags"],
                                                                                 choice_file=f"{N}/eval/mitosync/m1_version_choice.json"),
               settings=dict(interval=a.interval, event_budget=a.kappa, phase_lock=a.phase_lock, num_frames=a.num_frames, cond_version=2, entries=1,
                             overlap_mode="none", density_mixture=a.densities, theta=f"{N}/eval/mitosync/theta_{ds}.json",
                             theta_sha256=sha(f"{N}/eval/mitosync/theta_{ds}.json"), library=f"{N}/data/mitosync/lib_{ds}_head.pkl",
                             root=f"{N}/data/mitosync/kin_head"),
               n_clips=k, runs=runs, clips=clips, lineages=lin, twin=lctc, verify=dict(json=vj, log=vl, pid=vp.pid, mode="async (detached verify_sim_v3.py)"),
               totals=dict(divisions=int(sum(c["n_div"] or 0 for c in clips.values())),
                           lineage_violations=int(sum(c["lineage_violations"] or 0 for c in clips.values())),
                           phase_lock_violations=int(sum(c["phase_lock_violations"] or 0 for c in clips.values()))))
    json.dump(man, open(f"{root}/lineage_manifest.json", "w"), indent=1)
    for r in runs:   # the clip dirs were moved out; keep only the run summaries (in the manifest)
        for x in (r["out"], r["out"] + "_ctc"):
            if os.path.exists(x):
                shutil.copy(f"{r['out']}/summary.json", f"{root}/run{r['run']}_summary.json") if x == r["out"] and os.path.exists(f"{r['out']}/summary.json") else None
                shutil.rmtree(x)
    shutil.rmtree(sub, ignore_errors=True)
    log(f"{ds}/{a.corpus}: {k} clips, {man['totals']['divisions']} programmed divisions, violations lineage {man['totals']['lineage_violations']} "
        f"lock {man['totals']['phase_lock_violations']}; verify_sim_v3 running detached (pid {vp.pid}) -> {vj}; manifest {root}/lineage_manifest.json")


if __name__ == "__main__":
    main()
