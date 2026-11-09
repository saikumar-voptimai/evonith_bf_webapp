#!/usr/bin/env python3
r"""BF-02 STRATA 1.5: scenario runs for the decks on the Open-Meteo plan moisture (temporary runner, 25 Oct 2026).

Runs 42 engine jobs on this PC (the 17 what-if days in STRATA mode, the pellet and nut-coke size sweeps) in parallel;
the cloud session runs the rest. Plans use the Open-Meteo blast moisture of the plan day 15 Oct 2026 (15.2 g/Nm3, from
the shipped Open-Meteo file) and the shipped calibration re-solved at its window's Open-Meteo mean (18.45 g/Nm3).

The code is taken from GitHub, branch feature/262_blend_omega_uncertainty, commit ba8642b, in a separate git worktree:
the working copy of the repository is not changed.

Usage, from the root of the evonith_bf_webapp repository (the environment created by `uv sync`):

    uv run python run_strata16_jobs.py                # all cores but one
    uv run python run_strata16_jobs.py --workers 4    # fixed number of processes
    uv run python run_strata16_jobs.py --list         # show the jobs and stop
    .venv\Scripts\python.exe run_strata16_jobs.py --workers 4   # Windows, venv as it is

Interrupted runs resume: finished jobs are kept in ./strata16_jobs_out and skipped on the next start.
Output: ./strata16_jobs_<date-time>.zip (upload this file to the chat).
"""
import os
import sys

# one BLAS thread per process (the jobs run in parallel), UTF-8 file handling in the worker processes, no live weather
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
os.environ['PYTHONUTF8'] = '1'
os.environ['COKE_ENERGY_WEATHER_LIVE'] = '0'
os.environ['COKE_ENERGY_TODAY'] = '2026-10-15'      # plan day: its Open-Meteo blast moisture (15.2 g/Nm3) is in the shipped file
if os.environ.get('STRATA15_SRC') and os.environ['STRATA15_SRC'] not in sys.path:
    sys.path.insert(0, os.environ['STRATA15_SRC'])

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402

COMMIT = 'ba8642b'
BRANCH = 'feature/262_blend_omega_uncertainty'
HERE = os.path.dirname(os.path.abspath(__file__))

# what-if days (index = position in the full list of 17; days 0-7 are already computed)
DAYS = {
    0: ('Normal day', {}, {}),
    1: ('Blast ≤ 105,000 Nm³/h', {'V_max': 105000.0}, {}),
    2: ('Blast ≤ 100,000 Nm³/h', {'V_max': 100000.0}, {}),
    3: ('Oxygen ≤ 5,000 Nm³/h', {'O2_flow_max': 5000.0}, {}),
    4: ('Oxygen ≤ 4,000 Nm³/h', {'O2_flow_max': 4000.0}, {}),
    5: ('Hot blast ≤ 1,150 °C', {'HBT_max': 1150.0}, {'HBT': 1150.0}),
    6: ('Hot blast ≤ 1,100 °C', {'HBT_max': 1100.0}, {'HBT': 1100.0}),
    7: ('Charging ≤ 6.2 charges/h', {'charges_per_h': 6.2}, {}),
    8: ('Hot-blast pressure ≤ 2.70 bar g', {'Pb_max': 2.70}, {}),
    9: ('Slag cap 320 kg/tHM', {'slag_max': 320.0}, {}),
    10: ('27.0 t per charge', {'charge_t': 27.0}, {}),
    11: ('PCI 150 kg/tHM', {}, {'pci_plant': 150.0}),
    12: ('PCI 220 kg/tHM', {}, {'pci_plant': 220.0}),
    13: ('Nut coke 50 kg/tHM', {}, {'nut_plant': 50.0}),
    14: ('HMT 1,490 °C, Si 0.60 %', {}, {'HMT': 1490.0, 'hm_Si': 0.60}),
    15: ('Dry season, 7 g/Nm³', {}, {'humidity_g': 7.0}),
    16: ('Wet season, 21 g/Nm³', {}, {'humidity_g': 21.0}),
}
DAYT = [('Normal day', {}), ('Pb ≤ 2.70 bar g', {'Pb_max': 2.70}), ('Blast ≤ 105,000 Nm³/h', {'V_max': 105000.0})]
O2DAYS = [('Oxygen ≤ 5,000 Nm³/h', 5000.0, 3.6), ('Oxygen ≤ 4,000 Nm³/h', 4000.0, 2.9)]


def job_ids():
    """Longest jobs first. This runner does the STRATA-mode what-if days and the pellet and nut-coke sweeps; the cloud
    session does the rest (fixed-mode days, oxygen days, Gomti, reference plans, forced blends)."""
    ids = [f'day{i:02d}' for i in DAYS]
    ids += [f'pel{p:02d}_{k}' for p in (5, 10, 15, 20, 25) for k in range(3)]
    ids += [f'nut{d + 4:d}_{k}' for d in (-4, -2, 0, 2, 4) for k in range(2)]
    return ids


# ---------------------------------------------------------------------------------------------- engine set-up
_CACHE = {}


def _setup():
    if _CACHE:
        return _CACHE
    import copy
    import yaml
    from utils.burden_engine.engine import Engine, Settings, Limits, REFERENCE_INPUTS_PATH, summary
    from utils.burden_engine.scenarios import run_cheapest_level, run_max
    src = os.environ['STRATA15_SRC']
    with open(os.path.join(src, 'config', 'burden_optimiser.yml'), encoding='utf-8') as f:
        pb = yaml.safe_load(f)['burden_optimiser']['prices']
    with open(REFERENCE_INPUTS_PATH, encoding='utf-8') as f:
        uat = json.load(f)
    inp = copy.deepcopy(uat)   # 16 Oct 2026 UAT yard at the plant prices of 23 Oct 2026 + slag credit
    for m in inp['materials']:
        m['price'] = float(pb['materials'].get(m['id'], m['price']))
    for x in inp['fluxes']:
        x['price'] = float(pb['fluxes'].get(x['id'], x['price']))
    inp['fuel'].update({k: float(v) for k, v in pb['fuel'].items()})
    inp['fuel']['slag_price'] = float(pb['slag_price_rs_per_t'])
    mats = {m['id']: m for m in inp['materials']}
    uatm = [m['id'] for m in inp['materials'] if m.get('selected_in_uat', True)]
    _CACHE.update(Engine=Engine, Settings=Settings, Limits=Limits, summary=summary, run_max=run_max,
                  run_cheapest_level=run_cheapest_level, inp=inp, UATM=uatm,
                  EXT=uatm + ['nmdc_donimalai', 'gomti_clo', 'lloyds_oversize_clo'],
                  B0={i: (mats[i]['lo'], mats[i]['hi']) for i in mats})
    return _CACHE


NAMES = {'sinter_sp_02': 'Sinter', 'geomin_clo': 'Geomin', 'lloyds_clo': 'Lloyds CLO', 'nmdc_rom': 'NMDC ROM',
         'lloyds_pellet': 'Pellet', 'nmdc_donimalai': 'Donimalai', 'gomti_clo': 'Gomti', 'lloyds_oversize_clo': 'Lloyds oversize',
         'limestone': 'Limestone', 'dolomite': 'Dolomite', 'quartz': 'Quartzite'}


def brief(eng, r):
    if r is None:
        return dict(ok=False)
    d = _CACHE['summary'](r, eng)
    om = r.get('omega') or {}
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith(('kg_', 'sh_'))}
    out['ok'] = True
    out['shares'] = {k[3:]: round(v, 2) for k, v in d.items() if k.startswith('sh_') and v > 0.05}
    out['flux'] = {k[3:]: round(v, 2) for k, v in d.items() if k.startswith('kg_') and v > 0.05}
    out['tier'] = om.get('certainty', 'Held')
    out['investigate'] = om.get('investigate')
    out['outside'] = om.get('outside')
    return out


def blend_txt(sh):
    return ' / '.join(f"{NAMES.get(k, k)} {v:.1f}" for k, v in sorted(sh.items(), key=lambda kv: -kv[1]))


def _err(ex):
    return dict(ok=False, err=str(ex)[:200])


# ---------------------------------------------------------------------------------------------- jobs
def _day(lch, sch, mode='blend', o2_tag=None):
    from dataclasses import replace
    C = _setup()
    eng = C['Engine'](C['inp'])
    s = replace(C['Settings'](omega_mode=mode), **sch)
    if o2_tag is not None:
        s = replace(s, O2_per_air=float(eng.hy.o2_per_air_from_tag(o2_tag)))
    L = replace(C['Limits'](), **lch)
    row = {}
    try:
        row['step2'] = brief(eng, eng.min_cost(prod_t_day=2270.0, s=s, L=L, fuel=True))
    except Exception as ex:  # noqa: BLE001
        row['step2'] = _err(ex)
    try:
        row['max'] = brief(eng, C['run_max'](eng, s, L)['result'])
    except Exception as ex:  # noqa: BLE001
        row['max'] = _err(ex)
    try:
        ch = (C['run_cheapest_level'](eng, s, L, search_o2=True, n_o2=9, P_lo=84.0, step=1.0) if mode == 'fixed' else
              C['run_cheapest_level'](eng, s, L, search_o2=False, P_lo=84.0, step=2.0))
        row['cheapest'] = brief(eng, ch['best'])
        fr = ch['frontier']
        row['frontier'] = fr.to_dict('records') if hasattr(fr, 'to_dict') else []
    except Exception as ex:  # noqa: BLE001
        row['cheapest'] = _err(ex)
    return row


def _max(mats, bounds, lch, **sk):
    from dataclasses import replace
    C = _setup()
    e = C['Engine'](C['inp'], materials=mats, bounds={k: bounds[k] for k in mats}) if mats else C['Engine'](C['inp'])
    try:
        return brief(e, C['run_max'](e, C['Settings'](omega_mode='blend', **sk), replace(C['Limits'](), **lch))['result'])
    except Exception as ex:  # noqa: BLE001
        return _err(ex)


def run_job(jid):
    from dataclasses import replace
    C = _setup()
    if jid.startswith('day'):
        name, lch, sch = DAYS[int(jid[3:])]
        return _day(lch, sch)
    if jid.startswith('o2'):
        i, mode = int(jid[2]), jid.split('_')[1]
        _, cap, tag = O2DAYS[i]
        return _day({'O2_flow_max': cap}, {}, mode=mode, o2_tag=tag)
    if jid.startswith('pel'):
        p, k = int(jid[3:5]), int(jid[-1])
        b = dict(C['B0']); b['lloyds_pellet'] = (p - 0.02, p + 0.02)
        return _max(C['UATM'], b, DAYT[k][1])
    if jid.startswith('gom'):
        g, k = int(jid[3:5]), int(jid[-1])
        b = dict(C['B0']); b['gomti_clo'] = (max(0, g - 0.02), g + 0.02)
        return _max(C['EXT'], b, DAYT[k][1])
    if jid.startswith('nut'):
        d, k = int(jid[3]) - 4, int(jid[-1])
        return _max(None, None, DAYT[k][1], nut_size_delta_mm=float(d))
    if jid == 'b105':
        eng = C['Engine'](C['inp'])
        s = replace(C['Settings'](omega_mode='blend'), O2_per_air=float(eng.hy.o2_per_air_from_tag(5.2354)))
        ch = C['run_cheapest_level'](eng, s, replace(C['Limits'](), V_max=105000.0), search_o2=False, P_lo=84.0, step=1.0)
        return dict(cheapest=brief(eng, ch['best']))
    if jid.startswith('gcost'):
        g = int(jid[5:])
        mats = C['UATM'] + ['gomti_clo']
        b = {i: C['B0'][i] for i in mats}; b['gomti_clo'] = (max(0, g - 0.02), g + 0.02)
        row = dict(gomti=g)
        for mode in ('fixed', 'blend'):
            e = C['Engine'](C['inp'], materials=mats, bounds=b)
            try:
                r = e._try(C['Settings'](omega_mode=mode), 2270 / 24, C['Limits'](), True)
                row[mode] = brief(e, r)
                if r is not None:
                    row[mode]['blend_txt'] = blend_txt(row[mode]['shares'])
            except Exception as ex:  # noqa: BLE001
                row[mode] = _err(ex)
        return row
    raise ValueError(jid)


def worker(args):
    jid, out_dir = args
    t0 = time.time()
    try:
        res = run_job(jid)
    except Exception:  # noqa: BLE001
        with open(os.path.join(out_dir, jid + '.err'), 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        return jid, False, time.time() - t0
    tmp = os.path.join(out_dir, jid + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=str)
    os.replace(tmp, os.path.join(out_dir, jid + '.json'))
    return jid, True, time.time() - t0


# ---------------------------------------------------------------------------------------------- repository
def git(repo, *a, check=True):
    return subprocess.run(['git', '-C', repo, *a], check=check, capture_output=True, text=True)


def prepare_source(repo):
    """Worktree of commit COMMIT next to the repository; returns its src/ folder."""
    repo = os.path.abspath(repo)
    if git(repo, 'rev-parse', '--git-dir', check=False).returncode != 0:
        sys.exit(f'Not a git repository: {repo}. Run from the evonith_bf_webapp root or pass --repo.')
    if git(repo, 'cat-file', '-e', COMMIT + '^{commit}', check=False).returncode != 0:
        print(f'Fetching {BRANCH} from origin ...', flush=True)
        r = git(repo, 'fetch', 'origin', BRANCH, check=False)
        if r.returncode != 0 or git(repo, 'cat-file', '-e', COMMIT + '^{commit}', check=False).returncode != 0:
            sys.exit(f'Commit {COMMIT} not found after fetching {BRANCH}:\n{r.stderr}')
    wt = os.path.join(os.path.dirname(repo), 'strata16_worktree')
    if not os.path.isdir(os.path.join(wt, 'src')):
        git(repo, 'worktree', 'prune', check=False)
        r = git(repo, 'worktree', 'add', '--detach', wt, COMMIT, check=False)
        if r.returncode != 0:
            sys.exit('git worktree add failed:\n' + r.stderr)
    head = git(wt, 'rev-parse', '--short', 'HEAD').stdout.strip()
    if not COMMIT.startswith(head[:7]):
        sys.exit(f'Worktree {wt} is at {head}, expected {COMMIT}. Delete the folder and run again.')
    return repo, wt, os.path.join(wt, 'src')


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.', help='evonith_bf_webapp repository (default: current folder)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'strata16_jobs_out'))
    ap.add_argument('--list', action='store_true', help='list the jobs and stop')
    ap.add_argument('--only', default='', help='comma-separated job ids (default: all)')
    ap.add_argument('--part', default='all', choices=('all', 'laptop', 'cloud'),
                    help='laptop = oxygen days, pellet and nut-coke sweeps, blast-limited day, Gomti cost; cloud = the rest')
    ap.add_argument('--keep-worktree', action='store_true', help='do not remove the git worktree at the end')
    a = ap.parse_args()
    ids = [j for j in job_ids() if not a.only or j in a.only.split(',')]
    if a.part != 'all':
        lap = list(ids)
        ids = lap if a.part == 'laptop' else [j for j in ids if j not in lap]
    if a.list:
        print('\n'.join(ids)); print(len(ids), 'jobs'); return
    repo, wt, src = prepare_source(a.repo)
    os.environ['STRATA15_SRC'] = src
    sys.path.insert(0, src)
    os.makedirs(a.out, exist_ok=True)
    todo = [j for j in ids if not os.path.exists(os.path.join(a.out, j + '.json'))]
    print(f'STRATA 1.5 scenario jobs: commit {COMMIT}, {len(ids) - len(todo)} done, {len(todo)} to run, '
          f'{a.workers} processes, output {a.out}', flush=True)
    t0, n_ok, n_bad = time.time(), 0, 0
    if todo:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(a.workers, len(todo)), maxtasksperchild=1) as pool:
            for k, (jid, ok, dt) in enumerate(pool.imap_unordered(worker, [(j, a.out) for j in todo]), start=1):
                n_ok += ok; n_bad += (not ok)
                el = time.time() - t0
                eta = el / k * (len(todo) - k)
                print(f'[{k:2d}/{len(todo)}] {jid:10s} {"ok" if ok else "FAILED (see .err file)"}  {dt / 60:5.1f} min'
                      f'   elapsed {el / 60:5.1f} min, about {eta / 60:5.0f} min left', flush=True)
    done = [j for j in ids if os.path.exists(os.path.join(a.out, j + '.json'))]
    import numpy, scipy  # noqa: E401
    manifest = dict(commit=COMMIT, branch=BRANCH, jobs_done=done, jobs_missing=[j for j in ids if j not in done],
                    workers=a.workers, minutes=round((time.time() - t0) / 60, 1), python=sys.version.split()[0],
                    numpy=numpy.__version__, scipy=scipy.__version__, platform=platform.platform(),
                    finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(a.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
    zpath = os.path.join(os.getcwd(), time.strftime('strata16_jobs_%Y%m%d_%H%M.zip'))
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(a.out)):
            if fn.endswith(('.json', '.err')):
                z.write(os.path.join(a.out, fn), arcname='jobs/' + fn)
    if not a.keep_worktree and not manifest['jobs_missing']:
        git(repo, 'worktree', 'remove', '--force', wt, check=False)
        shutil.rmtree(wt, ignore_errors=True)
    print(f'\n{len(done)}/{len(ids)} jobs finished ({n_bad} failed in this run). Upload this file to the chat:\n  {zpath}')
    if manifest['jobs_missing']:
        print('Missing:', ', '.join(manifest['jobs_missing']), '- run the script again to retry them.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
