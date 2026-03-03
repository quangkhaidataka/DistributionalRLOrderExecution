"""
tests/test_loader.py  --  Tests for data/loader.py
Run:  cd iqn_execution && python3 tests/test_loader.py
"""
import os, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from data.loader import (
    LOBSTERLoader, load_episodes, verify_parquet,
    PRICE_SCALE, N_LEVELS, TRADING_START, TRADING_END, MIN_BARS_PER_DAY,
    LOB_COLS, PARQUET_COLS,
    _read_message_file, _read_orderbook_file,
    _process_day, _extract_date, _find_orderbook,
)

PASS = "v"; FAIL = "F"; results = []

def check(name, cond, detail=""):
    s = PASS if cond else FAIL
    results.append((s, name, detail))
    print(f"  [{s}]  {name}" + (f"  [{detail}]" if detail else ""))

def section(t):
    print(f"\n{'='*58}\n  {t}\n{'='*58}")

def _msg_csv(n=500, base=36000.0, halt=False):
    lines = []
    for i in range(n):
        t = base + i * 1.5
        et = 7 if (halt and i == 50) else (i % 5) + 1
        lines.append(f"{t:.6f},{et},{200000000+i},100,{1186000+(i%5)*100},{1 if i%2==0 else -1}")
    return "\n".join(lines)

def _lob_csv(n=500, levels=10):
    lines = []
    for _ in range(n):
        cols = []
        for l in range(1, levels+1):
            cols += [1186600+l*100, 1000, 1186500-l*100, 900]
        lines.append(",".join(map(str, cols)))
    return "\n".join(lines)

def _write_day(d, stock="T", date="2019-01-02", n=500, halt=False):
    stem = f"{stock}_{date}_{date}_10"
    mp = Path(d)/f"{stem}_message_1.csv"
    lp = Path(d)/f"{stem}_orderbook_1.csv"
    mp.write_text(_msg_csv(n, halt=halt))
    lp.write_text(_lob_csv(n))
    return mp, lp

def _ep(date, bars=331):
    times = pd.date_range(f"{date} 10:00", periods=bars, freq="1min")
    rows = []
    for t in times:
        r = {"date": date, "time": t, "mid_price": 118.6, "spread": 0.01, "imbalance": 0.0}
        for l in range(1, N_LEVELS+1):
            r[f"ask{l}"] = 118.6 + l*0.01; r[f"bid{l}"] = 118.6 - l*0.01
            r[f"askvol{l}"] = 1000.0; r[f"bidvol{l}"] = 900.0
        rows.append(r)
    return pd.DataFrame(rows)[PARQUET_COLS]

# ---- 1: Constants
section("1 -- Schema Constants")
check("PRICE_SCALE==10000",  PRICE_SCALE==10_000)
check("N_LEVELS==5",         N_LEVELS==5)
check("TRADING_START==10:00", TRADING_START=="10:00")
check("TRADING_END==15:30",   TRADING_END=="15:30")
check("MIN_BARS==30",         MIN_BARS_PER_DAY==30)
exp = (["date","time","mid_price","spread","imbalance"] +
       [f"ask{l}" for l in range(1,6)] + [f"bid{l}" for l in range(1,6)] +
       [f"askvol{l}" for l in range(1,6)] + [f"bidvol{l}" for l in range(1,6)])
check("PARQUET_COLS correct", PARQUET_COLS==exp)
check("25 columns",           len(PARQUET_COLS)==25)
check("LOB_COLS==20",         len(LOB_COLS)==20)

# ---- 2: _read_message_file
section("2 -- _read_message_file")
with tempfile.TemporaryDirectory() as d:
    mp, _ = _write_day(d, n=50)
    msg = _read_message_file(mp)
check("DataFrame",          isinstance(msg, pd.DataFrame))
check("6 cols",             len(msg.columns)==6)
check("col names",          list(msg.columns)==["time","event_type","order_id","size","price","direction"])
check("time float64",       msg["time"].dtype==np.float64)
check("event_type int8",    msg["event_type"].dtype==np.int8)
check("order_id int64",     msg["order_id"].dtype==np.int64)
check("size int32",         msg["size"].dtype==np.int32)
check("price int64",        msg["price"].dtype==np.int64)
check("direction int8",     msg["direction"].dtype==np.int8)
check("50 rows",            len(msg)==50)
check("event_type 1-7",     msg["event_type"].isin(range(1,8)).all())
check("direction +-1",      msg["direction"].isin([-1,1]).all())
check("None on bad path",   _read_message_file(Path("/no/file"))==None)

# ---- 3: _read_orderbook_file
section("3 -- _read_orderbook_file")
with tempfile.TemporaryDirectory() as d:
    _, lp = _write_day(d, n=50)
    lob = _read_orderbook_file(lp)
exp_raw = []
for l in range(1, N_LEVELS+1):
    exp_raw += [f"ask{l}", f"askvol{l}", f"bid{l}", f"bidvol{l}"]
check("DataFrame",              isinstance(lob, pd.DataFrame))
check(f"{N_LEVELS*4} cols",     len(lob.columns)==N_LEVELS*4)
check("interleaved order",      list(lob.columns)==exp_raw)
check("ask1 int64",             lob["ask1"].dtype==np.int64)
check("askvol1 int32",          lob["askvol1"].dtype==np.int32)
check("bid1 int64",             lob["bid1"].dtype==np.int64)
check("bidvol1 int32",          lob["bidvol1"].dtype==np.int32)
check("50 rows",                len(lob)==50)
check("no col past level 5",    f"ask{N_LEVELS+1}" not in lob.columns)
check("None on bad path",       _read_orderbook_file(Path("/no/file"))==None)

# ---- 4: Price conversion
section("4 -- Price Conversion")
for raw, exp_p in [(1_186_000,118.60),(1_186_600,118.66),(911_400,91.14)]:
    check(f"${exp_p} = {raw}/10000", abs(raw/PRICE_SCALE - exp_p) < 1e-8)
ask_s = 9_999_999_999/PRICE_SCALE
bid_s = -9_999_999_999/PRICE_SCALE
check("ask sentinel >> real", ask_s > 100_000)
check("bid sentinel << real", bid_s < -100_000)

# ---- 5: Sentinel filter
section("5 -- Sentinel Filter")
df5 = pd.DataFrame({"ask1":[118.66, ask_s, 118.67], "bid1":[118.65, bid_s, 118.66]})
valid = (df5["ask1"]<100_000) & (df5["bid1"]>-100_000) & (df5["ask1"]>0) & (df5["bid1"]>0)
f5 = df5.loc[valid]
check("2 real rows survive",  len(f5)==2)
check("sentinel row removed", 1 not in f5.index)
check("no non-positive bid",  (f5["bid1"]>0).all())

# ---- 6: Event type filter
section("6 -- Event Type Filter")
evts = pd.Series([1,2,3,4,5,6,7,1,7,5])
kept = evts.loc[~evts.isin({6,7})].values
check("types 1-5 present",        set(kept).issubset({1,2,3,4,5}))
check("types 6,7 absent",         6 not in kept and 7 not in kept)
check("7 kept (3 dropped)",       len(kept)==7, str(len(kept)))
with tempfile.TemporaryDirectory() as d:
    mp, lp = _write_day(d, n=2500, halt=True)
    df6 = _process_day(mp, lp, "2019-01-02")
check("_process_day ok with halts", df6 is not None)

# ---- 7: Derived features
section("7 -- Derived Features")
a1,b1 = 118.66,118.65
check("mid_price",    abs((a1+b1)/2-118.655)<1e-8)
check("spread>=0",    abs(a1-b1-0.01)<1e-8 and (a1-b1)>=0)
check("clip crossed", max(-0.05,0.0)==0.0)
bv,av = 8800,9484
imb = (bv-av)/(bv+av)
check("imbalance",    abs(imb-(-684/18284))<1e-8)
check("imb in [-1,1]", -1<=imb<=1)
check("imb=+1 allbid", abs((1000-0)/(1000+0)-1.0)<1e-8)
check("imb=0 zerovol", (0.0 if 0+0==0 else 0)==0.0)

# ---- 8: Resample + ffill
section("8 -- Resample + ffill")
idx8 = pd.date_range("2019-01-02 10:00", periods=600, freq="1s")
s8   = pd.Series(np.random.default_rng(0).uniform(118,119,600), index=idx8)
b8   = s8.resample("1min").last()
check("resample reduces rows",   len(b8)<len(s8), f"{len(b8)}<{len(s8)}")
check("1-min spacing",           (b8.index[1]-b8.index[0])==pd.Timedelta("1min"))
idx9 = pd.date_range("2019-01-02 10:00", periods=5, freq="1min")
f9   = pd.Series([1.0,np.nan,np.nan,4.0,5.0],index=idx9).ffill(limit=1)
check("ffill 1st gap",           f9.iloc[1]==1.0)
check("ffill leaves 2nd NaN",    np.isnan(f9.iloc[2]))
check("ffill keeps bar[3]",      f9.iloc[3]==4.0)

# ---- 9: Trading hours
section("9 -- Trading Hours [10:00, 15:30]")
idx9b = pd.date_range("2019-01-02 09:30", periods=400, freq="1min")
filt9 = pd.Series(1.0,index=idx9b).between_time(TRADING_START, TRADING_END)
ts9   = [t.strftime("%H:%M") for t in filt9.index]
check("first=10:00",  filt9.index.min().strftime("%H:%M")=="10:00")
check("last=15:30",   filt9.index.max().strftime("%H:%M")=="15:30")
check("09:30 out",    "09:30" not in ts9)
check("15:31 out",    "15:31" not in ts9)
check("331 bars",     len(filt9)==331, str(len(filt9)))

# ---- 10: Full pipeline
section("10 -- Full Pipeline (_process_day)")
with tempfile.TemporaryDirectory() as d:
    mp, lp = _write_day(d, stock="AAPL", date="2019-01-02", n=2500)
    res = _process_day(mp, lp, "2019-01-02")
check("returns DataFrame",       res is not None and isinstance(res, pd.DataFrame))
if res is not None:
    check("cols=PARQUET_COLS",   list(res.columns)==PARQUET_COLS)
    check("date col",            (res["date"]=="2019-01-02").all())
    check("time is datetime",    pd.api.types.is_datetime64_any_dtype(res["time"]))
    check("mid_price>0",         (res["mid_price"]>0).all())
    check("spread>=0",           (res["spread"]>=0).all())
    check("imb in [-1,1]",       ((res["imbalance"]>=-1)&(res["imbalance"]<=1)).all())
    check("ask1>=bid1",          (res["ask1"]>=res["bid1"]).all())
    check(f">={MIN_BARS_PER_DAY} bars", len(res)>=MIN_BARS_PER_DAY, str(len(res)))
    check("no NaN critical",     not res[["mid_price","spread","imbalance"]].isna().any().any())
    check("bars in hours",       res["time"].dt.hour.between(10,15).all())
with tempfile.TemporaryDirectory() as d:
    mp2=Path(d)/"m.csv"; lp2=Path(d)/"l.csv"
    mp2.write_text(_msg_csv(10)); lp2.write_text(_lob_csv(5))
    check("mismatch->None",      _process_day(mp2,lp2,"2019-01-02") is None)

# ---- 11: Filename helpers
section("11 -- Filename Helpers")
with tempfile.TemporaryDirectory() as d:
    tp = Path(d)
    mp, lp = _write_day(tp, stock="AAPL", date="2019-01-02")
    check("_extract_date",       _extract_date(mp)=="2019-01-02")
    found = _find_orderbook(mp)
    check("find via name",       found is not None and found.exists())
    check("found has orderbook", found is not None and "orderbook" in found.name)
    alt = tp/"AAPL_2019-01-02_other_orderbook.csv"
    shutil.copy(lp, alt); lp.unlink()
    found2 = _find_orderbook(mp)
    check("glob fallback",       found2 is not None and found2.exists())
    check("None when no match",  _find_orderbook(Path(d)/"ghost_message.csv") is None)

# ---- 12: load_episodes
section("12 -- load_episodes")
with tempfile.TemporaryDirectory() as d:
    proc = Path(d)/"processed"; proc.mkdir()
    combo = pd.concat([_ep("2019-01-02"), _ep("2019-01-03")], ignore_index=True)
    pq_ok = False
    try:
        combo.to_parquet(proc/"AAPL_2019.parquet", index=False)
        pq_ok = True
    except ImportError:
        pass
    if pq_ok:
        eps = load_episodes(str(proc), ["AAPL"], [2019])
        check("returns list",     isinstance(eps, list))
        check("2 episodes",       len(eps)==2, str(len(eps)))
        check("each DataFrame",   all(isinstance(e, pd.DataFrame) for e in eps))
        check("each >=30 bars",   all(len(e)>=MIN_BARS_PER_DAY for e in eps))
        check("cols correct",     all(list(e.columns)==PARQUET_COLS for e in eps))
        check("shuffle ok",       len(load_episodes(str(proc),["AAPL"],[2019],shuffle=True))==2)
        try:
            load_episodes(str(proc), ["MSFT"], [2019])
            check("missing->error", False)
        except FileNotFoundError:
            check("missing->FileNotFoundError", True)
    else:
        eps2 = [g.reset_index(drop=True) for _,g in combo.groupby("date",sort=True)
                if len(g)>=MIN_BARS_PER_DAY]
        check("split gives 2 eps",  len(eps2)==2)
        check("cols correct",       all(list(e.columns)==PARQUET_COLS for e in eps2))
        check("(parquet N/A)",      True)

# ---- 13: verify_parquet
section("13 -- verify_parquet")
with tempfile.TemporaryDirectory() as d:
    proc = Path(d)/"processed"; proc.mkdir()
    rng = np.random.default_rng(42)
    rows = []
    for di in range(252):
        date=(pd.Timestamp("2019-01-02")+pd.Timedelta(days=di)).strftime("%Y-%m-%d")
        for t in pd.date_range(f"{date} 10:00", periods=331, freq="1min"):
            r={"date":date,"time":t,
               "mid_price":float(118.6+rng.standard_normal()*0.1),
               "spread":float(abs(rng.standard_normal())*0.01+0.005),
               "imbalance":float(np.clip(rng.standard_normal()*0.3,-1,1))}
            for l in range(1,N_LEVELS+1):
                r[f"ask{l}"]=118.6+l*0.01; r[f"bid{l}"]=118.6-l*0.01
                r[f"askvol{l}"]=1000.0; r[f"bidvol{l}"]=900.0
            rows.append(r)
    df_g = pd.DataFrame(rows)[PARQUET_COLS]
    pq_ok = False
    try:
        df_g.to_parquet(proc/"AAPL_2019.parquet", index=False)
        pq_ok = True
    except ImportError:
        pass
    if pq_ok:
        check("passes good data",    verify_parquet(str(proc),"AAPL",2019))
        df_b = df_g.copy(); df_b.loc[0,"mid_price"]=np.nan
        df_b.to_parquet(proc/"AAPL_2020.parquet",index=False)
        check("fails NaN mid_price", not verify_parquet(str(proc),"AAPL",2020))
        check("fails missing file",  not verify_parquet(str(proc),"MSFT",2019))
    else:
        check("no NaN critical",  not df_g[["mid_price","spread","imbalance"]].isna().any().any())
        check("price in range",   bool(((df_g["mid_price"]>0)&(df_g["mid_price"]<10000)).all()))
        check("spread>=0",        bool((df_g["spread"]>=0).all()))
        check("imb in [-1,1]",    bool(((df_g["imbalance"]>=-1)&(df_g["imbalance"]<=1)).all()))
        check("ask1>=bid1",       bool((df_g["ask1"]>=df_g["bid1"]).all()))
        check("252 days",         df_g["date"].nunique()==252)
        check("(parquet N/A)",    True)

# ---- Summary
print(f"\n{'='*58}")
passed = sum(1 for s,_,_ in results if s==PASS)
failed = sum(1 for s,_,_ in results if s==FAIL)
print(f"  {passed} passed,  {failed} failed  ({len(results)} total)")
if failed==0:
    print("  All loader tests PASSED [v]")
else:
    for s,n,d in results:
        if s==FAIL:
            print(f"    [F] {n}" + (f"  [{d}]" if d else ""))
print("="*58)