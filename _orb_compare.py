import warnings; warnings.filterwarnings('ignore')
import io, contextlib
from today_backtest import run_multi_today, prepare_sym_data
from collections import defaultdict

setups = {
    'LLY':{'support':1094.0,'resistance':1105.0},'AVGO':{'support':452.0,'resistance':457.0},
    'ARM':{'support':385.0,'resistance':400.0},'NET':{'support':243.67,'resistance':248.74},
    'NVDA':{'support':215.0,'resistance':217.8},'CRM':{'support':199.0,'resistance':205.23},
    'ZS':{'support':142.5,'resistance':146.6},'CRWV':{'support':114.0,'resistance':117.0},
}
days = ['2026-05-26','2026-05-27','2026-05-28','2026-05-29','2026-06-01']
pf_d = prepare_sym_data(list(setups), interval='1m', period='7d',
                        entry_freq='5min', quiet=True)

def run(d, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t = run_multi_today(setups, date_str=d, interval='1m', period='7d',
                            entry_freq='5min', stale_secs=360, sizing='live',
                            prefetched=pf_d, quiet=True, **kw)
    return t

configs = [
    ('Rip (baseline)',  {}),
    ('ORB-10 only',     {'orb_entry':True, 'use_lost_dir':False}),
    ('Rip + ORB-10',    {'orb_entry':True}),
]
print('='*62)
print('  5-DAY: Rip baseline vs ORB-10 vs Combined (8 stocks)')
print('='*62)
print('  {:18} {:>6} {:>5} {:>10} {:>5} {:>8}'.format('Config','Trades','WinR','Net','PF','DayW/L'))
print('  '+'-'*58)
results = {}
for label, kw in configs:
    all_t = []; day_nets = defaultdict(float)
    for d in days:
        t = run(d, **kw)
        for x in t: day_nets[d] += x['pnl']
        all_t.extend(t)
    wins   = [x for x in all_t if x['pnl']>0]
    losses = [x for x in all_t if x['pnl']<=0]
    net    = sum(x['pnl'] for x in all_t)
    pf_val = sum(x['pnl'] for x in wins)/abs(sum(x['pnl'] for x in losses)) if losses else 999
    wr     = 100*len(wins)//len(all_t) if all_t else 0
    dw     = sum(1 for v in day_nets.values() if v>0)
    dl     = sum(1 for v in day_nets.values() if v<=0)
    results[label] = day_nets
    print('  {:18} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L'.format(
        label, len(all_t), wr, net, pf_val, dw, dl))

print()
print('  Per-day breakdown:')
hdrs = [l[:14] for l,_ in configs]
print('  {:12}  {:>14}  {:>14}  {:>14}'.format('Day', *hdrs))
print('  '+'-'*60)
for d in days:
    vals = ['${:+,.0f}'.format(results[l][d]) for l,_ in configs]
    print('  {:12}  {:>14}  {:>14}  {:>14}'.format(d, *vals))
