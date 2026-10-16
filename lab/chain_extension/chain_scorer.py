"""Three-intersection chain evaluator A -> B -> C (extension of the official scorer.py).

Usage: python chain_scorer.py --scenario chain.json --schedule 111010101100
Schedule order: A slots 1..4, B slots 1..4, C slots 1..4; 1 = NS green, 0 = EW green.

Slot semantics are copied line by line from scorer.py; the only generalisation is that
there are two directed links, A->B and B->C, each on one phase with its own fraction:

  for t = 1..4, for node in (A, B, C), for phase in (NS, EW):
      inflow    = arrivals[node][phase][t] + (pending[node][t] if phase is the phase of the link INTO node)
      available = queue + inflow
      departures= min(available, capacity) if phase is green else 0
      queue     = available - departures
      delay    += queue ; overflow += max(0, queue - storage)
      if phase is the phase of the link OUT OF node and t < 4:
          pending[downstream][t+1] += floor(fraction * departures + 1/2)   (exact decimal half-up)
  switches = sum_{node} sum_{t=2..4} [x_{node,t} != x_{node,t-1}]
  cost     = delay + overflow_weight * overflow + switch_weight * switches

Transfers into B are A's departures on the A->B phase; transfers into C are B's departures on
the B->C phase (which, if both links use the same phase, include vehicles that came from A).
Standard library only, like scorer.py.
"""
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

NODES = ('A', 'B', 'C')
PHASES = ('NS', 'EW')
N_BITS = 12


def decode_schedule(raw):
    if isinstance(raw, dict):
        if 'bits' in raw:
            return decode_schedule(raw['bits'])
        if set(raw) == set(NODES):
            return [*raw['A'], *raw['B'], *raw['C']]
        raise ValueError('Schedule object must have bits or A, B and C arrays')
    if isinstance(raw, str):
        if len(raw) != N_BITS or any(c not in '01' for c in raw):
            raise ValueError('Schedule string must contain exactly twelve 0/1 characters')
        return [int(c) for c in raw]
    if isinstance(raw, (list, tuple)):
        return [int(b) for b in raw]
    raise ValueError('Invalid schedule format')


def validate_chain(s):
    for key in ('arrivals', 'service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight', 'links'):
        if key not in s:
            raise ValueError('Missing scenario key: ' + key)
    for node in NODES:
        for phase in PHASES:
            vals = s['arrivals'][node][phase]
            if len(vals) != 4 or any(type(x) is not int or x < 0 for x in vals):
                raise ValueError('Each arrival series must contain four nonnegative integers')
    links = s['links']
    if len(links) != 2 or (links[0]['from'], links[0]['to']) != ('A', 'B') or (links[1]['from'], links[1]['to']) != ('B', 'C'):
        raise ValueError('Expected links [A->B, B->C]')
    for link in links:
        if link['phase'] not in PHASES:
            raise ValueError('Link phase must be NS or EW')
        if not (0 <= link['fraction'] <= 1):
            raise ValueError('Link fraction must be between 0 and 1')
    if type(s['service_capacity']) is not int or s['service_capacity'] < 1 or type(s['storage_capacity']) is not int or s['storage_capacity'] < 0:
        raise ValueError('Capacity and storage must be valid integers')
    if s['overflow_weight'] < 0 or s['switch_weight'] < 0:
        raise ValueError('Weights must be nonnegative')


def transfer(fraction, departures):
    """floor(fraction * departures + 0.5), exact decimal half-up exactly as scorer.py."""
    amount = Decimal(str(fraction)) * Decimal(departures)
    return int(amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def evaluate(scenario, schedule):
    validate_chain(scenario)
    bits = decode_schedule(schedule)
    if len(bits) != N_BITS or any(b not in (0, 1) for b in bits):
        raise ValueError('Schedule must contain twelve bits')
    plans = {'A': bits[0:4], 'B': bits[4:8], 'C': bits[8:12]}
    link_in = {l['to']: l for l in scenario['links']}
    link_out = {l['from']: l for l in scenario['links']}
    queues = {n: {'NS': 0, 'EW': 0} for n in NODES}
    pending = {'B': [0, 0, 0, 0], 'C': [0, 0, 0, 0]}
    cap, store = scenario['service_capacity'], scenario['storage_capacity']
    delay = overflow = 0
    for t in range(4):
        for node in NODES:
            for phase in PHASES:
                inflow = scenario['arrivals'][node][phase][t]
                if node in link_in and phase == link_in[node]['phase']:
                    inflow += pending[node][t]
                available = queues[node][phase] + inflow
                is_green = plans[node][t] == (1 if phase == 'NS' else 0)
                departures = min(available, cap) if is_green else 0
                queues[node][phase] = available - departures
                delay += queues[node][phase]
                overflow += max(0, queues[node][phase] - store)
                if node in link_out and phase == link_out[node]['phase'] and t < 3:
                    pending[link_out[node]['to']][t + 1] += transfer(link_out[node]['fraction'], departures)
    switches = sum(plans[n][t] != plans[n][t - 1] for n in NODES for t in range(1, 4))
    cost = delay + scenario['overflow_weight'] * overflow + scenario['switch_weight'] * switches
    return {'cost': round(cost, 6), 'delay': delay, 'overflow': overflow, 'switches': switches,
            'final_queues': queues}


# ---------------------------------------------------------------- reductions to the official 2-node model
def _common(s):
    return {k: s[k] for k in ('service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight')}


def restrict_AB(s):
    """The official two-intersection scenario formed by A, B and the A->B link."""
    out = {'name': f"{s.get('name', 'chain')}|AB", 'arrivals': {n: s['arrivals'][n] for n in ('A', 'B')}, **_common(s),
           'link': dict(s['links'][0])}
    return out


def restrict_BC(s):
    """The official two-intersection scenario formed by B, C and the B->C link, renamed B->A, C->B."""
    link = dict(s['links'][1])
    link.update({'from': 'A', 'to': 'B'})
    return {'name': f"{s.get('name', 'chain')}|BC", 'arrivals': {'A': s['arrivals']['B'], 'B': s['arrivals']['C']},
            **_common(s), 'link': link}


def embed_two_node(s2, c_arrivals=None, bc_phase='NS', bc_fraction=0.0, name=None):
    """Chain scenario whose A, B part is the official scenario s2; C is appended downstream."""
    c_arr = c_arrivals or {'NS': [0, 0, 0, 0], 'EW': [0, 0, 0, 0]}
    return {'name': name or f"{s2.get('name', 's2')}+C", 'arrivals': {**s2['arrivals'], 'C': c_arr}, **_common(s2),
            'links': [dict(s2['link']), {'from': 'B', 'to': 'C', 'phase': bc_phase, 'fraction': bc_fraction}]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', required=True)
    p.add_argument('--schedule', required=True, help='12 bits, or path to JSON schedule')
    a = p.parse_args()
    s = json.loads(Path(a.scenario).read_text())
    raw = json.loads(Path(a.schedule).read_text()) if Path(a.schedule).suffix.lower() == '.json' else a.schedule
    print(json.dumps(evaluate(s, raw), indent=2))


if __name__ == '__main__':
    main()
