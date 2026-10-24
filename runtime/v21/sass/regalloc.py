#!/usr/bin/env python3
"""regalloc.py - register liveness + recoloring for Maxwell SASS listings (masm companion).

Recoloring keeps every instruction and its order (so results are bit-identical) and only renames
general registers so the kernel fits a register budget -- e.g. ptxas's 80-register prefetch tile at
64 registers = 4 blocks/SM, which ptxas itself can only reach by spilling.

Model (conservative):
  * defs: the first operand of value-producing instructions; width from .64/.128 on loads;
    no register def for stores, setp/compare-to-predicate, branches, barriers.
  * uses: every other register operand; memory operands [Rn] use Rn, and Rn+1 for 64-bit
    addressing (.E); store data width from .64/.128.
  * a guarded def (@P / @!P) does not kill the old value (the register is also a use).
  * CFG: BRA (guarded = also falls through), EXIT, SSY/SYNC (SYNC continues at the innermost
    enclosing SSY target; a guarded SYNC also falls through).
Python 3.6, stdlib only.
"""
import re
import sys

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import masm  # noqa: E402

NO_DEF = re.compile(r'^(ST[A-Z]*|RED|BAR|BRA|SSY|SYNC|EXIT|DEPBAR|MEMBAR|NOP|ISETP|FSETP|PSETP|'
                    r'DSETP|HSETP2|PBK|BRK|CAL|RET|KIL|PCNT|CONT|TEXDEPBAR|CCTL)(\.|$)')
FIXED = {1, 255}    # R1 = stack pointer (c[0x0][0x20]), RZ


def width_of(opcode):
    if '.128' in opcode:
        return 4
    if '.64' in opcode:
        return 2
    return 1


def insn_regs(text):
    """-> (defs set, uses set, guarded, opcode, ops) for one instruction text."""
    g, opc, ops = masm.parse_insn(text)
    base = opc.split('.')[0]
    guarded = g != (7, 0)
    defs, uses = set(), set()
    w = width_of(opc)
    is_mem_op = base.startswith('LD') or base.startswith('ST') or base in ('RED', 'ATOM', 'ATOMS')
    wide_addr = '.E' in opc and base in ('LDG', 'STG', 'LD', 'ST', 'RED', 'ATOM')
    has_def = not NO_DEF.match(opc)
    for i, (k, v, f) in enumerate(ops):
        if k == 'R':
            if v in FIXED or v == 255:
                continue
            if i == 0 and has_def:
                n = w if is_mem_op else 1
                for j in range(n):
                    defs.add(v + j)
            else:
                # store data operand carries the access width
                n = w if (is_mem_op and base.startswith('ST')) else 1
                for j in range(n):
                    uses.add(v + j)
        elif k == 'M':
            r = v[0]
            if r != 255 and r not in FIXED:
                uses.add(r)
                if wide_addr:
                    uses.add(r + 1)
    if guarded:
        uses |= defs
    return defs, uses, guarded, opc, ops


def build(func_words, func_texts):
    """-> list of instruction dicts in address order with succ lists (indices)."""
    insns = []
    for i in sorted(func_texts):
        t = func_texts[i]
        d, u, gd, opc, ops = insn_regs(t)
        insns.append(dict(idx=len(insns), addr=i * 8, text=t, defs=d, uses=u, guarded=gd, opc=opc,
                          ops=ops))
    by_addr0 = {x['addr']: x['idx'] for x in insns}

    class _ByAddr(dict):
        # a target on a bundle's control-word slot continues at the next instruction
        def __missing__(self, a):
            if a + 8 in by_addr0:
                return by_addr0[a + 8]
            raise KeyError(a)
    by_addr = _ByAddr(by_addr0)
    # SSY regions: (ssy_addr, target)
    ssys = []
    for x in insns:
        b = x['opc'].split('.')[0]
        if b in ('SSY', 'PBK'):
            ssys.append((x['addr'], x['ops'][0][1]))
    for x in insns:
        b = x['opc'].split('.')[0]
        nxt = x['idx'] + 1 if x['idx'] + 1 < len(insns) else None
        succ = []
        if b == 'BRA':
            succ.append(by_addr[x['ops'][-1][1]])
            if x['guarded'] and nxt is not None:
                succ.append(nxt)
        elif b == 'EXIT' or b == 'RET':
            if x['guarded'] and nxt is not None:
                succ.append(nxt)
        elif b in ('SYNC', 'BRK'):
            enc = [t for a, t in ssys if a < x['addr'] < t]
            if not enc:
                raise ValueError('SYNC at %04x outside any SSY' % x['addr'])
            succ.append(by_addr[min(enc)])
            if x['guarded'] and nxt is not None:
                succ.append(nxt)
        else:
            if nxt is not None:
                succ.append(nxt)
        x['succ'] = succ
    return insns


def liveness(insns):
    """Per-instruction live-in / live-out sets (fixpoint over the instruction graph)."""
    n = len(insns)
    lin = [set() for _ in range(n)]
    lout = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            x = insns[i]
            o = set()
            for s in x['succ']:
                o |= lin[s]
            ni = x['uses'] | (o - x['defs'])
            if o != lout[i] or ni != lin[i]:
                lout[i], lin[i] = o, ni
                changed = True
    return lin, lout


def report(insns, lin, lout):
    peak = max(len(lin[i] | insns[i]['defs']) for i in range(len(insns)))
    worst = [i for i in range(len(insns)) if len(lin[i] | insns[i]['defs']) == peak]
    maxreg = max(max(x['defs'] | x['uses'] | {0}) for x in insns)
    return peak, worst, maxreg


def main(argv):
    if len(argv) < 4 or argv[1] != 'live':
        print('usage: regalloc.py live <listing.sass> <function-substring>')
        return 2
    funcs = masm.parse_cuobjdump(open(argv[2]).read())
    fs = [f for f in funcs if argv[3] in f]
    if len(fs) != 1:
        print('need exactly one match: %r' % fs)
        return 2
    w, t = funcs[fs[0]]
    insns = build(w, t)
    lin, lout = liveness(insns)
    peak, worst, maxreg = report(insns, lin, lout)
    print('%s: %d insns, highest register R%d, PEAK LIVE %d (+R1 stack ptr)' %
          (fs[0], len(insns), maxreg, peak))
    for i in worst[:4]:
        print('  peak at %04x %s' % (insns[i]['addr'], insns[i]['text']))
    # live-count histogram at FFMAs (inner loop pressure)
    ff = [len(lin[i]) for i, x in enumerate(insns) if x['opc'].startswith('FFMA')]
    if ff:
        print('  live at FFMA: min %d max %d' % (min(ff), max(ff)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
