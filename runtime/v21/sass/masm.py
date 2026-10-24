#!/usr/bin/env python3
"""masm.py - a minimal Maxwell (sm_5x) SASS assembler, learned from ptxas output.

No third-party code: the instruction encodings are LEARNED from `cuobjdump -sass` listings of
kernels compiled by the CUDA toolkit already on the box. Every instruction form gets a signature
(opcode + modifiers + operand kinds/flags); its constant bits ("base") and the bit position of each
operand field are fitted from the examples, and every example must re-encode bit-exactly or the
form is rejected. Assembling a form never seen in the corpus is an error (add a probe kernel that
makes ptxas emit it).

Source format (one instruction per line; '#' comments; labels 'name:'):
    <ctl> [@[!]Pn] OPCODE.MODS operands ;
    ctl = s<stall 0-15><y|-><write barrier 0-5|-><read barrier 0-5|->w<wait mask hex2>u<reuse hex1>
e.g.  s5y--w00u0 FFMA R3, R4, R5, R3 ;
Control words are emitted every 3 instructions (Maxwell bundle layout).

Python 3.6 compatible (runs on the Jetson's stock python3), stdlib only.
"""
import re
import struct
import sys

HEXW = re.compile(r'/\*\s*(0x[0-9a-f]{16})\s*\*/')
ADDR = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*)')
FUNC = re.compile(r'Function : (\S+)')


# ----------------------------------------------------------------------------- cuobjdump parsing
def parse_cuobjdump(text):
    """-> {func: (words[list of int], texts{word_index: text})}"""
    funcs = {}
    cur = None
    words = texts = None
    for line in text.splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            words, texts = [], {}
            funcs[cur] = (words, texts)
            continue
        if cur is None:
            continue
        a = ADDR.search(line)
        if a:
            addr = int(a.group(1), 16)
            t = HEXW.sub('', a.group(2))
            t = t.replace('{', ' ').replace('}', ' ').replace(';', ' ')
            texts[addr // 8] = norm_text(t)
        for h in HEXW.findall(line):
            words.append(int(h, 16))
    return funcs


def norm_text(t):
    t = re.sub(r'c\[(0x[0-9a-f]+)\]\s+\[', r'c[\1][', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ----------------------------------------------------------------------------- control words
def ctl_decode(w, slot):
    f = (w >> (21 * slot)) & 0x1fffff
    return dict(stall=f & 0xf, yld=(f >> 4) & 1, wb=(f >> 5) & 7, rb=(f >> 8) & 7,
                wait=(f >> 11) & 0x3f, reuse=(f >> 17) & 0xf)


def ctl_str(c):
    return 's%x%s%s%sw%02xu%x' % (c['stall'], 'y' if c['yld'] else '-',
                                 '-' if c['wb'] == 7 else str(c['wb']),
                                 '-' if c['rb'] == 7 else str(c['rb']), c['wait'], c['reuse'])


CTL_RE = re.compile(r'^s([0-9a-f])([y-])([0-5-])([0-5-])w([0-9a-f]{2})u([0-9a-f])$')


def ctl_parse(s):
    m = CTL_RE.match(s)
    if not m:
        raise ValueError('bad control token %r' % s)
    st, y, wb, rb, wt, ru = m.groups()
    return dict(stall=int(st, 16), yld=1 if y == 'y' else 0, wb=7 if wb == '-' else int(wb),
                rb=7 if rb == '-' else int(rb), wait=int(wt, 16), reuse=int(ru, 16))


def ctl_encode(cs):
    w = 0
    for i, c in enumerate(cs):
        f = (c['stall'] | c['yld'] << 4 | c['wb'] << 5 | c['rb'] << 8 | c['wait'] << 11 |
             c['reuse'] << 17)
        w |= f << (21 * i)
    return w


# ----------------------------------------------------------------------------- operand parsing
def split_ops(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


FLOATLIT = re.compile(r'^[-+]?(\d+(\.\d*)?([eE][-+]?\d+)?|INF|QNAN|NAN)$')


def parse_operand(op, opcode):
    """-> (kind, value, flags). kind in R P C M I F S B SB."""
    flags = ''
    o = op
    if o.startswith('|') and o.endswith('|'):
        flags += 'abs'
        o = o[1:-1]
    while o and o[0] in '-!~':
        flags += {'-': 'neg', '!': 'not', '~': 'inv'}[o[0]]
        o = o[1:]
    if o.startswith('|') and o.endswith('|'):
        flags += 'abs'
        o = o[1:-1]
    # register
    m = re.match(r'^(R\d+|RZ)((\.[A-Za-z0-9]+)*)$', o)
    if m:
        v = 255 if m.group(1) == 'RZ' else int(m.group(1)[1:])
        suf = [x for x in m.group(2).split('.') if x and x != 'reuse']
        return ('R', v, flags + ''.join('.' + x for x in suf))
    m = re.match(r'^(P[0-6]|PT)$', o)
    if m:
        return ('P', 7 if o == 'PT' else int(o[1:]), flags)
    m = re.match(r'^c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]((\.[A-Z0-9]+)*)$', o)
    if m:
        return ('C', (int(m.group(1), 16), int(m.group(2), 16)), flags + m.group(3))
    m = re.match(r'^\[(.*)\]((\.[A-Z0-9]+)*)$', o)
    if m:
        inner = m.group(1).replace(' ', '')
        mm = re.match(r'^(R\d+|RZ)(\.X4|\.X8|\.U32)?(\+(-?0x[0-9a-f]+))?$', inner)
        if mm:
            r = 255 if mm.group(1) == 'RZ' else int(mm.group(1)[1:])
            off = int(mm.group(4), 16) if mm.group(4) else 0
            return ('M', (r, off), flags + (mm.group(2) or '') + m.group(2))
        mm = re.match(r'^(-?0x[0-9a-f]+)$', inner)
        if mm:
            return ('M', (255, int(mm.group(1), 16)), flags + 'abs_addr' + m.group(2))
        raise ValueError('memory operand %r' % op)
    if o.startswith('SR_') or o in ('SRZ',):
        return ('S', o, flags)
    m = re.match(r'^SB([0-5])$', o)
    if m:
        return ('SB', int(m.group(1)), flags)
    m = re.match(r'^(-?)0x([0-9a-f]+)$', o)
    if m:
        v = int(m.group(2), 16)
        if m.group(1):
            v = -v
        if opcode.split('.')[0] in ('BRA', 'SSY', 'PBK', 'CAL', 'PCNT', 'JCAL'):
            return ('B', v, flags)
        if flags == 'neg':
            return ('I', -v, '')
        return ('I', v, flags)
    if FLOATLIT.match(o):
        if 'INF' in o or 'NAN' in o:
            f = float(o.replace('QNAN', 'nan'))
        else:
            f = float(o)
        if flags == 'neg':
            f, flags = -f, ''
        bits = struct.unpack('<I', struct.pack('<f', f))[0]
        if 'QNAN' in o:
            bits = 0x7fffffff if f == f or True else bits
        # integer-looking literals on integer opcodes are integers
        if re.match(r'^\d+$', o) and opcode.split('.')[0][0] not in 'FHD':
            return ('I', int(o), flags)
        return ('F', bits, flags)
    raise ValueError('operand %r' % op)


def parse_insn(text):
    guard = (7, 0)
    t = text
    m = re.match(r'^@(!?)(P[0-6]|PT)\s+(.*)$', t)
    if m:
        guard = (7 if m.group(2) == 'PT' else int(m.group(2)[1:]), 1 if m.group(1) else 0)
        t = m.group(3)
    parts = t.split(' ', 1)
    opcode = parts[0]
    ops = [parse_operand(o, opcode) for o in split_ops(parts[1])] if len(parts) > 1 else []
    return guard, opcode, ops


def signature(opcode, ops):
    return opcode + '|' + ','.join(k + (':' + f if f else '') for k, _, f in ops)


# ----------------------------------------------------------------------------- field encoders
def bits(v, w):
    return v & ((1 << w) - 1)


def cand_encoders(kind, slot):
    """Preference-ordered candidate encodings: (name, mask, fn(value)->bits)."""
    out = []
    if kind == 'R':
        order = [0, 8, 20, 39] if slot == 0 else [8, 20, 39, 0]
        for p in order:
            out.append(('R@%d' % p, 0xff << p, (lambda p: lambda v: (v & 0xff) << p)(p)))
    elif kind == 'P':
        for p in [3, 0, 39, 48, 44, 12, 8, 20, 29, 36, 47, 0x2f]:
            out.append(('P@%d' % p, 7 << p, (lambda p: lambda v: (v & 7) << p)(p)))
    elif kind == 'C':
        out.append(('C/4@20b34', (0x3fff << 20) | (0x1f << 34),
                    lambda v: ((v[1] >> 2) & 0x3fff) << 20 | (v[0] & 0x1f) << 34))
        out.append(('C@20b36', (0xffff << 20) | (0x1f << 36),
                    lambda v: (v[1] & 0xffff) << 20 | (v[0] & 0x1f) << 36))
    elif kind == 'M':
        out.append(('M r8 o24@20', (0xff << 8) | (0xffffff << 20),
                    lambda v: (v[0] & 0xff) << 8 | bits(v[1], 24) << 20))
        out.append(('M r8 o32@20', (0xff << 8) | (0xffffffff << 20),
                    lambda v: (v[0] & 0xff) << 8 | bits(v[1], 32) << 20))
        out.append(('M r8 o20@20', (0xff << 8) | (0xfffff << 20),
                    lambda v: (v[0] & 0xff) << 8 | bits(v[1], 20) << 20))
    elif kind == 'I':
        out.append(('I19s56', (0x7ffff << 20) | (1 << 56),
                    lambda v: bits(v, 19) << 20 | (1 if v < 0 else 0) << 56))
        out.append(('I32@20', 0xffffffff << 20, lambda v: bits(v, 32) << 20))
        out.append(('I24@20', 0xffffff << 20, lambda v: bits(v, 24) << 20))
        for p, w in [(20, 5), (28, 8), (20, 8), (39, 8), (48, 8), (36, 5), (8, 4), (20, 16), (40, 4),
                     (51, 5), (53, 3), (20, 4), (39, 5), (28, 4), (46, 2)]:
            out.append(('I%d@%d' % (w, p), ((1 << w) - 1) << p,
                        (lambda p, w: lambda v: bits(v, w) << p)(p, w)))
    elif kind == 'F':
        out.append(('F19s56', (0x7ffff << 20) | (1 << 56),
                    lambda v: ((v >> 12) & 0x7ffff) << 20 | ((v >> 31) & 1) << 56))
        out.append(('F32@20', 0xffffffff << 20, lambda v: (v & 0xffffffff) << 20))
    elif kind == 'B':
        out.append(('B24rel', 0xffffff << 20, None))   # needs the address: handled specially
    elif kind == 'S':
        out.append(('S8@20', 0xff << 20, None))       # name -> code learned per corpus
    elif kind == 'SB':
        for p in [26, 20, 23, 29]:
            out.append(('SB@%d' % p, 7 << p, (lambda p: lambda v: (v & 7) << p)(p)))
    return out


class Model:
    def __init__(self):
        self.forms = {}       # sig -> (base, [encoder names per slot], has_guard)
        self.sregs = {}       # name -> code
        self.bad = {}         # sig -> reason
        self.count = {}

    # -- learning
    def learn(self, examples):
        """examples: list of (text, word, addr)."""
        by = {}
        for text, word, addr in examples:
            try:
                g, opc, ops = parse_insn(text)
            except ValueError as e:
                self.bad.setdefault('PARSE ' + text.split(' ')[0], str(e))
                continue
            by.setdefault(signature(opc, ops), []).append((g, ops, word, addr))
        for sig, ex in by.items():
            self.count[sig] = len(ex)
            # special registers: learn codes first
            for g, ops, word, addr in ex:
                for k, v, f in ops:
                    if k == 'S':
                        self.sregs[v] = (word >> 20) & 0xff
            nslots = len(ex[0][1])
            chosen = []
            # bits 16-19 are the guard predicate for most forms (PT = 7 when unguarded); a few
            # (SSY, ...) have no guard field: then those bits belong to the base
            hasg = all(((word >> 16) & 7) == g[0] and ((word >> 19) & 1) == g[1]
                       for g, ops, word, addr in ex)
            used0 = (0xf << 16) if hasg else 0
            # per slot: every candidate that reproduces its field in all examples
            cands = []
            for s in range(nslots):
                kind = ex[0][1][s][0]
                cs = []
                for name, mask, fn in cand_encoders(kind, s):
                    if all(self._field(name, fn, ops[s][1], addr) is not None and
                           (word & mask) == self._field(name, fn, ops[s][1], addr)
                           for g, ops, word, addr in ex):
                        cs.append((name, mask, fn))
                cands.append(cs)
            if any(not c for c in cands):
                self.bad[sig] = 'no encoding for slot %d' % [i for i, c in enumerate(cands) if not c][0]
                continue
            # depth-first over the preference order: non-overlapping fields + constant base
            found = []

            def dfs(s, used, picks):
                if found:
                    return
                if s == nslots:
                    if len(set(word & ~used for g, ops, word, addr in ex)) == 1:
                        found.append((list(picks), used))
                    return
                for c in cands[s]:
                    if c[1] & used:
                        continue
                    picks.append(c)
                    dfs(s + 1, used | c[1], picks)
                    picks.pop()
                    if found:
                        return

            dfs(0, used0, [])
            if not found:
                self.bad[sig] = 'no consistent field assignment (unmodelled modifier?)'
                continue
            chosen, used = found[0]
            bases = set(word & ~used for g, ops, word, addr in ex)
            self.forms[sig] = (bases.pop(), [c[0] for c in chosen], hasg)

    def _field(self, name, fn, v, addr):
        if name == 'B24rel':
            return bits(v - (addr + 8), 24) << 20
        if name == 'S8@20':
            return (self.sregs[v] & 0xff) << 20 if v in self.sregs else None
        return fn(v)

    # -- encoding
    def encode(self, text, addr):
        g, opc, ops = parse_insn(text)
        sig = signature(opc, ops)
        if sig not in self.forms:
            raise KeyError(sig)
        base, names, hasg = self.forms[sig]
        w = base | ((g[0] & 7) << 16 | (g[1] & 1) << 19 if hasg else 0)
        for s, name in enumerate(names):
            fn = None
            for n, mask, f in cand_encoders(ops[s][0], s):
                if n == name:
                    fn = f
            w |= self._field(name, fn, ops[s][1], addr)
        return w


# ----------------------------------------------------------------------------- source <-> words
def func_examples(words, texts):
    ex = []
    for i, w in enumerate(words):
        if i % 4 == 0 or i not in texts:
            continue
        ex.append((texts[i], w, i * 8))
    return ex


def to_source(words, texts):
    """cuobjdump listing -> masm source (control decoded into per-line ctl tokens)."""
    lines = []
    for i, w in enumerate(words):
        if i % 4 == 0:
            ctlw = w
            continue
        c = ctl_decode(ctlw, (i % 4) - 1)
        t = texts.get(i, 'NOP')
        lines.append('%s %s ;' % (ctl_str(c), re.sub(r'\.reuse', '', t)))
    return lines


def assemble(model, lines):
    """masm source lines -> list of 64-bit words (control + instructions)."""
    insns = []
    for ln in lines:
        ln = ln.split('#', 1)[0].strip()
        if not ln:
            continue
        ln = ln.rstrip(';').strip()
        ctl, txt = ln.split(' ', 1)
        insns.append((ctl_parse(ctl), txt.strip()))
    words = []
    for b in range(0, len(insns), 3):
        grp = insns[b:b + 3]
        while len(grp) < 3:
            grp.append((ctl_parse('s0---w00u0'), 'NOP'))
        words.append(ctl_encode([c for c, _ in grp]))
        for j, (c, t) in enumerate(grp):
            addr = (len(words)) * 8
            words.append(model.encode(t, addr))
    return words


# ----------------------------------------------------------------------------- cubin patching
def elf_sections(data):
    """-> {name: (offset, size)} for an ELF64 little-endian file."""
    if data[:4] != b'\x7fELF' or data[4] != 2:
        raise ValueError('not an ELF64 file')
    shoff = struct.unpack_from('<Q', data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    hdrs = []
    for i in range(shnum):
        o = shoff + i * shentsize
        name, typ, flags, addr, off, size = struct.unpack_from('<IIQQQQ', data, o)
        hdrs.append((name, off, size))
    stro = hdrs[shstrndx][1]
    out = {}
    for name, off, size in hdrs:
        e = data.index(b'\0', stro + name)
        out[data[stro + name:e].decode()] = (off, size)
    return out


def patch_cubin(cubin, listing, func, source, out):
    data = bytearray(open(cubin, 'rb').read())
    funcs = parse_cuobjdump(open(listing).read())
    m = Model()
    corpus = []
    for f, (w, t) in funcs.items():
        corpus += func_examples(w, t)
    m.learn(corpus)
    words = assemble(m, open(source).read().splitlines())
    secs = elf_sections(bytes(data))
    key = '.text.' + func
    if key not in secs:
        raise SystemExit('masm: no section %s in %s' % (key, cubin))
    off, size = secs[key]
    blob = b''.join(struct.pack('<Q', w) for w in words)
    if len(blob) != size:
        raise SystemExit('masm: %s assembles to %d bytes, section is %d (size changes need the '
                         'ELF metadata rewrite, not implemented yet)' % (func, len(blob), size))
    data[off:off + size] = blob
    open(out, 'wb').write(bytes(data))
    return 0


# ----------------------------------------------------------------------------- command line
def main(argv):
    if len(argv) >= 4 and argv[1] == 'dump':
        funcs = parse_cuobjdump(open(argv[2]).read())
        fs = [f for f in funcs if argv[3] in f]
        if len(fs) != 1:
            print('dump: function must match exactly one of %r' % fs)
            return 2
        w, t = funcs[fs[0]]
        print('# masm source for %s (decoded from cuobjdump; ctl tokens carry stall/yield/'
              'barriers/reuse)' % fs[0])
        print('\n'.join(to_source(w, t)))
        return 0
    if len(argv) == 7 and argv[1] == 'patch':
        return patch_cubin(argv[2], argv[3], argv[4], argv[5], argv[6])
    if len(argv) < 3 or argv[1] != 'heldout':
        print('usage: masm.py heldout <listing.sass> <function-substring>\n'
              '       masm.py dump <listing.sass> <function-substring>\n'
              '       masm.py patch <in.cubin> <listing.sass> <mangled-func> <src.masm> <out.cubin>')
        return 2
    funcs = parse_cuobjdump(open(argv[2]).read())
    target = [f for f in funcs if argv[3] in f]
    if len(target) != 1:
        print('target must match exactly one function: %r' % target)
        return 2
    target = target[0]
    m = Model()
    corpus = []
    for f, (w, t) in funcs.items():
        if f != target:
            corpus += func_examples(w, t)
    m.learn(corpus)
    words, texts = funcs[target]
    src = to_source(words, texts)
    # assemble each instruction against the held-out model
    total = miss = wrong = 0
    unseen = {}
    for i, w in enumerate(words):
        if i % 4 == 0 or i not in texts:
            continue
        total += 1
        try:
            e = m.encode(texts[i], i * 8)
        except KeyError as k:
            miss += 1
            unseen[str(k)] = unseen.get(str(k), 0) + 1
            continue
        except ValueError as v:
            miss += 1
            unseen['PARSE ' + str(v)] = unseen.get('PARSE ' + str(v), 0) + 1
            continue
        if e != w:
            wrong += 1
            if wrong <= 10:
                print('WRONG %04x %-50s got %016x want %016x' % (i * 8, texts[i], e, w))
    print('corpus forms learned: %d, rejected: %d' % (len(m.forms), len(m.bad)))
    print('held-out %s: %d insns, %d encoded exactly, %d unseen forms, %d WRONG' %
          (target, total, total - miss - wrong, miss, wrong))
    for k, n in sorted(unseen.items(), key=lambda x: -x[1])[:25]:
        print('  unseen x%d %s' % (n, k))
    # full round trip including control words, model trained on everything
    m2 = Model()
    allx = []
    for f, (w, t) in funcs.items():
        allx += func_examples(w, t)
    m2.learn(allx)
    for k, v in list(m2.bad.items())[:30]:
        print('  rejected(all) %s: %s' % (k, v))
    try:
        out = assemble(m2, src)
    except KeyError as k:
        print('full round trip: form %s not encodable' % k)
        return 1
    same = out == words[:len(out)] and len(out) == len(words)
    print('full round trip (all-corpus model, ctl decode/encode): %s (%d words)' %
          ('BYTE-IDENTICAL' if same else 'DIFFERS', len(words)))
    if not same:
        for i, (a, b) in enumerate(zip(out, words)):
            if a != b:
                print('  first diff word %d: %016x vs %016x (%s)' % (i, a, b, texts.get(i, 'ctl')))
                break
    if m.bad:
        for k, v in list(m.bad.items())[:15]:
            print('  rejected %s: %s' % (k, v))
    return 0 if (wrong == 0 and same) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
