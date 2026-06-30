"""Dynamic bbox-relative gravity ONNX (Phase 21).

Uses compile-time max grid extent (from task examples) to bound graph size.
Runtime gh/gw scalars mask the active top-left region on the 30×30 canvas.
"""

from __future__ import annotations

from onnx import TensorProto, helper

from arc_genome.data.encoding import CH, GH, GW
from arc_genome.onnx.bounded import _bbox_scalars, _f32, _i64
from arc_genome.onnx.model import make_model

DT = TensorProto.FLOAT


def max_grid_extent(task_data: dict) -> tuple[int, int]:
    mh = mw = 0
    for split in ("train", "test"):
        for ex in task_data.get(split, []):
            h, w = len(ex["input"]), len(ex["input"][0])
            mh, mw = max(mh, h), max(mw, w)
    return max(mh, 1), max(mw, 1)


def _cell_sum(nodes, inits, tensor: str, r: int, c: int, tag: str) -> str:
    s, e = f"{tag}_s", f"{tag}_e"
    inits += [_i64(s, [0, 0, r, c]), _i64(e, [1, 1, r + 1, c + 1])]
    sl = f"{tag}_sl"
    nodes.append(helper.make_node("Slice", [tensor, s, e], [sl]))
    out = f"{tag}_v"
    nodes.append(helper.make_node("ReduceSum", [sl], [out], keepdims=0, axes=[0, 1, 2, 3]))
    return out


def _input_ch(nodes, inits, ch: int, r: int, c: int, tag: str) -> str:
    s, e = f"{tag}_s", f"{tag}_e"
    inits += [_i64(s, [0, ch, r, c]), _i64(e, [1, ch + 1, r + 1, c + 1])]
    sl = f"{tag}_sl"
    nodes.append(helper.make_node("Slice", ["input", s, e], [sl]))
    out = f"{tag}_v"
    nodes.append(helper.make_node("ReduceSum", [sl], [out], keepdims=0, axes=[0, 1, 2, 3]))
    return out


def _eq_tensors(nodes, a: str, b: str, tag: str) -> str:
    sub = f"{tag}_sub"
    nodes.append(helper.make_node("Sub", [a, b], [sub]))
    ab = f"{tag}_ab"
    nodes.append(helper.make_node("Abs", [sub], [ab]))
    ok = f"{tag}_ok"
    nodes.append(helper.make_node("Less", [ab, "half"], [ok]))
    out = f"{tag}_m"
    nodes.append(helper.make_node("Cast", [ok], [out], to=DT))
    return out


def build_gravity_model(direction: str, max_h: int, max_w: int):
    if direction not in ("up", "down"):
        raise ValueError(direction)

    nodes: list = []
    inits = [
        _f32("half", 0.5),
        _f32("one", 1.0),
        _f32("zero", 0.0),
        _i64("ch1_s", [0, 1, 0, 0]),
        _i64("ch1_e", [1, CH, GH, GW]),
        _i64("plane_shape", [1, 1, GH, GW]),
        _i64("sh1", [1]),
    ]
    for r in range(max_h):
        inits.append(_f32(f"rf{r}", float(r)))
    for c in range(max_w):
        inits.append(_f32(f"cf{c}", float(c)))

    nodes.append(helper.make_node("Slice", ["input", "ch1_s", "ch1_e"], ["ch19"]))
    nodes.append(helper.make_node("ReduceSum", ["ch19"], ["colored"], keepdims=1, axes=[1]))
    nodes.append(helper.make_node("ReduceMax", ["input"], ["in_grid"], keepdims=1, axes=[1]))
    gh, gw = _bbox_scalars(nodes, inits, "in_grid", "ge")

    # colored scalars [max_h, max_w]
    cg = [[_cell_sum(nodes, inits, "colored", r, c, f"cg{r}x{c}") for c in range(max_w)] for r in range(max_h)]

    col_cum: dict[int, list[str]] = {}
    col_k: dict[int, str] = {}
    for c in range(max_w):
        run = None
        lst = []
        for r in range(max_h):
            if run is None:
                run = cg[r][c]
            else:
                nm = f"cc{c}_r{r}"
                nodes.append(helper.make_node("Add", [run, cg[r][c]], [nm]))
                run = nm
            lst.append(run)
        col_cum[c] = lst
        terms = []
        for r in range(max_h):
            gt = f"ck{c}_gt{r}"
            nodes.append(helper.make_node("Greater", [gh, f"rf{r}"], [gt]))
            m = f"ck{c}_m{r}"
            nodes.append(helper.make_node("Cast", [gt], [m], to=DT))
            tm = f"ck{c}_tm{r}"
            nodes.append(helper.make_node("Mul", [cg[r][c], m], [tm]))
            terms.append(tm)
        ktot = terms[0]
        for i, tm in enumerate(terms[1:], 1):
            nm = f"ck{c}_k{i}"
            nodes.append(helper.make_node("Add", [ktot, tm], [nm]))
            ktot = nm
        col_k[c] = ktot

    ch_planes = []
    for ch in range(CH):
        cells = []
        for r_out in range(GH):
            for c_out in range(GW):
                tag = f"y{ch}_{r_out}_{c_out}"
                if r_out >= max_h or c_out >= max_w:
                    z = f"{tag}_z"
                    nodes.append(helper.make_node("Mul", ["zero", "zero"], [z]))
                    out = f"{tag}_out"
                    nodes.append(helper.make_node("Reshape", [z, "sh1"], [out]))
                    cells.append(out)
                    continue

                rf, cf = f"rf{r_out}", f"cf{c_out}"
                rgt = f"{tag}_rgt"
                cgt = f"{tag}_cgt"
                nodes.append(helper.make_node("Greater", [gh, rf], [rgt]))
                nodes.append(helper.make_node("Greater", [gw, cf], [cgt]))
                rin = f"{tag}_rin"
                cin = f"{tag}_cin"
                nodes.append(helper.make_node("Cast", [rgt], [rin], to=DT))
                nodes.append(helper.make_node("Cast", [cgt], [cin], to=DT))
                in_m = f"{tag}_in"
                nodes.append(helper.make_node("Mul", [rin, cin], [in_m]))

                matches = []
                if direction == "up":
                    want = f"{tag}_want"
                    inits.append(_f32(want, float(r_out + 1)))
                    lt = f"{tag}_lt"
                    nodes.append(helper.make_node("Sub", [col_k[c_out], "one"], [lt]))
                    for s in range(max_h):
                        pres = f"{tag}_pr{s}"
                        nodes.append(helper.make_node("Greater", [cg[s][c_out], "half"], [pres]))
                        pm = f"{tag}_pm{s}"
                        nodes.append(helper.make_node("Cast", [pres], [pm], to=DT))
                        rank_ok = _eq_tensors(nodes, col_cum[c_out][s], want, f"{tag}_rk{s}")
                        rk_lt = f"{tag}_rkl{s}"
                        nodes.append(helper.make_node("Greater", [col_k[c_out], rf], [rk_lt]))
                        lm = f"{tag}_lm{s}"
                        nodes.append(helper.make_node("Cast", [rk_lt], [lm], to=DT))
                        m = f"{tag}_m{s}"
                        nodes.append(helper.make_node("Mul", [pm, rank_ok], [m]))
                        m2 = f"{tag}_m2{s}"
                        nodes.append(helper.make_node("Mul", [m, lm], [m2]))
                        matches.append(m2)
                else:
                    lb = f"{tag}_lb"
                    nodes.append(helper.make_node("Sub", [gh, col_k[c_out]], [lb]))
                    lbm1 = f"{tag}_lbm1"
                    nodes.append(helper.make_node("Sub", [lb, "one"], [lbm1]))
                    ge = f"{tag}_ge"
                    nodes.append(helper.make_node("Greater", [rf, lbm1], [ge]))
                    gm = f"{tag}_gm"
                    nodes.append(helper.make_node("Cast", [ge], [gm], to=DT))
                    lt = f"{tag}_lt"
                    nodes.append(helper.make_node("Sub", [gh, rf], [lt]))
                    lg = f"{tag}_lg"
                    nodes.append(helper.make_node("Greater", [lt, "half"], [lg]))
                    lgm = f"{tag}_lgm"
                    nodes.append(helper.make_node("Cast", [lg], [lgm], to=DT))
                    want_a = f"{tag}_wa"
                    nodes.append(helper.make_node("Add", [rf, "one"], [want_a]))
                    want_b = f"{tag}_wb"
                    nodes.append(helper.make_node("Sub", [want_a, lb], [want_b]))
                    for s in range(max_h):
                        pres = f"{tag}_pr{s}"
                        nodes.append(helper.make_node("Greater", [cg[s][c_out], "half"], [pres]))
                        pm = f"{tag}_pm{s}"
                        nodes.append(helper.make_node("Cast", [pres], [pm], to=DT))
                        rank_ok = _eq_tensors(nodes, col_cum[c_out][s], want_b, f"{tag}_rk{s}")
                        m = f"{tag}_m{s}"
                        nodes.append(helper.make_node("Mul", [pm, rank_ok], [m]))
                        m2 = f"{tag}_m2{s}"
                        nodes.append(helper.make_node("Mul", [m, gm], [m2]))
                        m3 = f"{tag}_m3{s}"
                        nodes.append(helper.make_node("Mul", [m2, lgm], [m3]))
                        matches.append(m3)

                terms = []
                for s in range(max_h):
                    src = _input_ch(nodes, inits, ch, s, c_out, f"{tag}_x{s}")
                    wm = f"{tag}_w{s}"
                    nodes.append(helper.make_node("Mul", [matches[s], src], [wm]))
                    terms.append(wm)
                gathered = terms[0]
                for i, wm in enumerate(terms[1:], 1):
                    nm = f"{tag}_g{i}"
                    nodes.append(helper.make_node("Add", [gathered, wm], [nm]))
                    gathered = nm

                if ch == 0:
                    if direction == "up":
                        empty = f"{tag}_empty"
                        nodes.append(helper.make_node("Sub", [col_k[c_out], "one"], [empty]))
                        em = f"{tag}_em"
                        nodes.append(helper.make_node("Greater", [rf, empty], [em]))
                    else:
                        lb0 = f"{tag}_lb0"
                        nodes.append(helper.make_node("Sub", [gh, col_k[c_out]], [lb0]))
                        em = f"{tag}_em"
                        nodes.append(helper.make_node("Greater", [lb0, rf], [em]))
                    emf = f"{tag}_emf"
                    nodes.append(helper.make_node("Cast", [em], [emf], to=DT))
                    fill = f"{tag}_fill"
                    nodes.append(helper.make_node("Mul", [in_m, emf], [fill]))
                    val = f"{tag}_val"
                    nodes.append(helper.make_node("Add", [gathered, fill], [val]))
                else:
                    val = gathered

                out = f"{tag}_out"
                nodes.append(helper.make_node("Mul", [val, in_m], [out]))
                out1 = f"{tag}_o1"
                nodes.append(helper.make_node("Reshape", [out, "sh1"], [out1]))
                cells.append(out1)

        plane = f"plane{ch}"
        nodes.append(helper.make_node("Concat", cells, [plane], axis=0))
        resh = f"ch{ch}"
        nodes.append(helper.make_node("Reshape", [plane, "plane_shape"], [resh]))
        ch_planes.append(resh)

    nodes.append(helper.make_node("Concat", ch_planes, ["output"], axis=1))
    return make_model(nodes, inits)


def build_gravity_up_model(max_h: int = GH, max_w: int = GW):
    return build_gravity_model("up", max_h, max_w)


def build_gravity_down_model(max_h: int = GH, max_w: int = GW):
    return build_gravity_model("down", max_h, max_w)


def _src_row_from_matches(nodes, inits, matches: list[str], max_h: int, tag: str) -> str:
    """Weighted sum of row indices where exactly one match is active."""
    terms = []
    for s, m in enumerate(matches):
        if s < max_h:
            inits.append(_f32(f"{tag}_ws{s}", float(s)))
            tm = f"{tag}_t{s}"
            nodes.append(helper.make_node("Mul", [m, f"{tag}_ws{s}"], [tm]))
            terms.append(tm)
    if not terms:
        return "zero"
    row = terms[0]
    for i, tm in enumerate(terms[1:], 1):
        nm = f"{tag}_sr{i}"
        nodes.append(helper.make_node("Add", [row, tm], [nm]))
        row = nm
    return row


def build_gravity_model_slim(direction: str, max_h: int, max_w: int):
    """Single-Gather gravity — all channels share spatial index (~10× smaller)."""
    if direction not in ("up", "down", "left", "right"):
        raise ValueError(direction)

    nodes: list = []
    inits = [
        _f32("half", 0.5),
        _f32("one", 1.0),
        _f32("zero", 0.0),
        _i64("flat_shape", [1, CH, GH * GW]),
        _i64("out_shape", [1, CH, GH, GW]),
        _i64("mask_shape", [1, 1, GH, GW]),
        _i64("sh1", [1]),
        _f32("gw_f", float(GW)),
        _i64("ch1_s", [0, 1, 0, 0]),
        _i64("ch1_e", [1, CH, GH, GW]),
    ]
    for r in range(GH):
        inits.append(_f32(f"rf{r}", float(r)))
    for c in range(GW):
        inits.append(_f32(f"cf{c}", float(c)))

    nodes.append(helper.make_node("Slice", ["input", "ch1_s", "ch1_e"], ["ch19"]))
    nodes.append(helper.make_node("ReduceSum", ["ch19"], ["colored"], keepdims=1, axes=[1]))
    nodes.append(helper.make_node("ReduceMax", ["input"], ["in_grid"], keepdims=1, axes=[1]))
    gh, gw = _bbox_scalars(nodes, inits, "in_grid", "ge")

    cg = [[_cell_sum(nodes, inits, "colored", r, c, f"cg{r}x{c}") for c in range(max_w)] for r in range(max_h)]
    col_cum: dict[int, list[str]] = {}
    col_k: dict[int, str] = {}
    for c in range(max_w):
        run = None
        lst = []
        for r in range(max_h):
            if run is None:
                run = cg[r][c]
            else:
                nm = f"cc{c}_r{r}"
                nodes.append(helper.make_node("Add", [run, cg[r][c]], [nm]))
                run = nm
            lst.append(run)
        col_cum[c] = lst
        terms = []
        for r in range(max_h):
            gt = f"ck{c}_gt{r}"
            nodes.append(helper.make_node("Greater", [gh, f"rf{r}"], [gt]))
            m = f"ck{c}_m{r}"
            nodes.append(helper.make_node("Cast", [gt], [m], to=DT))
            tm = f"ck{c}_tm{r}"
            nodes.append(helper.make_node("Mul", [cg[r][c], m], [tm]))
            terms.append(tm)
        ktot = terms[0] if terms else "zero"
        for i, tm in enumerate(terms[1:], 1):
            nm = f"ck{c}_k{i}"
            nodes.append(helper.make_node("Add", [ktot, tm], [nm]))
            ktot = nm
        col_k[c] = ktot

    flat_indices: list[str] = []
    content_masks: list[str] = []
    bg_masks: list[str] = []
    for r in range(GH):
        for c in range(GW):
            tag = f"r{r}c{c}"
            rf, cf = f"rf{r}", f"cf{c}"
            nodes.append(helper.make_node("Greater", [gh, rf], [f"{tag}_rgt"]))
            nodes.append(helper.make_node("Greater", [gw, cf], [f"{tag}_cgt"]))
            nodes.append(helper.make_node("Cast", [f"{tag}_rgt"], [f"{tag}_rm"], to=DT))
            nodes.append(helper.make_node("Cast", [f"{tag}_cgt"], [f"{tag}_cm"], to=DT))
            in_m = f"{tag}_in"
            nodes.append(helper.make_node("Mul", [f"{tag}_rm", f"{tag}_cm"], [in_m]))

            if r >= max_h or c >= max_w:
                fi = f"{tag}_fi0"
                nodes.append(helper.make_node("Cast", ["zero"], [fi], to=TensorProto.INT64))
                fi1 = f"{tag}_fi1"
                nodes.append(helper.make_node("Reshape", [fi, "sh1"], [fi1]))
                flat_indices.append(fi1)
                msk1 = f"{tag}_msk0"
                nodes.append(helper.make_node("Reshape", ["zero", "sh1"], [msk1]))
                content_masks.append(msk1)
                bg_masks.append(msk1)
                continue

            matches = []
            if direction == "up":
                want = f"{tag}_want"
                inits.append(_f32(want, float(r + 1)))
                slot_ok = f"{tag}_slot"
                nodes.append(helper.make_node("Sub", [col_k[c], "one"], [f"{tag}_kless"]))
                nodes.append(helper.make_node("Greater", [col_k[c], rf], [f"{tag}_slotgt"]))
                nodes.append(helper.make_node("Cast", [f"{tag}_slotgt"], [slot_ok], to=DT))
                for s in range(max_h):
                    pres = f"{tag}_pr{s}"
                    nodes.append(helper.make_node("Greater", [cg[s][c], "half"], [pres]))
                    pm = f"{tag}_pm{s}"
                    nodes.append(helper.make_node("Cast", [pres], [pm], to=DT))
                    rank_ok = _eq_tensors(nodes, col_cum[c][s], want, f"{tag}_rk{s}")
                    rk_lt = f"{tag}_rkl{s}"
                    nodes.append(helper.make_node("Greater", [col_k[c], rf], [rk_lt]))
                    lm = f"{tag}_lm{s}"
                    nodes.append(helper.make_node("Cast", [rk_lt], [lm], to=DT))
                    m = f"{tag}_m{s}"
                    nodes.append(helper.make_node("Mul", [pm, rank_ok], [m]))
                    m2 = f"{tag}_m2{s}"
                    nodes.append(helper.make_node("Mul", [m, lm], [m2]))
                    matches.append(m2)
            else:  # down
                lb = f"{tag}_lb"
                nodes.append(helper.make_node("Sub", [gh, col_k[c]], [lb]))
                lbm1 = f"{tag}_lbm1"
                nodes.append(helper.make_node("Sub", [lb, "one"], [lbm1]))
                ge = f"{tag}_ge"
                nodes.append(helper.make_node("Greater", [rf, lbm1], [ge]))
                gm = f"{tag}_gm"
                nodes.append(helper.make_node("Cast", [ge], [gm], to=DT))
                lt = f"{tag}_lt"
                nodes.append(helper.make_node("Sub", [gh, rf], [lt]))
                lg = f"{tag}_lg"
                nodes.append(helper.make_node("Greater", [lt, "half"], [lg]))
                lgm = f"{tag}_lgm"
                nodes.append(helper.make_node("Cast", [lg], [lgm], to=DT))
                slot_ok = f"{tag}_slot"
                nodes.append(helper.make_node("Cast", [gm], [slot_ok], to=DT))
                want_a = f"{tag}_wa"
                nodes.append(helper.make_node("Add", [rf, "one"], [want_a]))
                want_b = f"{tag}_wb"
                nodes.append(helper.make_node("Sub", [want_a, lb], [want_b]))
                for s in range(max_h):
                    pres = f"{tag}_pr{s}"
                    nodes.append(helper.make_node("Greater", [cg[s][c], "half"], [pres]))
                    pm = f"{tag}_pm{s}"
                    nodes.append(helper.make_node("Cast", [pres], [pm], to=DT))
                    rank_ok = _eq_tensors(nodes, col_cum[c][s], want_b, f"{tag}_rk{s}")
                    m = f"{tag}_m{s}"
                    nodes.append(helper.make_node("Mul", [pm, rank_ok], [m]))
                    m2 = f"{tag}_m2{s}"
                    nodes.append(helper.make_node("Mul", [m, gm], [m2]))
                    m3 = f"{tag}_m3{s}"
                    nodes.append(helper.make_node("Mul", [m2, lgm], [m3]))
                    matches.append(m3)

            src_row = _src_row_from_matches(nodes, inits, matches, max_h, tag)
            nodes.append(helper.make_node("Mul", [src_row, "gw_f"], [f"{tag}_rowoff"]))
            nodes.append(helper.make_node("Add", [f"{tag}_rowoff", cf], [f"{tag}_flatf"]))
            fi = f"{tag}_fi"
            nodes.append(helper.make_node("Cast", [f"{tag}_flatf"], [fi], to=TensorProto.INT64))
            fi1 = f"{tag}_fi1"
            nodes.append(helper.make_node("Reshape", [fi, "sh1"], [fi1]))
            flat_indices.append(fi1)

            has_src = matches[0]
            for i, m in enumerate(matches[1:], 1):
                nm = f"{tag}_hm{i}"
                nodes.append(helper.make_node("Add", [has_src, m], [nm]))
                has_src = nm
            content_m = f"{tag}_cm1"
            nodes.append(helper.make_node("Mul", [in_m, slot_ok], [f"{tag}_sm"]))
            nodes.append(helper.make_node("Mul", [f"{tag}_sm", has_src], [content_m]))
            empty_m = f"{tag}_em"
            nodes.append(helper.make_node("Sub", ["one", slot_ok], [f"{tag}_ns"]))
            nodes.append(helper.make_node("Mul", [in_m, f"{tag}_ns"], [empty_m]))
            cm1 = f"{tag}_cmsk"
            nodes.append(helper.make_node("Reshape", [content_m, "sh1"], [cm1]))
            content_masks.append(cm1)
            em1 = f"{tag}_bmsk"
            nodes.append(helper.make_node("Reshape", [empty_m, "sh1"], [em1]))
            bg_masks.append(em1)

    nodes.append(helper.make_node("Concat", flat_indices, ["flat_idx"], axis=0))
    nodes.append(helper.make_node("Concat", content_masks, ["content_flat"], axis=0))
    nodes.append(helper.make_node("Concat", bg_masks, ["bg_flat"], axis=0))
    nodes.append(helper.make_node("Reshape", ["content_flat", "mask_shape"], ["content2d"]))
    nodes.append(helper.make_node("Reshape", ["bg_flat", "mask_shape"], ["bg2d"]))
    nodes.append(helper.make_node("Reshape", ["input", "flat_shape"], ["flat_in"]))
    nodes.append(helper.make_node("Gather", ["flat_in", "flat_idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Reshape", ["gathered", "out_shape"], ["raw_out"]))
    nodes.append(helper.make_node("Mul", ["raw_out", "content2d"], ["colored_out"]))
    ch0_slice_s, ch0_slice_e = "ch0_s", "ch0_e"
    inits += [_i64(ch0_slice_s, [0, 0, 0, 0]), _i64(ch0_slice_e, [1, 1, GH, GW])]
    nodes.append(helper.make_node("Slice", ["colored_out", ch0_slice_s, ch0_slice_e], ["ch0_old"]))
    nodes.append(helper.make_node("Add", ["ch0_old", "bg2d"], ["ch0_new"]))
    ch_rest = []
    for k in range(1, CH):
        ks, ke = f"ck{k}s", f"ck{k}e"
        inits += [_i64(ks, [0, k, 0, 0]), _i64(ke, [1, k + 1, GH, GW])]
        xk = f"ck{k}sl"
        nodes.append(helper.make_node("Slice", ["colored_out", ks, ke], [xk]))
        ch_rest.append(xk)
    nodes.append(helper.make_node("Concat", ["ch0_new", *ch_rest], ["output"], axis=1))
    return make_model(nodes, inits)
