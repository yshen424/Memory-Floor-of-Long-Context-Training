import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

MiB = 1 << 20

def ev():
    return torch.cuda.Event(enable_timing=True)

def merge(spans):
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def overlap(span, union):
    s, e = span
    tot = 0.0
    for us, ue in union:
        if ue <= s:
            continue
        if us >= e:
            break
        tot += min(e, ue) - max(s, us)
    return tot

def med(v):
    return statistics.median(v) if v else float("nan")

class Bench:
    def __init__(self, copy_mb, phase_s):
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr)
        self.dev = torch.device("cuda", lr)
        self.phase_s = phase_s
        n = copy_mb * MiB
        self.copy_bytes = n
        self.d = {k: torch.empty(n, dtype=torch.uint8, device=self.dev) for k in ("a", "b")}
        self.h = {k: torch.empty(n, dtype=torch.uint8, pin_memory=True) for k in ("a", "b")}
        self.s_copy = torch.cuda.Stream()
        self.s_copy2 = torch.cuda.Stream()
        self.s_comm = torch.cuda.Stream()
        x = torch.ones(1, device=self.dev)
        dist.all_reduce(x)
        torch.cuda.synchronize()

    def _copies(self, stream, direction, n, rec, key="a"):
        with torch.cuda.stream(stream):
            for _ in range(n):
                s, e = ev(), ev()
                s.record(stream)
                if direction == "d2h":
                    self.h[key].copy_(self.d[key], non_blocking=True)
                else:
                    self.d[key].copy_(self.h[key], non_blocking=True)
                e.record(stream)
                rec.append((s, e, direction, self.copy_bytes))

    def _gathers(self, inp, out, n, rec):
        with torch.cuda.stream(self.s_comm):
            for _ in range(n):
                s, e = ev(), ev()
                s.record(self.s_comm)
                w = dist.all_gather_into_tensor(out, inp, async_op=True)
                w.wait()
                e.record(self.s_comm)
                rec.append((s, e, "ag", out.numel()))

    def _resolve(self, t0, rec):
        torch.cuda.synchronize()
        return [(t0.elapsed_time(s), t0.elapsed_time(e), d, b) for s, e, d, b in rec]

    def _agree(self, n):
        t = torch.tensor([n], device=self.dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def _n_copies(self, gbps_guess=20.0):
        per_ms = self.copy_bytes / (gbps_guess * 1e9) * 1e3
        return max(20, int(self.phase_s * 1e3 / per_ms))

    @staticmethod
    def _gbps(spans):
        return [b / ((e - s) * 1e-3) / 1e9 for s, e, _, b in spans if e > s]

    @staticmethod
    def _throughput(spans):
        if not spans:
            return float("nan")
        t = max(e for _, e, _, _ in spans) - min(s for s, _, _, _ in spans)
        return sum(b for *_, b in spans) / (t * 1e-3) / 1e9

    def copy_solo(self):
        res = {}
        for who in range(self.world):
            for d in ("d2h", "h2d"):
                dist.barrier()
                if self.rank == who:
                    t0 = ev(); t0.record()
                    rec = []
                    self._copies(self.s_copy, d, self._n_copies(), rec)
                    sp = self._resolve(t0, rec)
                    res[d] = {"median_gbps": med(self._gbps(sp)),
                              "throughput_gbps": self._throughput(sp)}
                torch.cuda.synchronize()
        return res

    def copy_all(self):
        res = {}
        for d in ("d2h", "h2d"):
            dist.barrier()
            t0 = ev(); t0.record()
            rec = []
            self._copies(self.s_copy, d, self._n_copies(), rec)
            sp = self._resolve(t0, rec)
            res[d] = {"median_gbps": med(self._gbps(sp)),
                      "throughput_gbps": self._throughput(sp)}
        return res

    def copy_bidir(self):
        dist.barrier()
        t0 = ev(); t0.record()
        r1, r2 = [], []
        n = self._n_copies()
        self._copies(self.s_copy, "d2h", n, r1, "a")
        self._copies(self.s_copy2, "h2d", n, r2, "b")
        s1, s2 = self._resolve(t0, r1), self._resolve(t0, r2)
        return {"d2h_throughput_gbps": self._throughput(s1),
                "h2d_throughput_gbps": self._throughput(s2)}

    def _ag_bufs(self, total_mb):
        per = total_mb * MiB // self.world
        inp = torch.empty(per, dtype=torch.uint8, device=self.dev)
        out = torch.empty(per * self.world, dtype=torch.uint8, device=self.dev)
        return inp, out

    def ag_solo(self, total_mb):
        inp, out = self._ag_bufs(total_mb)
        for _ in range(5):
            dist.all_gather_into_tensor(out, inp)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = ev(); t0.record()
        rec = []
        self._gathers(inp, out, 20, rec)
        sp = self._resolve(t0, rec)
        per_ms = med([e - s for s, e, *_ in sp])
        n = self._agree(max(20, int(self.phase_s * 1e3 / max(per_ms, 1e-3))))
        dist.barrier()
        t0 = ev(); t0.record()
        rec = []
        self._gathers(inp, out, n, rec)
        sp = self._resolve(t0, rec)
        dur = [e - s for s, e, *_ in sp]
        nbytes = out.numel()
        return {"n": n, "median_ms": med(dur),
                "algbw_gbps": nbytes / (med(dur) * 1e-3) / 1e9,
                "busbw_gbps": nbytes * (self.world - 1) / self.world / (med(dur) * 1e-3) / 1e9}

    def concurrent(self, total_mb, direction, solo_ms):
        inp, out = self._ag_bufs(total_mb)
        n_ag = self._agree(max(20, int(self.phase_s * 1e3 / max(solo_ms, 1e-3))))
        dist.barrier()
        t0 = ev(); t0.record()
        rc, rg = [], []
        self._gathers(inp, out, n_ag, rg)
        self._copies(self.s_copy, direction, self._n_copies(), rc)
        sc, sg = self._resolve(t0, rc), self._resolve(t0, rg)
        union = merge([(s, e) for s, e, *_ in sg])
        win_ms = sum(e - s for s, e in union)
        lo, hi = union[0][0], union[-1][1]
        in_bytes = 0.0
        out_bytes = 0.0
        alpha_in, alpha_out = [], []
        for s, e, _, b in sc:
            if e <= s:
                continue
            a = overlap((s, e), union) / (e - s)
            in_bytes += a * b
            out_bytes += (1 - a) * b
            g = b / ((e - s) * 1e-3) / 1e9
            if a >= 0.95:
                alpha_in.append(g)
            elif a < 0.05:
                alpha_out.append(g)
        c_lo = min(s for s, *_ in sc)
        c_hi = max(e for _, e, *_ in sc)
        copy_ms = c_hi - c_lo
        cov = sum(max(0.0, min(e, c_hi) - max(s, c_lo)) for s, e in union)
        gaps_ms = max(copy_ms - cov, 1e-6)
        return {
            "n_ag": n_ag, "n_copy": len(sc),
            "ag_median_ms": med([e - s for s, e, *_ in sg]),
            "ag_vs_solo": med([e - s for s, e, *_ in sg]) / solo_ms,
            "window_ms": win_ms, "copy_span_ms": copy_ms,
            "copy_ms_in_windows": cov,
            "in_window_copy_gbps": in_bytes / (max(cov, 1e-6) * 1e-3) / 1e9,
            "out_window_copy_gbps": out_bytes / (gaps_ms * 1e-3) / 1e9,
            "copies_alpha_ge_0.95": len(alpha_in),
            "copies_alpha_lt_0.05": len(alpha_out),
            "median_gbps_alpha_ge_0.95": med(alpha_in),
            "median_gbps_alpha_lt_0.05": med(alpha_out),
            "collectives_span_ms": hi - lo,
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy_mb", type=int, default=8)
    ap.add_argument("--phase_s", type=float, default=2.0)
    ap.add_argument("--ag_mb", type=str, default="16,128",
                    help="all-gather OUTPUT sizes (MiB). 16 ~ one FSDP layer at "
                         "the anchor shape; 128 makes windows long enough to "
                         "classify 8 MiB copies cleanly.")
    a = ap.parse_args()
    b = Bench(a.copy_mb, a.phase_s)
    sizes = [int(x) for x in a.ag_mb.split(",")]

    mine = {"rank": b.rank, "device": torch.cuda.get_device_name(b.dev)}
    mine["copy_solo"] = b.copy_solo()
    mine["copy_all"] = b.copy_all()
    mine["copy_bidir"] = b.copy_bidir()
    mine["ag_solo"] = {m: b.ag_solo(m) for m in sizes}
    mine["concurrent"] = {
        f"{m}MiB_{d}": b.concurrent(m, d, mine["ag_solo"][m]["median_ms"])
        for m in sizes for d in ("d2h", "h2d")
    }

    allr = [None] * b.world
    dist.all_gather_object(allr, mine)
    if b.rank == 0:
        peer = [[torch.cuda.can_device_access_peer(i, j) if i != j else None
                 for j in range(torch.cuda.device_count())]
                for i in range(torch.cuda.device_count())]
        nccl_v = ".".join(map(str, torch.cuda.nccl.version()))
        payload = {"ranks": allr, "can_device_access_peer": peer,
                   "nccl_version": nccl_v, "torch": torch.__version__,
                   "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
        big = max(sizes)
        r_copy, r_coll = [], []
        for d in ("d2h", "h2d"):
            for r in allr:
                c = r["concurrent"][f"{big}MiB_{d}"]
                r_copy.append(c["in_window_copy_gbps"] / r["copy_all"][d]["throughput_gbps"])
                r_coll.append(c["ag_vs_solo"])
        rc, rl = med(r_copy), med(r_coll)
        if rc < 0.10:
            verdict = "EXCLUSION"
        elif rc >= 0.90 and rl <= 1.10:
            verdict = "INDEPENDENT"
        else:
            verdict = "SHARING"
        payload["verdict"] = {"r_copy": rc, "r_coll": rl, "verdict": verdict,
                              "r_copy_per_rank_dir": r_copy, "r_coll_per_rank_dir": r_coll}
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1)

        print(f"NCCL {nccl_v}  env={payload['env']}")
        print("can_device_access_peer:", peer)
        print(f"{'rank':>4} {'solo d2h':>9} {'solo h2d':>9} {'all d2h':>8} {'all h2d':>8} "
              f"{'bidir d/h':>11}   GB/s")
        for r in allr:
            print(f"{r['rank']:>4} {r['copy_solo']['d2h']['throughput_gbps']:>9.2f} "
                  f"{r['copy_solo']['h2d']['throughput_gbps']:>9.2f} "
                  f"{r['copy_all']['d2h']['throughput_gbps']:>8.2f} "
                  f"{r['copy_all']['h2d']['throughput_gbps']:>8.2f} "
                  f"{r['copy_bidir']['d2h_throughput_gbps']:>5.2f}/{r['copy_bidir']['h2d_throughput_gbps']:<5.2f}")
        for m in sizes:
            s = allr[0]["ag_solo"][m]
            print(f"all_gather {m:>4} MiB solo: {s['median_ms']:.3f} ms  "
                  f"algbw {s['algbw_gbps']:.2f}  busbw {s['busbw_gbps']:.2f} GB/s (rank 0)")
        print(f"{'concurrent':<16} {'rank':>4} {'AG x solo':>9} {'in-win GB/s':>11} "
              f"{'out-win GB/s':>12} {'n(a>=.95)':>9} {'med(a>=.95)':>11} {'med(a<.05)':>10}")
        for k in allr[0]["concurrent"]:
            for r in allr:
                c = r["concurrent"][k]
                print(f"{k:<16} {r['rank']:>4} {c['ag_vs_solo']:>9.3f} "
                      f"{c['in_window_copy_gbps']:>11.2f} {c['out_window_copy_gbps']:>12.2f} "
                      f"{c['copies_alpha_ge_0.95']:>9d} {c['median_gbps_alpha_ge_0.95']:>11.2f} "
                      f"{c['median_gbps_alpha_lt_0.05']:>10.2f}")
        print(f"VERDICT ({big} MiB all_gather): r_copy={rc:.3f} r_coll={rl:.3f} -> {verdict}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
