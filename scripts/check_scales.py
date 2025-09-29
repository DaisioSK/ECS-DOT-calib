# check_scales.py
import json, sys, math, statistics as st
p = sys.argv[1]
j = json.load(open(p))
scales = [(k, float(v["scale"])) for k,v in j.items()]
vals = [s for _,s in scales if math.isfinite(s) and s>0]
print(f"total tensors={len(scales)}  positive_scales={len(vals)}")
print(f"min={min(vals):.6g}  p10={st.quantiles(vals,n=10)[0]:.6g}  median={st.median(vals):.6g}  p90={st.quantiles(vals,n=10)[-1]:.6g}  max={max(vals):.6g}")
print("\nTop-8 smallest:")
for k,s in sorted(scales, key=lambda x:x[1])[:8]:
    print(f"  {s:.6g}  | {k}")
print("\nTop-8 largest:")
for k,s in sorted(scales, key=lambda x:x[1], reverse=True)[:8]:
    print(f"  {s:.6g}  | {k}")
# 简单地把“相同数值”的别名凑一下，看看是否存在明显重复（可能是同一张量不同展示名）
from collections import defaultdict
buckets = defaultdict(list)
for k,s in scales:
    buckets[round(s, 10)].append(k)
dups = {s:names for s,names in buckets.items() if len(names)>1}
print(f"\nscales with duplicate names count = {len(dups)} (same numeric scale shared by multiple tensor names)")

