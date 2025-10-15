# save as filter_log.py
import sys

inp = sys.argv[1] if len(sys.argv) > 1 else "sc.log"
out = sys.argv[2] if len(sys.argv) > 2 else "sc.filtered.log"

# errors='ignore' 防止偶发非 UTF-8 字节导致中断
with open(inp, "r", errors="ignore") as fin, open(out, "w") as fout:
    write = fout.write
    for line in fin:
        if "csb_adaptor" in line or "dbb_adaptor" in line:
            write(line)
