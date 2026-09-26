# -*- coding: utf-8 -*-
"""解析 HKLM\\SYSTEM\\...\\Enum\\DISPLAY 下缓存的 EDID，弄清每块屏的真实身份与原生分辨率。"""
import winreg
from pathlib import Path

BASE = r"SYSTEM\CurrentControlSet\Enum\DISPLAY"

# EDID 里的厂商 PNP id 用 5bit 压缩编码
def pnp_id(b3: int, b4: int) -> str:
    v = (b3 << 8) | b4
    return "".join(chr(((v >> s) & 0x1F) + 0x40) for s in (10, 5, 0))


def parse(edid: bytes) -> dict:
    info = {
        "bytes": len(edid),
        "pnp": pnp_id(edid[8], edid[9]),
        "product": f"0x{edid[11]:02X}{edid[10]:02X}",
        "week": edid[16],
        "year": edid[17] + 1990,
        "descriptors": [],
        "preferred": None,
    }
    # 四个 18 字节描述符
    for off in (54, 72, 90, 108):
        d = edid[off:off + 18]
        if len(d) < 18:
            continue
        if d[0:3] == b"\x00\x00\x00" and d[3] in (0xFC, 0xFF, 0xFE):
            text = d[5:18].split(b"\x0a")[0].decode("ascii", "ignore").strip()
            info["descriptors"].append(text)
        elif d[0:2] != b"\x00\x00":
            # 详细时序：像素时钟 + 分辨率
            pixclk = ((d[1] << 8) | d[0]) * 10  # kHz
            hact = d[2] | ((d[4] & 0xF0) << 4)
            vact = d[5] | ((d[7] & 0xF0) << 4)
            if info["preferred"] is None and hact and vact:
                info["preferred"] = f"{hact}x{vact} @ {pixclk/1000:.1f}MHz"
    return info


def walk(key, path=""):
    out = []
    try:
        n_sub, n_val, _ = winreg.QueryInfoKey(key)
    except OSError:
        return out
    for i in range(n_sub):
        name = winreg.EnumKey(key, i)
        sub = winreg.OpenKey(key, name)
        child_path = f"{path}\\{name}"
        try:
            dp = winreg.OpenKey(sub, "Device Parameters")
            try:
                edid, _ = winreg.QueryValueEx(dp, "EDID")
            except FileNotFoundError:
                edid = None
            out.append((child_path, edid))
        except FileNotFoundError:
            out.extend(walk(sub, child_path))
    return out


root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, BASE)
print(f"{'容器':<46} {'PNP':<5} {'产品':<8} {'年份':<6} {'原生/首选':<22} 名称")
print("-" * 118)
for path, edid in walk(root):
    if edid is None:
        print(f"{path:<46} {'-':<5} {'-':<8} {'-':<6} {'*无 EDID*':<22}")
        continue
    try:
        info = parse(bytes(edid))
    except Exception as exc:  # noqa: BLE001
        print(f"{path:<46} 解析失败: {exc}")
        continue
    name = " / ".join(info["descriptors"]) or "-"
    print(f"{path:<46} {info['pnp']:<5} {info['product']:<8} {info['year']:<6} {str(info['preferred']):<22} {name}")
