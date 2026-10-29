# Toolchain setup

Host: Windows 10 Home 19045, 16 GB RAM.

## Status (2026-10-27)

| Item | State |
|---|---|
| Hardware virtualization (VT-x / AMD-V) | Enabled in BIOS 2026-10-27 (HP ENVY x360 m6, Insyde F.23: Configuration → Virtualization Technology); `systeminfo` now reports a running hypervisor |
| WSL | 3.0.3.0, kernel 6.18.54.1; Ubuntu 24.04.5 LTS (WSL2), user `santosh` |
| `.wslconfig` | `memory=12GB`, `swap=8GB` |
| Verilator / Icarus | 5.020 / 12.0 (Ubuntu apt) |
| XLS | `v0.0.0-10898-g8ddae8bf4` at `~/opt/xls` (symlink to `~/opt/xls-<tag>`), tarball sha256 `055b8289…09e4ca3` |
| Docker | 29.1.3 (Ubuntu `docker.io`), systemd-managed; `santosh` in `docker` group |
| ORFS | `openroad/orfs:latest`, digest `sha256:131a6803…6d44901a`, 6.1 GB on disk |
| Python | Windows 3.14.3; WSL 3.12.3 (repo venv at `~/src/xls-power-rl/.venv`) |

## 1. Enable virtualization in BIOS/UEFI

`systeminfo` reports `Virtualization Enabled In Firmware: No`. WSL2 needs it.

1. Settings → Update & Security → Recovery → Advanced startup → Restart now → Troubleshoot → Advanced options → UEFI Firmware Settings.
2. Find the setting (usually under Advanced / CPU Configuration): **Intel Virtualization Technology (VT-x)** or **SVM Mode** on AMD. Enable it, save, reboot.
3. Check: `systeminfo` should now show `Virtualization Enabled In Firmware: Yes`.

## 2. Install WSL2 + Ubuntu 24.04 (admin PowerShell)

The Windows 10 inbox `wsl.exe` only knows an unversioned `Ubuntu`, so install current WSL first:

```powershell
wsl --install --web-download --no-distribution   # reboot after this
wsl --install -d Ubuntu-24.04
```

Reboot when asked, create the Linux user, then cap memory in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=12GB
```

Run `wsl --shutdown` to apply.

## 3. Inside WSL

- Clone this repo under the Linux filesystem (`~/src/xls-power-rl`), not `/mnt/c` (much faster I/O).
- XLS: release tarball from github.com/google/xls/releases (pin the version in `docs/decisions.md`).
- Verilator, Icarus: `sudo apt install verilator iverilog`.
- Docker Engine in WSL (Docker Desktop is optional), then pull the OpenROAD-flow-scripts image.

Gotchas:
- Windows PowerShell 5.1 mangles embedded double quotes passed to `wsl -- bash -c '...'`; put multi-step commands in a script and run `wsl -- bash script.sh`.
- `wsl -u root` runs as root without a password prompt; used for apt and Docker setup.

Milestone for Phase 1: one INT8 MAC goes DSLX → Verilog → ORFS on ASAP7 → one `report_power` number. **Reached 2026-10-27** (see README). Full RTL→GDS for `mac_p2` takes about 2 minutes, peak about 0.8 GB RAM (detailed route).

ORFS runs in Docker with the repo mounted read-only at `/xpr` and the run directory at `/work`; outputs land in `~/runs/orfs/<top>/{results,logs,reports}`. `orfs_image.txt` records the image digest used.
