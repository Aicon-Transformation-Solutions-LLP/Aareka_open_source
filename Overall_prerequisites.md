# What you need to run the Aareka Collector

The **Collector** runs on your machines and reads their telemetry: power, GPU
utilisation and memory, which process or VM is using each GPU, and, if you set it up,
token counts from your inference server. It sends those readings to the **Processor**,
which works out who used the power, energy per token, and GPU waste. The Collector is
read-only: it never changes anything on your machines, and it only makes outbound
connections to your Processor.

You don't need everything below. Start with the basics; add the rest to get more
accurate numbers. The Collector always uses the best source you give it, labels
every number as **measured** (from a real sensor) or **estimated**, and is honest
when something can't be measured — it never makes a number up.

---

## 1. The basics (to run it at all)

- **Python 3.9 or newer.** That's the only hard requirement — the main program uses
  nothing else, so there's nothing to install to get started.
- **Linux is best.** It also runs on Windows, but Linux lets you measure CPU power
  for real (explained below).

---

## 2. To measure GPU power

- Install the **NVIDIA driver** (the one that gives you the `nvidia-smi` command).
- If you can run `nvidia-smi` and see your GPUs, you're done — you get real,
  measured GPU power automatically.

---

## 3. To see GPU waste

Nothing extra. With the NVIDIA driver from section 2, the Collector (version 0.6.0 or
newer) also sends, for every GPU, its **utilisation**, **memory used and total**, and
**model name**. The Processor combines these with the measured power to say whether each
GPU is working, lightly used, or sitting idle, and for how long.

- **Works in local mode** (the Collector running on the GPU machine itself). The vCenter,
  Kubernetes and fleet modes don't send these readings yet, so their GPUs don't appear in
  GPU waste.
- **A reading the GPU doesn't report** is sent as "not available", never as zero.
- **On a Windows desktop or laptop,** ordinary apps that draw their windows on the GPU
  (a browser, a chat app) show up as processes using it, so the GPU can look *held* even
  when no AI workload is running. On a server this doesn't happen.

To check what the Collector will see, run:
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,name --format=csv
```

---

## 4. To measure energy per token

The Collector reads token counts from your inference server's metrics page. It does
not need access to your models, prompts or responses.

- **Your inference server must publish metrics.**
  - vLLM and TGI do this by default.
  - SGLang needs to be started with `--enable-metrics`.
  - llama.cpp needs to be started with `--metrics`.
  - Ollama and LM Studio don't publish token metrics, so they are not supported yet.
- **The Collector must be able to reach that page.** Run it on the same machine as the
  inference server, or on a machine that can reach it over your internal network — for
  example `http://<server>:8000/metrics` for vLLM.
- **If your metrics page needs a login, or only your monitoring system may read it,**
  allow the Collector's machine to read it. The Collector cannot send credentials to the
  metrics page yet.
- **Tell the Collector where to look** by setting `AAREKA_METRICS_URL`, as shown under
  "Add energy per token" in the README Quickstart.

To check it works, open the metrics address in a browser or run:
```bash
curl http://<server>:8000/metrics
```
You should see lines containing `tokens_total`.

---

## 5. To measure CPU power

- **On Linux:** the Collector reads the CPU's built-in energy meter automatically —
  a real measurement. You just need permission to read it (usually root).
- **On Windows:** there is no such meter, so CPU power can only be **estimated**.
  For the estimate, install one small package:
  ```bash
  pip install psutil
  ```
  Without it, CPU power is simply left out — never guessed.

---

## 6. For the most accurate whole-machine number (optional, but best)

Servers usually have a small management chip that knows the machine's *total* power
from the wall — including fans, disks, and power-supply loss. It goes by names like
**BMC, iDRAC, iLO, Redfish, or IPMI**.

- Give the Collector **read-only access** to it (its network address and a read-only
  login) and you get the exact whole-machine number.
- Without it, the Collector still works — it just says *"the parts I could measure
  add up to at least X watts; the rest is estimated,"* instead of inventing the
  missing piece.

---

## 7. If you run Kubernetes or VMware (optional)

To automatically match power to your pods or VMs, install the matching helper:

```bash
pip install kubernetes     # for Kubernetes
pip install pyVmomi        # for VMware / vCenter
```

You'll also need normal read-only access — your Kubernetes cluster config, or a
read-only vCenter login.

---

## 8. Try it right now

Measure the machine you're on:
```bash
python imp_server_power_source.py
```

Watch its power live, second by second:
```bash
python imp_live.py
```

---

## 9. Collecting across a whole estate — vCenter, Kubernetes, or a fleet (beta)

These modes are **beta**: built and tested against fixtures, and validated against your
real environment on first connect. Pick one with `AAREKA_SOURCE`. They measure power and
ownership; GPU waste is not available in these modes yet (section 3).

**VMware / vCenter** (`AAREKA_SOURCE=vcenter`) — one machine reaches vCenter; nothing to
install per host.
```bash
pip install pyvmomi
export AAREKA_SOURCE=vcenter
export AAREKA_VCENTER=vcenter.your.org
export AAREKA_VCENTER_USER='readonly@vsphere.local'
export AAREKA_VCENTER_PASSWORD='...'
export AAREKA_VCENTER_DEPT_ATTR='Department'   # vCenter custom attribute holding the BU
# export AAREKA_VCENTER_INSECURE=1             # only for self-signed lab certs
```
Needs: a **read-only** vCenter account and network access to the vCenter API (443).

**Kubernetes** (`AAREKA_SOURCE=kubernetes`) — one machine with a kubeconfig, or run in-cluster.
```bash
pip install kubernetes
export AAREKA_SOURCE=kubernetes
export AAREKA_KUBECONFIG=~/.kube/config        # omit to use in-cluster credentials
export AAREKA_PROM_URL=http://prometheus:9090  # power from dcgm-exporter / node-exporter
# export AAREKA_PROM_NODE_LABEL=Hostname       # metric label carrying the node name
```
Needs: **read-only** cluster access, and a Prometheus scraping dcgm-exporter (GPU) +
node-exporter (CPU).

**Fleet** (`AAREKA_SOURCE=fleet`) — one control node sweeps many servers and labels each whole
server with its business unit from a map your IT team supplies.
```bash
export AAREKA_SOURCE=fleet
export AAREKA_INVENTORY=inventory.json          # [{"host","rack","transport":"ssh|local","prom_url"/"bmc"/...}]
export AAREKA_HOST_DEPT_MAP='{"gpu-a":"Research","gpu-b":"Finance"}'   # server -> BU (from IT)
```
Needs: read-only SSH to each server (or Prometheus/BMC endpoints) and the server→BU map.

---

## In short

- **Just want to try it?** Python + the NVIDIA driver. Nothing else.
- **Want to see idle GPUs?** Nothing extra in local mode (section 3).
- **Want energy per token?** Your inference server's metrics page, reachable from the
  Collector (section 4).
- **Want exact whole-machine numbers?** Add read-only access to the server's
  management chip (BMC/IPMI).
- **Want it mapped to pods/VMs?** Add the Kubernetes or VMware helper.
- **Whole estate at once?** vCenter, Kubernetes, or fleet mode — section 9 (beta).

Everything it can't measure is shown as an estimate and clearly labeled — never
disguised as a real reading.
