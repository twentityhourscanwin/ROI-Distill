# Cursor 模型可用性与网络诊断问题总结

> 文档说明：本文档整理自 2026-06-18 关于 Cursor IDE 网络诊断、模型列表（GPT / Claude / Opus）不可用及代理配置的讨论，可直接复制粘贴至飞书文档。

---

## 一、问题背景

在使用 Cursor IDE 时遇到以下现象：

1. **网络诊断全部通过**（`Result: true`），包括 Ping、streamSSE、stream 等测试；
2. **模型列表中找不到 GPT、Claude Code、Opus 等模型**，只能看到少量模型（如 Composer、Grok、Kimi 等）；
3. 已配置梯子（VPN/代理），但模型仍不可见；
4. **某次突然又能使用 GPT、Claude 等模型**，行为不稳定。

典型网络诊断日志示例：

```
[2026-06-18T11:18:43.449Z] Sending ping 1
[2026-06-18T11:18:44.543Z] Response: 'ping' in 1094ms
...
[2026-06-18T11:18:48.346Z] Result: true

[2026-06-18T11:18:43.450Z] Starting streamSSE
[2026-06-18T11:18:44.427Z] Response: 'foo' in 974ms
...
[2026-06-18T11:18:50.233Z] Result: true

[2026-06-18T11:18:43.452Z] Starting stream
[2026-06-18T11:18:44.980Z] Response: 'foo' in 1527ms
...
（stream 延迟约 1.5–2.1 秒）
```

---

## 二、核心结论（一句话）

**网络诊断通过 ≠ 可以使用 GPT/Claude/Opus。**

- 网络诊断只验证：能否连通 Cursor API、流式通道是否正常；
- 模型是否出现在列表中：由 **Cursor 服务端根据当前出口 IP 所在地区** 过滤，与网络诊断结果无关。

---

## 三、网络诊断各项含义

| 测试项 | 通过说明什么 | 不说明什么 |
|--------|--------------|------------|
| **Ping** | 能连上 `api2.cursor.sh`，延迟约 925–1094ms（偏高但可连通） | 不验证账号订阅、不验证地区 |
| **streamSSE** | HTTP/1.1 单向流式（Chat）可用 | 不验证 GPT/Claude 是否对该 IP 开放 |
| **stream** | HTTP/2 双向流式（Agent）可用 | 不验证模型目录权限 |

**注意**：国内 IP 同样可能 Ping/stream 全绿，但 GPT/Claude 仍会被隐藏。

---

## 四、为什么模型列表里没有 GPT / Claude / Opus

### 4.1 地区限制（主因）

OpenAI（GPT）、Anthropic（Claude/Opus）、Google（Gemini）对 API 有 **地区访问限制**。Cursor 会在把模型列表下发给客户端 **之前**，按请求来源 IP 在服务端过滤。

官方文档：[Regions | Cursor Docs](https://cursor.com/docs/account/regions)

> 部分模型提供商有地区限制，受限地区的模型不会出现在 Cursor 中，但其他可用模型仍可正常使用。

在受限地区（如中国大陆 IP），个人版用户通常只能看到：

- Cursor 自有模型（Composer / Premium）
- 未受同等限制的其他提供商模型（如 Grok、Kimi 等）

**这与 Pro 订阅是否生效无关**——订阅决定「有没有权限用」，地区限制决定「该地区能不能用」。

### 4.2 与订阅、配额的关系

| 情况 | 表现 |
|------|------|
| 地区受限 | 模型 **直接不在列表里** |
| 订阅不足 / 配额用尽 | 模型在列表里，使用时报错或提示限额 |

本次问题属于 **地区/出口 IP** 类，而非订阅类。

---

## 五、为什么挂了梯子仍然不行

### 5.1 Cursor 默认可能不走系统代理（最高频原因）

Cursor 基于 Electron，代理行为与浏览器不同：

| 设置 | 默认值 | 实际效果 |
|------|--------|----------|
| `Http: Proxy Support` | **`override`** | 仅使用 `settings.json` 中的 `http.proxy` |
| `http.proxy` | **空** | 等于不使用任何代理 |

典型结果：

```
浏览器  →  走 Clash/系统代理  →  ipinfo 显示海外 IP  ✅
Cursor  →  直连国内网络      →  服务端看到国内 IP  ❌ 模型被过滤
网络诊断 →  国内也能连通 API   →  Result: true       ✅
```

### 5.2 其他常见原因

1. **仅开系统代理，未开 TUN**：Electron 应用可能不跟随系统代理，需 TUN 模式或显式配置 `http.proxy`；
2. **Clash 规则把 `*.cursor.sh` 走了 DIRECT**：Cursor 流量绕过梯子，仍用国内 IP；
3. **浏览器 IP ≠ Cursor IP**：在浏览器查 ipinfo 不能代表 Cursor 的出口 IP；
4. **节点仍在受限地区**：部分节点（如香港/台湾等）可能仍被过滤，建议美国/日本/新加坡等；
5. **云端 IDE 环境**：若代码跑在阿里云等国内服务器上，本机梯子不影响云端出口 IP。

### 5.3 推荐配置

**方法 A：让 Cursor 跟随系统代理**

1. `Ctrl+,` → 搜索 `Proxy Support` → 改为 **`on`** 或 **`fallback`**
2. 搜索 `HTTP/2` → 勾选 **Disable HTTP/2**
3. **完全退出** Cursor 后重新打开

**方法 B：显式填写代理（Clash 示例）**

`Ctrl+Shift+P` → `Open User Settings (JSON)`：

```json
{
  "http.proxySupport": "on",
  "http.proxy": "http://127.0.0.1:7890",
  "https.proxy": "http://127.0.0.1:7890",
  "http.proxyStrictSSL": false,
  "cursor.general.disableHttp2": true
}
```

> 端口改为 Clash/V2Ray 的 **HTTP 代理端口**（非 SOCKS 端口，除非另行配置）。

**Clash 规则建议**

- `*.cursor.sh`、`*.cursor.com`、`*.cursorapi.com` 等应 **走代理节点**
- 不要将这些域名设为 DIRECT（除非仅为修复 streaming 缓冲，但会加重地区限制）

---

## 六、为什么「突然又能用」GPT / Claude

模型列表是 **实时按当前出口 IP** 过滤的，因此会出现时有时无：

| 可能原因 | 说明 |
|----------|------|
| 代理路由终于生效 | 重启 Cursor、开 TUN、改 Proxy Support 后，Cursor 开始走海外节点 |
| VPN 节点自动切换 | 从美国/日本节点切到国内节点（或相反）会立即影响模型列表 |
| 网络出口变化 | 换 WiFi、换热点、代理重连导致 IP 变化 |
| 先隐藏后显示 | 与「刚安装时暂时显示、首次请求失败后隐藏」相反，IP 变海外后会重新显示 |

**再次强调**：不是网络诊断变了，而是 **Cursor 服务端看到的 IP 地区变了**。

---

## 七、排查清单

### 7.1 确认问题类型

- [ ] 模型 **不在列表里** → 地区/IP 问题
- [ ] 模型在列表里但 **报错/超时** → 网络流式/代理/HTTP2 问题
- [ ] 提示 **订阅/配额** → 账号套餐问题

### 7.2 代理与 Cursor 配置

- [ ] `Http: Proxy Support` 是否为 `on` 或 `fallback`（非默认 `override`）
- [ ] 是否显式配置了 `http.proxy` / `https.proxy`
- [ ] 是否开启 **Disable HTTP/2**
- [ ] Clash 是否开启 **TUN 模式**
- [ ] Clash 日志中 `api2.cursor.sh` 是否走 **PROXY**（非 DIRECT）
- [ ] 节点是否为 **美国/日本/新加坡** 等支持地区

### 7.3 环境确认

- [ ] 使用的是 **本机 Desktop Cursor** 还是 **云端/SSH Remote IDE**
- [ ] 若为云端 IDE，本机梯子 **无法** 改变模型列表，需云端出口代理或改用本机 Cursor

### 7.4 验证方式

| 检查方式 | 能验证什么 |
|----------|------------|
| 浏览器打开 ipinfo.io | 浏览器是否走梯子 |
| Cursor 网络诊断 | API 是否连通（**不显示 IP 地区**） |
| Clash 连接日志 | Cursor 请求是否走代理、出口节点 |
| 模型列表是否出现 GPT/Claude | Cursor 实际出口 IP 是否在支持地区 |

---

## 八、逻辑关系图

```
                    ┌─────────────────────┐
                    │   用户打开 Cursor    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │   网络诊断        │              │   拉取模型列表    │
    │ Ping/stream/SSE  │              │ 按出口 IP 过滤    │
    └────────┬─────────┘              └────────┬─────────┘
              │                                 │
              ▼                                 ▼
    只测 API 是否连通                   国内 IP → 隐藏 GPT/Claude
    国内/海外都可能 true                海外 IP → 显示 GPT/Claude
              │                                 │
              └──────── 两者独立，互不等价 ────────┘
```

---

## 九、参考链接

- [Cursor Regions 官方文档](https://cursor.com/docs/account/regions)
- [Cursor Network Configuration](https://docs.anyweb.dev/docs/enterprise/network-configuration)
- Cursor 社区论坛相关讨论：地区限制、Disable HTTP/2、`Http: Proxy Support` 默认值问题

---

## 十、后续建议

1. 按 **第七节排查清单** 固定代理配置，避免节点自动切换导致模型时有时无；
2. 模型稳定可用时，说明 Cursor 请求已 **稳定走海外出口**；若再次消失，优先查 Clash 日志与节点；
3. 长期在国内云端开发时，考虑 **本机 Desktop Cursor + SSH** 或给云环境配置 **稳定海外出口代理**。

---

*文档生成时间：2026-06-18*
