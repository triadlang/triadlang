# 命令行 — 每个命令与示例

## 运行和检查

```sh
./triad run app.tri               # 运行（只安全导入）
./triad run app.tri --unsafe      # 运行（完整 Python）
./triad check app.tri             # 查类型，不执行
./triad fmt app.tri               # 就地格式化
./triad repl                      # 交互草稿本
./triad watch app.tri             # 保存重跑
```

## 理解和测量

```sh
./triad test                      # 项目测试
./triad bench app.tri             # 计时
./triad debug app.tri -b 12       # 12 行断点
./triad tui app.tri               # 交互检查器
./triad jit-stats                 # 热点
./triad docgen src/ -f html       # 从代码出文档
./triad lsp                       # 编辑器支持（服务）
```

## 交付和展示

```sh
./triad compile app.tri --native -o app   # 本地二进制
./triad bundle app.tri -o app.pyz         # 独立 .pyz
./triad serve --port 8000                 # API 服务
./triad play                              # 浏览器里 3D 引擎
./triad plot app.tri --out fig.png        # 运行 + PNG
```

## 物理和记忆

```sh
./triad solve --N 64 --T 2.0 --dim 1      # 演化，不用 .tri
./triad observables run.npy               # 读存过的场
./triad memory status                     # 晶体记忆
./triad memory record "想法"              # 存
./triad memory recall "想法"              # 取
```

## 项目

```sh
./triad init woapp    # 脚手架
./triad setup          # 向导安装
./triad install        # 依赖
./triad publish        # 本地仓库
./triad list           # 已安装
./triad doctor         # 环境报告
```

```
  写 ──► check ──► run ──► bench ──► bundle/compile ──► 交付
            │        │
           fmt     debug/tui/plot
```
