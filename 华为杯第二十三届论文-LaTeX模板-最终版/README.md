# “华为杯”第二十三届中国研究生数学建模竞赛 · LaTeX 论文模板

本模板严格依据当届官方文件制作：**附件3《论文模板》**（封面 4 个标识、题目/摘要/关键词页）与
**附件2《论文格式规范》**（页边距、字体字号、页码、参考文献著录）。使用 **XeLaTeX** 编译。

## 一、编译（务必按顺序）

```bash
xelatex main.tex      # 第 1 遍：生成 main.aux（引用键清单）
bibtex  main          # 读 gmcm.bst 生成 main.bbl（按正文引用顺序编号）
xelatex main.tex      # 第 2 遍：写入参考文献
xelatex main.tex      # 第 3 遍：交叉引用、页码归位
```

- 只有纯文本、**不含参考文献**时才可省略 `bibtex` 那一趟。
- 一键（Git Bash）：`xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex`
- 编译产物（`main.aux/.bbl/.log/.toc/main.pdf`）都不必提交；本目录只保留 `main.pdf` 作编译样张。

## 二、文件结构

| 文件 | 作用 |
|---|---|
| `main.tex` | 论文主文件；四个可直接复制的示例（算法 / 子图 / 长表 / 源码） |
| `huawei-cup.cls` | 版式类：页边距、字体字号、页码、标题、封面、参考文献环境 |
| `gmcm.bst` | 参考文献样式 = GB/T 7714-2015 数字制（`gbt7714-unsrt`），**按引用顺序编号** |
| `reference.bib` | 参考文献库（示例 3 条，写作时必须替换为真实文献） |
| `figures/` | 官方封面 4 个标识（`cpipc.png`、`cpgmcm.png`、`huawei.jpg`、`xjtu-emblem.png`）+ 示例图 |
| `main.pdf` | 本模板编译样张（6 页），用于核对版式 |
| `latex-project.json` | 模板来源与全部源文件 SHA256 清单，请勿删除 |

## 三、已固化的格式要求（对应附件2）

- A4；页边距上 3.00 cm、下 1.75 cm、左 2.25 cm、右 2.25 cm。
- **无页眉**；页码位于**页脚中部**，**从摘要页开始**用阿拉伯数字 “1” 连续编号（封面不显示页码）。
- 论文题目三号黑体；一级标题四号黑体并居中；其余中文小四号宋体，单倍行距。
- 图/表/公式按节编号，输出为 `图5-1`、`表4-1`、`公式(5-1)`。
- 封面 4 个标识不可替换、不可删除；摘要通常不超过两页；除封面外不得出现学校、队号、姓名等信息。
- 参考文献按正文引用次序用方括号编号；书籍须标明页码。

### 内置章节顺序（已按队伍要求排好，不必手改）

一 问题重述 ｜ 二 问题分析 ｜ **三 模型假设与关键符号说明** ｜ **四 数据提取与预处理**
｜ 五 问题一 ｜ 六 问题二 ｜ **七 问题三** ｜ **八 问题四** ｜ 九 模型评价与推广 ｜ 参考文献 ｜ 附录A 程序代码与补充材料

其中第三节把「模型假设」与「关键符号说明」压成两小节；第四节的数据口径是**四问共用输入**，
不要在问题一里重复写数据预处理。

## 四、写作期能力（`main.tex` 内有可直接复制的示例）

| 需求 | 用法 | 示例位置 |
|---|---|---|
| 算法伪代码 | `algorithm` + `algorithmic`（关键字已中文化：输入/输出/当…执行/如果…则/结束…） | 5.2 节「求解算法」 |
| 子图 (a)(b) | `subcaption`：`\begin{subfigure}[b]{0.47\textwidth}` | 5.3 节 图5-1 |
| 跨页长表 | `longtable`：`\endfirsthead` / `\endhead` / `\endlastfoot`，表头自动重复 | 5.3 节 表5-1 |
| 源码附录 | `listings`：`\begin{lstlisting}[language=Python,caption=...,label=...]`，中文注释可正常显示 | 附录A 代码1 |

其他已加载：`xcolor`（配色）、`booktabs`（三线表）、`multirow`（合并单元格）、`siunitx`（单位）、
`natbib`（引用）、`hyperref`（超链接）。

> 算法环境用的是 LaTeX 发行版自带的 `algorithmic.sty` 语法（大写命令：`\REQUIRE \ENSURE \STATE
> \WHILE \ENDWHILE \IF \ELSE \ENDIF \RETURN`），不是 `algpseudocode`。

## 五、参考文献规则（重要）

- 用 BibTeX：正文 `\cite{key}`，文末 `\bibliographystyle{gmcm}` + `\bibliography{reference}`。
- `gmcm.bst` 为 **GB/T 7714 数字制**，编号 = **正文首次引用顺序**（附件2 要求“按引用次序列出”）；
  多人引用写 `\cite{key1,key2}`。
- 著录字段与附件2 三种格式一一对应：书籍用 `author/title/address/publisher/pages/year`，
  期刊用 `author/title/journal/volume/number/pages/year`，网络用 `author/title/url/urldate`。
- 标点采用 GB/T 7714 的半角形式（`, ` `.`），与附件2 描述的中文全角标点等价，属通行做法；
  若评委口径另有要求再手工调整。
- `\begin{HuaweiReferences}` 环境保留为**手写模式备选**（不接 BibTeX 时可用），正式写作建议走 BibTeX。
- 写作要求：只列**真正读过、能定位**的文献；不留占位作者与占位刊名。

## 六、官方模板没有的东西，本模板也不加

经逐词核对官方附件3：**没有目录、没有承诺书、没有编号专用页**；附件2 也要求“首页、摘要页、正文页顺序不能变”。
因此本模板不设目录与承诺书页，请勿自行添加。

## 七、字体与已知限制

- 优先使用 Windows 常见字体 `SimSun`、`SimHei`、`FangSong`、`STXinwei`（封面标题）、`LiSu`（标签）；
  缺字体时自动退回 Fandol 系列，不会编译报错。
- `listings` 的代码字体走 `\ttfamily`，中文注释由 `\setCJKmonofont`（仿宋）渲染。
- 封面“队员姓名”使用 `multirow` 跨三行居中；队员姓名过长时请检查封面是否换行错位。

## 八、版本记录

- 2026-09-26（第二次）：按队伍要求重排章节 —— 原「三 模型假设」「四 符号说明」合并压缩为
  **三、模型假设与关键符号说明**（3.1 模型假设 / 3.2 关键符号说明）；新增 **四、数据提取与预处理**
  （4.1 数据来源与读取方式 / 4.2 清洗口径与指标方向统一 / 4.3 数据划分与泄漏控制，作为四问共用输入）；
  顺次补齐 **七、问题三**、**八、问题四**，**九、模型评价与推广**；参考文献与附录A 不变。
  同时给「问题分析」补了 2.3/2.4 两问的小节。重编译 `xelatex→bibtex→xelatex×2` 通过（6 页，0 错误）。
- 2026-09-26：补齐 5 类写作期宏包（`listings`/`subcaption`/`longtable`/`xcolor`/`algorithm`+`algorithmic`/`multirow`）
  并在 `main.tex` 加入四个可复制示例；参考文献改为 BibTeX（`gmcm.bst`，按引用顺序自动编号）；
  清理无目录时的死代码 `\addcontentsline{toc}`、用 `\@addtoreset` 取代重复的 `\numberwithin`、
  封面改用 `multirow`；重编译 `xelatex→bibtex→xelatex×2` 通过（5 页，0 错误）。
- 2026-09-25：首版（依据附件2/附件3），单文件类 + 主文件 + 官方 4 标识。
