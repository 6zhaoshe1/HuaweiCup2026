\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{huawei-cup}[2026/09/25 v1.0 Huawei Cup postgraduate modeling paper]

\LoadClass[12pt,a4paper,UTF8]{ctexart}

\RequirePackage[a4paper,top=3.00cm,bottom=1.75cm,left=2.25cm,right=2.25cm,
  headheight=0pt,headsep=0pt,footskip=0.85cm]{geometry}
\RequirePackage{fontspec}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{amsmath,amssymb,mathtools}
\RequirePackage{siunitx}
\RequirePackage{enumitem}
\RequirePackage{caption}
\RequirePackage{float}
\RequirePackage{fancyhdr}
% 参考文献：gmcm.bst = GB/T 7714 数字制（gbt7714-unsrt），须配 natbib[numbers]
% 才会输出 [1][2][3] 并按正文首次引用顺序编号（natbib 需在 hyperref 之前加载）。
\RequirePackage[numbers]{natbib}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{etoolbox}

% ===== 写作期能力补充（2026-09-26）=====
% 表格与配色
\RequirePackage{xcolor}
\RequirePackage{multirow}
\RequirePackage{longtable}
% 子图（须在 caption 之后加载；caption 见上方 \RequirePackage）
\RequirePackage{subcaption}
% 源码排版（附录贴关键程序用）
\RequirePackage{listings}
% 算法伪代码（algorithm + algorithmic：MiKTeX 自带 algorithms 宏包组）
\RequirePackage{algorithm}
\RequirePackage{algorithmic}

% 代码清单样式：CJK 等宽走 \setCJKmonofont（FangSong），因此中文注释可正常显示
\lstset{
  basicstyle=\ttfamily\zihao{5},
  numbers=left,
  numberstyle=\tiny\color{black!55},
  numbersep=6pt,
  frame=single,
  rulecolor=\color{black!35},
  framexleftmargin=4pt,
  breaklines=true,
  breakatwhitespace=false,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  tabsize=4,
  keywordstyle=\color{blue!70!black}\bfseries,
  commentstyle=\color{green!45!black},
  stringstyle=\color{red!60!black},
  captionpos=t,
  abovecaptionskip=4pt,
  extendedchars=true,
  escapeinside={(*@}{@*)}
}
\renewcommand{\lstlistingname}{代码}

% 算法环境中文化（algorithmic.sty 的关键字；可按需继续 \renewcommand 扩展）
\floatname{algorithm}{算法}
\renewcommand{\algorithmicrequire}{\textbf{输入：}}
\renewcommand{\algorithmicensure}{\textbf{输出：}}
\renewcommand{\algorithmicif}{\textbf{如果}}
\renewcommand{\algorithmicthen}{\textbf{则}}
\renewcommand{\algorithmicelse}{\textbf{否则}}
\renewcommand{\algorithmicwhile}{\textbf{当}}
\renewcommand{\algorithmicdo}{\textbf{执行}}
\renewcommand{\algorithmicend}{\textbf{结束}}
\renewcommand{\algorithmicfor}{\textbf{对}}
\renewcommand{\algorithmicreturn}{\textbf{返回}}


% 附件2：除论文题目、一级标题外，其余中文使用小四号宋体、单倍行距。
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\IfFontExistsTF{SimSun}
  {\setCJKmainfont{SimSun}[AutoFakeBold=2]}
  {\setCJKmainfont{FandolSong-Regular}[BoldFont=FandolSong-Bold]}
\IfFontExistsTF{SimHei}
  {\setCJKsansfont{SimHei}[AutoFakeBold=2]}
  {\setCJKsansfont{FandolHei-Regular}[BoldFont=FandolHei-Bold]}
\IfFontExistsTF{FangSong}
  {\setCJKmonofont{FangSong}[AutoFakeBold=2]}
  {\setCJKmonofont{FandolFang-Regular}}
\IfFontExistsTF{STXinwei}
  {\newCJKfontfamily\hwtitlefont{STXinwei}[AutoFakeBold=2]}
  {\newCJKfontfamily\hwtitlefont{FandolHei-Regular}[AutoFakeBold=2]}
\IfFontExistsTF{LiSu}
  {\newCJKfontfamily\hwlabelfont{LiSu}[AutoFakeBold=2]}
  {\newCJKfontfamily\hwlabelfont{FandolHei-Regular}[AutoFakeBold=2]}

\zihao{-4}
\linespread{1.0}\selectfont
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setlength{\abovedisplayskip}{6pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{6pt plus 2pt minus 2pt}

% 附件2：无页眉；页码位于页脚中部。
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\zihao{5}\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% 附件2：一级标题四号黑体、居中。论文题目在摘要页单独按三号黑体设置。
\ctexset{
  section={format=\centering\heiti\zihao{4},name={,、},number=\chinese{section},
    beforeskip=12pt,afterskip=6pt,aftername=\hspace{0pt},aftertitle=\par,runin=false},
  subsection={format=\heiti\zihao{-4},name={,},number=\arabic{section}.\arabic{subsection},
    beforeskip=6pt,afterskip=3pt,aftertitle=\par,runin=false},
  subsubsection={format=\heiti\zihao{-4},name={,},number=\arabic{section}.\arabic{subsection}.\arabic{subsubsection},
    beforeskip=3pt,afterskip=0pt,aftertitle=\par,runin=false}
}

\captionsetup{font=small,labelsep=quad}
\captionsetup[figure]{name=图,position=bottom}
\captionsetup[table]{name=表,position=top}
% 图/表/公式按节编号，输出形如 图5-1、表4-1、公式(5-1)。
% \@addtoreset 只负责“随节重置计数”，编号格式由下面的 \renewcommand 统一指定；
% 这样就不必再用 amsmath 的 \numberwithin 重复设置同一件事。
\makeatletter
\@addtoreset{figure}{section}
\@addtoreset{table}{section}
\@addtoreset{equation}{section}
\makeatother
\renewcommand{\thefigure}{\thesection-\arabic{figure}}
\renewcommand{\thetable}{\thesection-\arabic{table}}
\renewcommand{\theequation}{\thesection-\arabic{equation}}

\newcommand{\contestname}{“华为杯”第二十三届中国研究生数学建模竞赛}
\newcommand{\paperschool}{请填写学校名称}
\newcommand{\teamnumber}{请填写参赛队号}
\newcommand{\memberone}{请填写队员一}
\newcommand{\membertwo}{请填写队员二}
\newcommand{\memberthree}{请填写队员三}
\newcommand{\papertitle}{请填写论文题目}
\newcommand{\paperkeywords}{关键词一；关键词二；关键词三}
\providecommand{\keywords}[1]{\renewcommand{\paperkeywords}{#1}}

\newcommand{\HuaweiCover}{%
  \thispagestyle{empty}
  \begin{center}
    \vspace*{1.05cm}
    \begin{tabular}{@{}c@{\hspace{0.45cm}}c@{\hspace{0.45cm}}c@{\hspace{0.45cm}}c@{}}
      \includegraphics[width=3.05cm]{figures/cpipc.png} &
      \includegraphics[width=2.25cm]{figures/cpgmcm.png} &
      \includegraphics[width=2.15cm]{figures/huawei.jpg} &
      \includegraphics[width=2.15cm]{figures/xjtu-emblem.png}
    \end{tabular}

    \vspace{1.45cm}
    {\hwtitlefont\bfseries\zihao{3} 中国研究生创新实践系列大赛\par}
    \vspace{0.55cm}
    {\hwtitlefont\bfseries\zihao{2} “华为杯”第二十三届中国研究生\par}
    \vspace{0.35cm}
    {\hwtitlefont\bfseries\zihao{2} 数学建模竞赛\par}
  \end{center}

  \vspace{2.05cm}
  {\zihao{3}\bfseries
  \renewcommand{\arraystretch}{1.75}
  \begin{tabular}{>{\raggedleft\arraybackslash}p{3.2cm}p{11.0cm}}
    学\quad 校 & \paperschool\\\cline{2-2}
    参赛队号 & \teamnumber\\\cline{2-2}
    \multirow{3}{*}{队员姓名} & 1.\quad \memberone\\\cline{2-2}
      & 2.\quad \membertwo\\\cline{2-2}
      & 3.\quad \memberthree\\\cline{2-2}
  \end{tabular}}
  \clearpage
}

\renewenvironment{abstract}{%
  \setcounter{page}{1}
  \pagestyle{fancy}
  \begin{center}
    {\hwtitlefont\bfseries\zihao{2} “华为杯”第二十三届中国研究生\par}
    \vspace{0.28cm}
    {\hwtitlefont\bfseries\zihao{2} 数学建模竞赛\par}
    \vspace{0.95cm}
    {\heiti\zihao{3}题\quad 目：\papertitle\par}
  \end{center}
  \vspace{0.85cm}
  {\hwlabelfont\zihao{3}\centering 摘\quad 要：\par}
  \vspace{0.3cm}
}{%
  \par\vspace{0.75cm}
  \noindent{\hwlabelfont\zihao{3} 关键词：}{\songti\zihao{-4}\paperkeywords}
  \clearpage
}

\newcommand{\maketitlepage}{\HuaweiCover}

% 备用手写参考文献环境（不接 BibTeX 时可直接用）。
% 正式写作请用 \bibliographystyle{gmcm} + \bibliography{reference}：gmcm = GB/T 7714
% 数字制（gbt7714-unsrt），会按正文引用顺序自动编号，正合附件2“按引用次序列出”。
\newenvironment{HuaweiReferences}[1]
  {\par\addvspace{12pt}
   \noindent\makebox[\textwidth][c]{\heiti\zihao{4}参考文献}\par
   \vspace{6pt}
   \list{[\arabic{enumi}]}{\usecounter{enumi}\setlength{\leftmargin}{2.8em}
     \setlength{\labelwidth}{2.3em}\setlength{\labelsep}{0.5em}
     \setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}}}
  {\endlist}

\endinput
