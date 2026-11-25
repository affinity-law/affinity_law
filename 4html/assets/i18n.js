'use strict';

// Presentation-only localization. Dataset values, Python source, and raw logs stay intact.
window.ModelI18n = (() => {
  const messages = {
    "调参预计需要 0.5–2 小时，实际耗时取决于数据量与硬件。":"Estimated tuning time is 0.5–2 hours, depending on data size and hardware.",
    "重新填写 DeepSeek API key":"Change DeepSeek API key",
    "DeepSeek 调参设置":"DeepSeek Tuning Setup",
    "填写自己的 DeepSeek API key，供本次调参使用。":"Enter your own DeepSeek API key for this tuning run.",
    "粘贴 DeepSeek API key":"Paste your DeepSeek API key",
    "密钥仅用于本次任务，不写入项目文件或浏览器存储。调参会向 DeepSeek 发送模型配置、特征摘要与验证结果，API 费用由你的账户承担。":"The key is used for this task and kept out of project files and browser storage. Tuning sends model settings, feature summaries and validation results to DeepSeek. API usage is billed to your account.",
    "选择自动调参后，在弹窗中填写自己的 DeepSeek API key。":"Select Automatic Tuning and enter your own DeepSeek API key in the dialog.",
    "请输入有效的 DeepSeek API key，不要包含空格或换行。":"Enter a valid DeepSeek API key without spaces or line breaks.",
    "确认":"Confirm",
    "取消":"Cancel",

    '时序工坊 · 规律驱动的模型构建':'affinity law | Model Workspace',
    '时序工坊首页':'affinity law home', '时序工坊':'affinity law',
    '跳到主要内容':'Skip to main content', '主导航':'Main navigation',
    '模型工作台':'Model Workspace', '资料与帮助':'Resources & Help', '模块库':'Module Library',
    '规律与证据':'Rules & Evidence', '使用指南':'User Guide', '本地工作台':'Local Workspace',
    '科研工作者':'Researcher', '研究空间':'Research Workspace', '研究协议已固定':'Research protocol preserved',
    '快速上手':'Quick Start', '从数据，走向':'From Data to ', '可解释的预测。':'Interpretable Forecasts',
    '导入数据，一次运行即可获得模型与预测效果。':'Import your data to build a model and evaluate its forecasts in one run.',
    '建模步骤':'Modeling workflow', '数据与设置':'Data & Settings', '自动建模':'Build & Evaluate',
    '模型结果':'Results', '准备本次建模':'Set Up a Model',
    '选择数据与预测设置，然后开始建模。':'Select a dataset and forecasting settings, then start modeling.',
    '支持单变量与多变量':'Univariate and multivariate data', '导入数据':'Import Data', '本次研究':'Current study',
    '将时间序列文件拖到这里':'Drop a time series file here',
    '支持 .npy、.csv · 每行一个时间点，每列一个变量':'NPY or CSV · Rows are time steps; columns are variables',
    '选择本地文件':'Choose File', '最大 50 MB · 按原始时间顺序读取':'Up to 50 MB · Original time order is preserved',
    'CSV 可有表头和首列时间戳，变量列请使用无缺失的数值。':'CSV may include a header and a timestamp column. Variable columns must contain numeric values without missing entries.',
    '模型设置':'Model Settings', '预测未来多少步':'Forecast Horizon', '是否调参':'Hyperparameter Tuning',
    '不调参':'Default Parameters', '自动调参':'Automatic Tuning',
    '使用默认参数，训练并评估':'Train and evaluate with default settings',
    '优化参数后评估，耗时更长':'Optimize, then evaluate; requires more time',
    '开始建模':'Start Modeling', '运行设置说明':'Run Settings',
    '默认输入 12 步，按 60% / 20% / 20% 划分数据。默认参数只训练评估一次（种子 2026），最多 300 轮、早停 15。':'The default input window is 12 steps, with a 60% / 20% / 20% train/validation/test split. Default evaluation uses one run (seed 2026), up to 300 epochs, and an early-stopping patience of 15.',
    '调参使用原有参数空间，输入窗口可为 12 / 32 / 64 步；需要本机配置 DeepSeek 密钥。内部候选筛选沿用原协议；最终模型展示种子 2026 的单次评估。原最终推荐涉及测试集比较。':'Tuning uses the existing search space, including input windows of 12, 32, or 64 steps. A local DeepSeek API key is required. Candidate selection follows the original protocol, which includes test-set comparisons. The final model is reported using seed 2026.',
    '没有数据？使用项目示例':'Try an Example Dataset', '建模结果':'Modeling Results',
    '下载模型 .py':'Download Model (.py)', '查看模型代码与使用方式':'Model Code & Usage',
    '复制代码':'Copy Code', '查看数据特征与推荐依据':'Data Features & Recommendation Rationale',
    '关键数据特征':'Key Data Features', '推荐理由':'Why These Modules', '下载完整报告':'Download Full Report',
    '开始新的建模':'New Modeling Run', '原项目的 28 个模块与组合范式，按建模职责组织。':'Explore 28 modules and composition patterns, organized by their modeling roles.',
    '直接读取项目部署 Law Table 的 28 条规律，保留原始条件与统计证据。':'Browse 28 rules from the project’s deployed law table, with their original conditions and statistical evidence.',
    '全部阶段':'All stages', '全部等级':'All grades', '全部模块':'All Modules',
    '搜索模块、特征或条件…':'Search modules, features, or conditions…', '搜索规律':'Search rules',
    '筛选阶段':'Filter by stage', '筛选证据等级':'Filter by evidence grade',
    '来源：2search/1search/output/08_cookbook_law_table_deploy.csv · 仅使用项目已固定的部署规则。':'Source: 2search/1search/output/08_cookbook_law_table_deploy.csv · Uses the project’s fixed deployment rules.',
    '从导入到导出':'From Data to Model', '让数据、规律、模型与运行设置始终可追溯。':'Keep datasets, rules, models, and run settings traceable.',
    '准备时间序列':'Prepare Your Time Series',
    'NPY 支持 [时间点] 或 [时间点, 变量] 的数值数组。CSV 可有表头与首列时间戳；不支持缺失值、无穷值和非数值变量。网页不会悄悄插值或重排时间。':'NPY accepts numeric arrays shaped as [time] or [time, variables]. CSV may include a header and a first-column timestamp. Missing, infinite, and nonnumeric values are not supported. The app does not interpolate values or reorder observations.',
    '了解运行模式':'Understand the Workflow',
    '历史示例直接展示项目已有特征与原选择器的规律初选，无需后端。新文件发送到计算服务后，调用原特征计算、候选生成和验证选优入口；计算完成前不会生成研究结论。':'Example previews use existing project features and rule-based selections. For uploaded data, the local service computes features, generates candidates, and selects a model using the original validation workflow.',
    '查看推荐依据':'Review the Rationale',
    '只采用 S/A 级规律，排除反偏好角，再按等级和 lift_ci_lo 排序。A/B/C 未命中使用发现集频率默认，E 未命中使用 Single Block。D 逆变换从 A 自动推导。':'The selector uses S/A-grade rules, excludes counter-preference regions, and ranks rules by grade and lift_ci_lo. Unmatched A/B/C stages use discovery-set frequency defaults; stage E defaults to Single Block. Stage D inverses are derived from stage A.',
    '一次开始，自动完成':'Start Once, Run Automatically',
    '在首页选择预测步数与是否调参，点击“开始建模”。工作台会自动衔接特征计算、模块选择、训练与评估，完成后打开结果页。':'Choose the forecast horizon and tuning mode, then select Start Modeling. Feature extraction, module selection, training, and evaluation run automatically. Results appear when the run finishes.',
    '复用模型与报告':'Reuse Models and Reports',
    '下载完整 .py 文件与包含特征、命中规则、架构和配置的 JSON 报告。源码需要项目原有 PyTorch 环境；默认源码不包含权重，不代表已完成训练。':'Download the Python model and a JSON report containing features, matched rules, architecture, and settings. Run the code in the project’s PyTorch environment. Source code does not include trained weights; evaluation status is shown separately.',
    '在本机启动工作台':'Launch Locally',
    '从 GitHub 下载完整项目，按根目录说明搭建 Python 环境，再运行 python -B 4html/start.py，浏览器会自动打开。前端与原计算流程在同一台电脑运行，无需填写服务地址。':'Download the complete project from GitHub, set up Python as described in the root README, and run python -B 4html/start.py. The browser opens automatically. The interface and computation run on the same computer.',
    '查看本地运行状态':'Check Local Runtime',
    '网页与连接层放在 4html，运行文件写入 4html/runs。项目原模型、算法、实验数据、历史结果和参数设置均保留。':'The interface and local adapter are in 4html; run artifacts are saved in 4html/runs. The original models, algorithms, datasets, historical results, and configuration files are preserved.',
    '让模型选择可解释，让研究可以复现。':'Interpretable model selection. Reproducible research.',
    '本地运行状态':'Local Runtime Status', '关闭':'Close',
    '在完整项目的根目录，激活已安装依赖的 Python 环境后运行：':'Activate the project’s Python environment, then run this command from the project root:',
    '浏览器会自动打开本机工作台。保持启动终端运行，按 Ctrl+C 停止。单独双击 HTML 可以体验示例，实际计算需要此本地进程。':'The local workspace opens in your browser. Keep the terminal running; press Ctrl+C to stop. Opening the HTML file directly supports example previews only; computation requires the local service.',
    '重新检查环境':'Check Again', '移除文件':'Remove file', '等待发送至计算服务校验':'Awaiting validation by the local service',
    '分布形态':'Distribution Shape', '趋势强度':'Trend Strength', '平稳性':'Stationarity', '周期性':'Periodicity',
    '时间依赖':'Temporal Dependence', '噪声波动':'Noise & Volatility', '多变量结构':'Multivariate Structure',
    '输入处理':'Preprocessing', '结构建模':'Backbone', '预测生成':'Prediction Head', '输出还原':'Output Restoration', '组合范式':'Composition',
    '保持输入不变，不额外引入预处理。':'Passes the input through without additional preprocessing.',
    '按窗口逐通道标准化，预测后逆变换回原始尺度。':'Standardizes each channel within the input window and restores predictions to the original scale.',
    '逐样本、逐变量可逆归一化，并使用可学习仿射变换。':'Applies reversible instance normalization per variable with a learnable affine transformation.',
    '使用一阶差分表达局部变化，输出端通过累加还原。':'Models local changes using first-order differencing and reconstructs outputs by cumulative summation.',
    '分离趋势与残差，骨干处理残差，输出时重组趋势。':'Separates trend and residual components, models the residuals, and restores the trend at the output.',
    '对重叠时间片段进行局部线性变换，再恢复原序列长度。':'Applies local linear transformations to overlapping patches and restores the original sequence length.',
    '通过项目定义的掩码处理输入位置与变量。':'Applies the project’s masks to input positions and variables.',
    '使用线性变换建模时间维度映射。':'Models temporal mappings with linear transformations.',
    '利用多层感知机表示非线性特征映射。':'Models nonlinear feature mappings with a multilayer perceptron.',
    '使用一维卷积提取局部时间模式。':'Extracts local temporal patterns with one-dimensional convolutions.',
    '通过膨胀因果卷积建模跨时间步的依赖。':'Captures temporal dependencies using dilated causal convolutions.',
    '使用门控循环单元更新时序隐状态。':'Updates temporal hidden states using gated recurrent units.',
    '使用带记忆单元的门控递归更新时序状态。':'Models temporal states using gated recurrence and memory cells.',
    '通过状态空间递推表示序列动态变化。':'Represents sequence dynamics through state-space recurrence.',
    '用自注意力与前馈层编码时间序列表征。':'Encodes time series using self-attention and feed-forward layers.',
    '通过基函数组合表示时间序列变化。':'Represents temporal variation as a combination of basis functions.',
    '将隐藏表征线性投影为指定预测长度的输出。':'Linearly projects hidden representations to the specified forecast horizon.',
    '通过多层感知机将隐藏表征映射为预测序列。':'Maps hidden representations to forecasts using a multilayer perceptron.',
    '将系数投影到基函数上生成预测。':'Generates forecasts by projecting coefficients onto basis functions.',
    '恢复 Z-score 标准化之前的数值尺度。':'Restores the scale used before Z-score standardization.',
    '恢复 RevIN 归一化之前的数值尺度。':'Restores the scale used before RevIN normalization.',
    '从差分预测逐步累加，重建原始数值。':'Reconstructs original values by cumulatively summing differenced forecasts.',
    '将预测分量与趋势重新组合为最终输出。':'Combines the predicted components and trend to produce the final output.',
    '只使用一个骨干模块进行建模。':'Uses a single backbone module.',
    '将两个不同骨干模块依次连接。':'Connects two distinct backbone modules in sequence.',
    '两个骨干并行处理，再将分支输出相加。':'Runs two backbones in parallel and adds their outputs.',
    '主分支与校正分支分别处理输入，再将输出相加。':'Processes the input through main and correction branches, then adds their outputs.',
    '使用可学习门控融合两个骨干的输出。':'Combines two backbone outputs using learnable gates.',
    '匹配当前预测长度的推荐规律。':'Matches a recommendation rule for the selected forecast horizon. ',
    '根据组合需要补充该模块。':'Included to complete the selected composition. ',
    '沿用项目默认组合。':'Uses the project’s default composition. ',
    '由项目默认配置或候选选优确定。':'Selected through the project’s default or candidate-validation workflow. ',
    '模型与评估结果已保存，可以直接下载使用。':'Your model and evaluation results are saved and ready to download.',
    '模型代码已生成，尚未完成训练评估。':'Model code is ready. Training and evaluation have not yet completed.',
    '输入步数':'Input Steps', '输出步数':'Forecast Steps', '变量数量':'Variables', '参数状态':'Parameter Status',
    '完成调参':'Tuned', '默认参数':'Default Parameters', '源码行数':'Lines of Code',
    '下载为调参后推荐源码。训练权重与完整记录保存在本机 4html/runs 目录。':'Downloads the recommended model after tuning. Trained weights and complete records are saved locally in 4html/runs.',
    '已完成默认参数评估。源码不包含权重；权重与预测数据保存在本机 4html/runs。':'Default-parameter evaluation is complete. Source code excludes weights; weights and predictions are saved locally in 4html/runs.',
    '默认导出为未训练架构源码，不包含训练权重。':'Exports the untrained architecture source without trained weights.',
    '模型结构':'Model Architecture', '下载结构图':'Download Diagram', '此历史结果未保存结构图。':'No architecture diagram was saved for this historical result.',
    '结果已保存到独立文件夹':'Results saved in a dedicated run folder', '复制文件夹路径':'Copy Folder Path',
    '查看保存位置与文件':'Output Location & Files', '本次结果文件夹路径':'Run output folder',
    '完整训练与预测文件保留在同一目录的子文件夹中。':'Complete training and prediction artifacts are stored in subfolders of this run directory.',
    '当前仅预览模型。启动本地服务后可自动计算并保存结果。':'Model preview only. Start the local service to run computations and save results.',
    '已复制':'Copied', '请按 Ctrl+C':'Press Ctrl+C', '模型效果':'Model Performance', '测试集评估':'Test Set Evaluation',
    '仅模型代码':'Model Code Only', '此结果尚未完成训练评估。':'Training and evaluation have not been completed for this result.',
    '训练并查看效果':'Train & Evaluate', '均方根误差':'Root Mean Squared Error', '平均绝对误差':'Mean Absolute Error',
    '加权绝对百分比误差':'Weighted Mean Absolute Percentage Error', '查看评估详情':'Evaluation Details',
    '指标来自一次训练评估':'Metrics are from a single training and evaluation run',
    '原始数据尺度，重叠窗口预测先平均后评分。WMAPE 显示百分比，报告保留比例。图表的变量或范围不影响上方指标。':'Metrics are computed in the original data scale after averaging overlapping-window predictions. WMAPE is displayed as a percentage and stored as a ratio. Chart selections do not change the overall metrics.',
    '最终推荐沿用原协议中的测试集比较。':'The final recommendation follows the original protocol’s test-set comparison.',
    '随机种子':'Random Seed', '标准差':'Standard Deviation', '本次结果链接（可收藏）':'Permalink to This Run',
    '真实值与预测值':'Observed vs. Predicted', '真实值':'Observed', '预测值':'Predicted',
    '预测图变量':'Prediction chart variable', '图表设置':'Chart Settings', '预测图随机种子':'Prediction chart seed',
    '每段显示最多 600 个连续测试点。':'Each segment displays up to 600 consecutive test points.',
    '下载对比图':'Download Plot', '正在读取预测曲线…':'Loading predictions…', '上一段':'Previous', '下一段':'Next',
    '正在读取该种子与变量的实际预测…':'Loading predictions for the selected seed and variable…',
    '移动到曲线上查看对应行的真实值与预测值。':'Hover over the plot to inspect observed and predicted values.',
    '未能读取曲线。指标已保留，可重新进入此步骤重试。':'Unable to load the plot. Metrics are preserved; reopen the result to retry.',
    '图片生成失败，请重试。':'Unable to generate the plot. Please retry.',
    '原序列行号（从 1 开始）':'Original Data Row (1-based)', '原始数值':'Value (Original Scale)',
    '当前架构 · 默认参数单次评估':'Current Architecture · Single-Run Default Evaluation',
    '当前架构 · 默认参数三种子测试':'Current Architecture · Historical Three-Seed Evaluation',
    '最终推荐：默认参数 · 单次评估':'Recommended Model: Default Parameters · Single-Run Evaluation',
    '最终推荐：调参配置 · 单次评估':'Recommended Model: Tuned Parameters · Single-Run Evaluation',
    '正在准备本次建模':'Preparing the modeling run', '正在准备建模':'Preparing Model',
    '正在训练与评估':'Training & Evaluating', '正在保存结果':'Saving Results', '正在优化模型':'Tuning Hyperparameters',
    '正在分析数据与选择模型':'Analyzing Data & Selecting Modules', '正在自动建模':'Modeling in progress',
    '本次建模未完成':'Modeling Incomplete', '本次建模已取消':'Modeling Cancelled',
    '完成后自动展示结果。关闭页面后，计算仍会在本机继续。':'Results will appear automatically. Computation continues locally if you close this page.',
    '查看运行详情':'Execution Details', '取消本次建模':'Cancel Run', '评估尚未完成':'Evaluation Incomplete',
    '本地计算已就绪':'Local Runtime Ready', '本地环境 · 依赖待安装':'Dependencies Required',
    '仅预览 · 未启动计算':'Preview Only · Runtime Offline', '正在检查本机运行环境…':'Checking the local runtime…',
    '原项目计算入口已就绪。':'The local computation service is ready.', '固定架构调参已就绪。':'Fixed-architecture tuning is ready.',
    '调参尚未就绪：':'Tuning is not ready: ', '请在本机环境配置 DEEPSEEK_API_KEY。':'Set DEEPSEEK_API_KEY in your local environment.',
    '计算期间无法更改预测长度。':'The forecast horizon cannot be changed during a run.',
    '计算中暂时不能移除文件。':'The dataset cannot be removed during a run.',
    '当前计算未结束，暂时不能替换文件。':'Wait for the current run to finish before replacing the dataset.',
    '请选择 .npy 或 .csv 格式的时间序列文件。':'Select a time series file in NPY or CSV format.',
    '文件为空，请检查后重新选择。':'The file is empty. Select a file containing data.',
    '文件超过 50 MB，请选择符合大小限制的数据。':'The file exceeds 50 MB. Select a smaller dataset.',
    '每次研究请选择一个时间序列文件。':'Select one time series file per run.',
    '没有匹配的规律，试试其他关键词或筛选条件。':'No matching rules. Try a different keyword or filter.',
    '该地址不是时序工坊计算服务。':'This address does not host the Time Series Lab service.',
    '本地计算服务未返回有效结果，请确认是从 start.py 打开的网页。':'The local service returned an invalid response. Open the workspace using start.py.',
    '请先在项目环境中运行 python -B 4html/start.py，再使用自动打开的本地网页计算数据。':'Run python -B 4html/start.py in the project environment, then use the local workspace that opens automatically.',
    '服务返回的模型结果不完整。':'The service returned incomplete model results.', '任务已停止。':'The run has stopped.',
    '覆盖数据集':'Datasets Covered', 'Lift 区间':'Lift Interval', '覆盖率':'Coverage', 'q 值':'q-value',
    '反偏好角：无':'Counter-preference region: None', '反偏好角：':'Counter-preference region: ', '原始条件：':'Original condition: ', '模块作用：':'Module role: ',
    '沿用原项目定义':'Uses the original project definition', '输出端配套执行逆变换':'Applies the corresponding inverse transformation at the output',
    'A · 输入处理':'A · Preprocess', 'B · 骨干':'B · Backbone', 'C · 预测头':'C · Head', 'D · 输出还原':'D · Restore',
    '输入序列':'Input Sequence', '预测输出':'Forecast', '主分支':'Main Branch', '校正分支':'Correction',
    '门控融合':'Gated Fusion', '相加':'Add', '张量形状：':'Tensor shapes: ', '模块连接沿用原组装器':'Connections follow the original assembler',
    'ETTh2 · 电力变压器':'ETTh2 · Electricity Transformer', 'Capital Bikeshare · 单车需求':'Capital Bikeshare · Bike Demand',
    'Mauna Loa · CO₂ 浓度':'Mauna Loa · CO₂ Concentration', '多变量 / 能源':'Multivariate / Energy',
    '单变量 / 交通':'Univariate / Transport', '单变量 / 环境':'Univariate / Environment',
    '输入张量':'Input tensor', '输入':'Input', '输出':'Output', '变量':'Variable', '误差':'Error',
    '步':'steps', '无':'None', '变量数':'Variables', '研':'R',
    '同分布通用强证据':'Strong general evidence within the same distribution',
    '同分布稳健条件证据':'Robust conditional evidence within the same distribution',
    '跨格通用':'Applies across strata', '部分格子稳定':'Stable in selected strata', '纯净':'No identified counter-preference region',
    '基于数据特征与研究规律的模块化时间序列预测工作台。导入数据、查看推荐依据、固定架构调参、导出 PyTorch 模型。':'A modular time series forecasting workspace for data import, rule-based model selection, fixed-architecture tuning, and PyTorch model export.'
  };
  Object.assign(messages, {
    '正在准备项目计算环境':'Preparing the local computation environment',
    '正在保存本次模型与报告':'Saving the model and report',
    '正在准备当前架构的默认参数评估':'Preparing default-parameter evaluation',
    '正在准备固定架构调参输入':'Preparing inputs for fixed-architecture tuning',
    '正在校验数据格式与时间划分':'Validating the data format and temporal splits',
    '正在计算 39 项指标和 7 项元特征':'Computing 39 indicators and 7 meta-features',
    '正在按原 R 流程进行规律候选与验证集选优（最多 6 个候选）':'Generating rule-based candidates and selecting on the validation set (up to 6 candidates)',
    '正在执行原 3par 32 配置调参、双种子筛选与三种子复验':'Running the original 32-configuration tuning protocol with two-seed screening and three-seed confirmation',
    '正在保存本次独立结果文件夹、模型输入输出说明与结构图':'Saving run artifacts, model dimensions, and the architecture diagram',
    '本次计算已完成':'Run completed', '原项目计算未完成':'Computation did not complete',
    '本次任务已取消，已生成文件保留在本次运行目录。':'The run was cancelled. Existing artifacts remain in its output folder.',
    '项目计算进程退出，请展开日志检查运行环境。':'The computation process exited. Expand the logs to check the runtime.',
    '计算未返回完整结果。':'Computation did not return complete results.',
    '无效的运行编号。':'Invalid run ID.', '无效的来源任务目录。':'Invalid source run directory.',
    '找不到本次运行。服务重启后请在 runs 目录查看已保存结果。':'Run not found. If the service was restarted, check the runs folder for saved results.',
    '请使用本机启动地址访问工作台。':'Access the workspace using the local startup URL.',
    '本次模型尚无完整的测试评估结果。':'Complete test results are not available for this model.',
    '接口不存在。':'API endpoint not found.', '请求长度无效。':'Invalid request length.',
    '请求缺少长度或超过 50 MB 限制。':'The request has no valid length or exceeds 50 MB.',
    '文件未完整上传。':'The upload is incomplete.', '上传必须使用 multipart/form-data。':'Uploads must use multipart/form-data.',
    '同一次请求只能包含一个文件与一个预测长度。':'Provide exactly one file and one forecast horizon per request.',
    '请提供一个时间序列文件。':'Provide a time series file.',
    '请上传 50 MB 以内、非空的 NPY 或 CSV 文件。':'Upload a nonempty NPY or CSV file of at most 50 MB.',
    '预测长度只能为 3 或 12。':'The forecast horizon must be 3 or 12.', '建模设置无效。':'Invalid modeling settings.',
    '请选择是否调参。':'Select a tuning mode.', '未知研究示例。':'Unknown example dataset.',
    '请先完成对应预测长度的数据计算。':'Complete data processing for this forecast horizon first.',
    '调参模型请查看已有的原流程测试结果。':'Use the existing test results for the tuned model.',
    '未找到本次任务。':'Run not found.', '取消请求已处理。':'Cancellation request processed.',
    'NPY 必须为数值数组，不支持 object、复数、布尔或日期数组。':'NPY must contain numeric values. Object, complex, Boolean, and date arrays are not supported.',
    'CSV 文件为空。':'The CSV file is empty.', 'CSV 各行列数不一致，或包含空行。':'CSV rows have inconsistent column counts or contain blank rows.',
    '首行存在混合数值与无效字段，无法可靠识别表头。':'The first row mixes numeric and invalid fields; the header cannot be identified reliably.',
    'CSV 只有表头，没有数据。':'The CSV contains a header but no data.',
    'CSV 首列必须为数值变量或可识别的时间戳。':'The first CSV column must contain numeric values or recognizable timestamps.',
    '时间戳存在缺失、重复或倒序，请整理后重新导入；网页不会重排数据。':'Timestamps are missing, duplicated, or out of order. Correct them before importing; the app does not reorder data.',
    'CSV 存在空值，请先完成研究所需的数据清理。':'The CSV contains missing values. Clean the data before importing.',
    'CSV 变量列包含非数值内容。':'CSV variable columns contain nonnumeric values.',
    '数据形状必须为 [时间点] 或 [时间点, 变量]。':'Data must have shape [time] or [time, variables].',
    '数据存在 NaN 或无穷值，请清理后重新导入。':'The data contains NaN or infinite values. Clean it before importing.',
    '60/20/20 划分后数据不足以构造 12 点输入与指定预测长度，请增加时间点。':'After the 60/20/20 split, there are too few observations for a 12-step input and the selected horizon. Provide more time points.',
    '原调参流程未产出唯一推荐模型，请检查本次运行日志。':'Tuning did not produce a unique recommended model. Check the run logs.',
    '调参模型应展示原调参流程的评估结果。':'Tuned models use evaluation results from the original tuning workflow.',
    '本次训练或测试失败，请查看运行日志。':'Training or testing failed. Check the run logs.',
    '调参结果缺少原始推荐参数，无法确认模型输入窗口。':'Recommended parameters are missing; the model input window cannot be confirmed.',
    '评估文件必须位于本次运行目录。':'Evaluation files must remain within this run directory.',
    '评估种子不完整，无法展示本次评估结果。':'Evaluation seeds are incomplete; results cannot be displayed.',
    '评估结果不是原始数据尺度。':'Evaluation results are not in the original data scale.',
    '评估指标与随机种子不匹配。':'The metrics do not match the random seed.',
    '评估窗口与最终推荐模型不匹配。':'The evaluation window does not match the recommended model.',
    '原评估指标缺失或包含非有限数值。':'Evaluation metrics are missing or contain nonfinite values.',
    '预测值与真实值的形状或数值无效。':'Observed and predicted arrays have invalid shapes or values.',
    '没有完整的测试集评估结果。':'Complete test-set evaluation results are not available.',
    '无法识别最终推荐来源，未展示其他配置的评估结果。':'The recommendation source cannot be identified. Results from other configurations are not shown.',
    '变量或曲线范围超出评估数据。':'The selected variable or plot range is outside the evaluation data.',
    '本次评估没有该随机种子的结果。':'This evaluation has no results for the selected seed.',
    '模型源码已生成，但评估结果无法读取：':'Model code is ready, but evaluation results could not be loaded: ',
    '任务状态暂时无法读取：':'Unable to retrieve run status: ', '运行编号':'Run ID',
    '请检查连接后重试，避免重复创建训练任务。':'Check the connection and retry; avoid starting a duplicate training run.'
  });
  Object.assign(messages, {
  "默认输入 12 步，按 60% / 20% / 20% 划分数据。默认参数使用三个随机种子评估，每次最多 300 轮、早停 15。": "The default input is 12 steps. Data are split 60% / 20% / 20%. Default parameters are evaluated with three seeds, up to 300 epochs and an early-stopping patience of 15 per run.",
  "调参采用 3par 的双种子筛选。前四名和默认配置用第三个种子复验，按验证集 RMSE 选优。输入长度可为 12 / 32 / 64 步，选择自动调参后，在弹窗中填写自己的 DeepSeek API key。": "3par screens configurations with two seeds. The top four and the default are checked with a third seed and selected by validation RMSE. Input lengths are 12, 32 or 64 steps. Select Automatic Tuning and enter your DeepSeek API key in the dialog.",
  "项目示例使用已选定的 L 架构和历史测试记录。新数据通过 L 生成候选并按验证集 RMSE 选出架构。": "Project examples use the selected L architectures and archived test results. For new data, L generates candidates and selects an architecture by validation RMSE.",
  "来源：2search/connect/document_summary.csv 与合并发现统计表。": "Source: 2search/connect/document_summary.csv and pooled discovery statistics.",
  "展示当前 L 流程使用的规律、完整条件和合并发现统计。": "Browse the rules, exact conditions and pooled discovery statistics used by the current L workflow.",
  "按建模职责浏览原项目的模块与组合范式。": "Browse the project’s modules and composition patterns by modeling role.",
  "已选定的历史 L 架构": "Archived L architecture",
  "L 架构验证选优完成": "L architecture selected on validation data",
  "正在按 L 流程进行规律候选与验证集选优（最多 6 个候选）": "Selecting among up to six L architecture candidates using validation RMSE",
  "L 默认参数 · 三种子评估": "L default parameters · Three-seed evaluation",
  "最终推荐：历史 L 默认参数 · 三种子评估": "Recommended model: archived L default · Three-seed evaluation",
  "最终推荐：调参配置 · 三种子评估": "Recommended model: tuned parameters · Three-seed evaluation",
  "最终推荐按 3par 协议与历史 L 的测试结果比较。": "The final recommendation follows the 3par comparison against archived L test results.",
  "本次配置按验证集 RMSE 选出。": "This configuration was selected by validation RMSE.",
  "当前模型与历史 L 评估记录不匹配，请重新从项目示例开始。": "This model does not match the archived L evaluation. Start again from a project example.",
  " · 默认参数评估完成": " · Default evaluation complete",
  " · 固定架构调参完成": " · Fixed-architecture tuning complete"
});
  Object.assign(messages, {
    '历史 L 默认配置 RMSE':'Archived L default RMSE',
    '验证选中配置 RMSE':'Validation-selected configuration RMSE',
    'RMSE 相对降幅':'Relative RMSE reduction',
  });
  Object.assign(messages, window.OUTPUT_ENGLISH, {
    '界面语言':'Language', '历史分析':'Analysis History', '查看历史分析':'View History',
    '刷新记录':'Refresh', '按数据集查看本机保存的分析、模型与评估记录。':'Browse locally saved analyses, models and evaluations by dataset.',
    '搜索数据集或运行编号…':'Search datasets or run IDs…', '搜索历史分析':'Search analysis history',
    '筛选运行状态':'Filter run status', '全部状态':'All statuses', '已完成':'Completed', '运行中':'Running',
    '失败':'Failed', '已中断':'Interrupted', '已取消':'Cancelled', '排队中':'Queued',
    '数据集':'Datasets', '运行记录':'Saved runs', '已评估':'Evaluated',
    '正在读取本地记录…':'Loading local history…', '暂无本地分析记录':'No local analyses yet',
    '完成一次建模后，结果会自动出现在这里。':'Completed modeling runs will appear here automatically.',
    '未找到匹配记录':'No matching runs', '试试其他数据集名称或状态。':'Try another dataset name or status.',
    '无法读取本地历史记录':'Local history is unavailable',
    '请运行 start.py 后打开本机页面，再刷新记录。':'Launch start.py, open the local workspace, then refresh.',
    '更新时间':'Updated', '运行编号':'Run ID', '预测步数':'Horizon', '参数设置':'Parameters',
    '状态':'Status', '操作':'Action', '打开结果':'Open Results', '查看进度':'View Progress',
    '仅架构':'Architecture only', '个记录':'runs', '所有文件保留在本机。':'All files stay on this computer.',
    '记录来自本机 runs 目录，按最近更新时间排序。相同数据集的不同预测步数和调参记录归为一组。':'Loaded from the local runs folder, newest first. Forecast horizons and tuning runs for the same dataset are grouped together.',
    '请先等待或取消当前任务，再打开历史结果。':'Wait for or cancel the current run before opening another result.',
    '部分记录无法读取，其他记录已正常显示。':'Some records could not be read. Available records are shown.',
    '此记录未完成，已生成的文件仍保存在本机。':'This run did not finish. Existing files are preserved locally.',
    '历史记录':'History', '输入张量':'Input tensor',
    '基于数据特征与研究规律的模块化时间序列预测工作台。导入数据、查看推荐依据、固定架构调参、导出 PyTorch 模型。':'A modular time series workspace for data analysis, rule-guided modeling, tuning and PyTorch exports.'
  });
  const escapeRE = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const phraseRE = new RegExp(Object.keys(messages).filter(key=>key!=='无').sort((a,b)=>b.length-a.length).map(escapeRE).join('|'), 'g');
  const patterns = [
    [/(\d+) 个记录/g, (_,n)=>`${n} ${n==='1'?'run':'runs'}`],
    [/依据(.+?)特征匹配推荐规律。/g, (_,names)=>`Matches a rule based on ${english(names.replace(/等$/,'')).toLowerCase().replace(', ',' and ')}${names.endsWith('等')?', among other features':''}. `],
    [/指标为 (\d+) 个随机种子的均值/g, (_,n)=>`Metrics are averaged across ${n} seeds`],
    [/，使用完整测试集的 ([\d,]+) 个时间点、(\d+) 个变量。/g, (_,n,c)=>`, using all ${n} test points and ${c} variable${c==='1'?'':'s'}. `],
    [/正在训练与评估一次：种子 (\d+)，最多 (\d+) 轮，早停 (\d+)/g, (_,s,e,p)=>`Single training/evaluation run: seed ${s}, up to ${e} epochs, early-stopping patience ${p}`],
    [/正在调用原训练与测试流程：种子 (\d+)（(\d+)\/(\d+)），最多 (\d+) 轮，早停 (\d+)/g, (_,s,i,n,e,p)=>`Training and evaluation: seed ${s} (${i}/${n}), up to ${e} epochs, patience ${p}`],
    [/原序列第 (.+?) 行 · 真实值 (.+?) · 预测值 (.+?) · 误差 (.+)$/g, (_,r,a,p,e)=>`Row ${r} · Observed ${a} · Predicted ${p} · Error ${e}`],
    [/测试点 (.+?) · 原序列行 (.+)/g, (_,n,r)=>`Test points ${n} · Original rows ${r}`],
    [/测试集真实值与预测值对比，变量 (\d+)，种子 (\d+)/g, (_,c,s)=>`Observed and predicted test values, variable ${c}, seed ${s}`],
    [/最终模型结构：输入 (\d+) 步，输出 (\d+) 步，(\d+) 个变量/g, (_,i,o,c)=>`Final architecture: ${i} input steps, ${o} forecast steps, ${c} variable${c==='1'?'':'s'}`],
    [/([\d,]+) 个时间点/g, (_,n)=>`${n} time points`], [/(\d+) 个变量/g, (_,n)=>`${n} variable${n==='1'?'':'s'}`],
    [/(\d+) 条规律/g, (_,n)=>`${n} rules`], [/([SA]) 级/g, (_,g)=>`Grade ${g}`],
    [/预测长度 = (\d+) 步/g, (_,n)=>`Forecast horizon = ${n} steps`],
    [/服务请求失败（(\d+)）/g, (_,n)=>`Service request failed (${n})`],
    [/当前 Python 环境缺少：(.+?)。请使用已安装根目录 requirements.txt 的项目环境启动。/g, (_,names)=>`Missing Python dependencies: ${names}. Start the app in the project environment with requirements.txt installed.`],
    [/缺少 (.+?)；/g, (_,names)=>`Missing dependencies: ${names}; `]
  ];
  function english(value) {
    let text=String(value??'');
    if(messages[text.trim()])return text.replace(text.trim(),messages[text.trim()]);
    for(const [pattern,replacement] of patterns)text=text.replace(pattern,replacement);
    return text.replace(phraseRE,match=>messages[match]).replace(/、/g,', ').replace(/。/g,'. ').replace(/；/g,'; ').replace(/：/g,': ');
  }
  // API labels are stored in English; use the Chinese source for switchable UI text.
  const sources=new Map(Object.entries(messages).map(([zh,en])=>[en,zh]));
  const source=value=>sources.get(value)||value;
  let language='en';
  try { if(localStorage.getItem('affinity-law.language')==='zh')language='zh'; } catch {}
  const originals=new WeakMap(), attributes=new WeakMap();
  const ignored='script,style,pre,code,[translate="no"]';
  function textNode(node) {
    if(node.parentElement?.closest(ignored))return;
    const current=node.nodeValue, saved=originals.get(node);
    const source=saved&&current===saved.output?saved.source:current;
    const output=language==='en'?english(source):source;
    originals.set(node,{source,output});
    if(current!==output)node.nodeValue=output;
  }
  function element(node) {
    if(node.closest(ignored))return;
    let saved=attributes.get(node); if(!saved){saved=new Map();attributes.set(node,saved);}
    for(const key of ['aria-label','placeholder','title','alt',...(node.matches('meta[name="description"]')?['content']:[])]) {
      if(!node.hasAttribute(key))continue;
      const current=node.getAttribute(key), previous=saved.get(key);
      const source=previous&&current===previous.output?previous.source:current;
      const output=language==='en'?english(source):source;
      saved.set(key,{source,output}); if(current!==output)node.setAttribute(key,output);
    }
  }
  function walk(root) {
    if(root.nodeType===Node.TEXT_NODE){textNode(root);return;}
    if(root.nodeType===Node.ELEMENT_NODE){if(root.closest(ignored))return;element(root);}
    for(const child of root.childNodes)walk(child);
  }
  const observer=new MutationObserver(records=>{
    observer.disconnect();
    for(const record of records) {
      if(record.type==='childList')for(const node of record.addedNodes)walk(node);
      else if(record.type==='characterData')textNode(record.target);
      else element(record.target);
    }
    observe();
  });
  function observe(){observer.observe(document.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['aria-label','placeholder','title','alt','content']});}
  function setLanguage(value) {
    language=value==='en'?'en':'zh';
    try {localStorage.setItem('affinity-law.language',language);}catch{}
    observer.disconnect(); document.documentElement.lang=language==='en'?'en':'zh-CN';
    document.querySelector('#language-select').value=language;
    walk(document.documentElement); observe();
  }
  function svg(source) {
    const doc=new DOMParser().parseFromString(source,'image/svg+xml');
    for(const node of doc.querySelectorAll('text,title,desc'))node.textContent=english(node.textContent);
    for(const node of doc.querySelectorAll('[aria-label]'))node.setAttribute('aria-label',english(node.getAttribute('aria-label')));
    return new XMLSerializer().serializeToString(doc.documentElement);
  }
  function output(value) {
    if(Array.isArray(value))return value.map(output);
    if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,output(item)]));
    return typeof value==='string'?english(value):value;
  }
  document.querySelector('#language-select').addEventListener('change',event=>setLanguage(event.target.value));
  setLanguage(language);
  return {english,svg,output,source,get language(){return language;}};
})();
