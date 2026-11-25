'use strict';

(() => {
  const data = window.RESEARCH_DATA;
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icons = {
    grid:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    history:'<path d="M3 11a9 9 0 1 1 2.6 7M3 4v7h7m2-5v6l4 2"/>',
    refresh:'<path d="M20 7a9 9 0 0 0-15-1L2 9m0-6v6h6m-4 8a9 9 0 0 0 15 1l3-3m0 6v-6h-6"/>',
    layers:'<path d="m12 3 10 6-10 6L2 9l10-6Zm-10 11 10 6 10-6M2 18l10 6 10-6"/>',
    book:'<path d="M12 5v15M3 3c4-1 6 0 9 2 3-2 5-3 9-2v16c-4-1-6 0-9 2-3-2-5-3-9-2V3Z"/>',
    compass:'<circle cx="12" cy="12" r="9"/><path d="m16 8-3 5-5 3 3-5 5-3Z"/>',
    sparkles:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3ZM20 2v4m-2-2h4"/>',
    settings:'<path d="m9 3-1 3-3 1-1 4 2 2v3l3 3 3-1 3 1 3-3v-3l2-2-1-4-3-1-1-3H9Z"/><circle cx="12" cy="11" r="3"/>',
    help:'<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 0 1 6 0c0 2-3 2-3 4m0 3h.01"/>',
    upload:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
    'file-chart':'<path d="M14 2H5v20h14V7l-5-5Zm0 0v6h5M8 17v-3m4 3v-6m4 6v-2"/>',
    plus:'<path d="M12 5v14M5 12h14"/>',
    info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
    sliders:'<path d="M6 3v7m0 4v7M12 3v2m0 4v12M18 3v11m0 4v3M3 10h6M9 5h6m0 9h6"/>',
    lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
    shield:'<path d="m12 2 8 3v6c0 5-5 9-8 11-3-2-8-6-8-11V5l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    'arrow-right':'<path d="M4 12h16m-6-6 6 6-6 6"/>',
    activity:'<path d="M2 12h4l4-9 4 18 4-9h4"/>',
    workflow:'<rect x="2" y="8" width="5" height="8" rx="1"/><rect x="17" y="2" width="5" height="7" rx="1"/><rect x="17" y="15" width="5" height="7" rx="1"/><path d="M7 12h5m0 0V5h5m-5 7v7h5"/>',
    zap:'<path d="m13 2-9 12h7l-1 8 10-13h-7l1-7Z"/>',
    check:'<path d="m5 12 4 4L19 6"/>',
    code:'<path d="m7 6-6 6 6 6m10-12 6 6-6 6M14 3l-4 18"/>',
    copy:'<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
    download:'<path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4"/>',
    terminal:'<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m6 9 3 3-3 3m7 0h5"/>',
    search:'<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
    leaf:'<path d="M20 3C10 1 2 9 6 16S22 20 20 3ZM4 21 16 8"/>',
    bike:'<circle cx="5" cy="16" r="4"/><circle cx="19" cy="16" r="4"/><path d="m5 16 5-9 5 9H5m10 0 3-12h-4M8 5h4"/>',
  };
  const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.grid}</svg>`;
  const hydrateIcons = () => $$('[data-icon]').forEach(el => el.outerHTML = icon(el.dataset.icon));
  const features = [
    ['distribution_shape_score','分布形态','偏态、尾部与分布形态的综合表征。'],
    ['trend_strength_score','趋势强度','长期变化与趋势成分的综合表征。'],
    ['stationarity_score','平稳性','序列统计性质稳定程度的综合表征。'],
    ['periodicity_score','周期性','重复模式与周期成分的综合表征。'],
    ['temporal_dependence_score','时间依赖','历史观测间时序关联的综合表征。'],
    ['noise_volatility_score','噪声波动','不规则变化与波动性的综合表征。'],
    ['multivariate_structure_score','多变量结构','变量之间关联与结构的综合表征。'],
  ];
  const featureNames = Object.fromEntries(features.map(f => [f[0], f[1]]));
  const stages = {A:'输入处理',B:'结构建模',C:'预测生成',D:'输出还原',E:'组合范式'};
  const descriptions = {
    'Identity':'保持输入不变，不额外引入预处理。',
    'Z-score':'按窗口逐通道标准化，预测后逆变换回原始尺度。',
    'RevIN':'逐样本、逐变量可逆归一化，并使用可学习仿射变换。',
    'Differencing':'使用一阶差分表达局部变化，输出端通过累加还原。',
    'Series Decomposition':'分离趋势与残差，骨干处理残差，输出时重组趋势。',
    'Patch':'对重叠时间片段进行局部线性变换，再恢复原序列长度。',
    'Mask':'通过项目定义的掩码处理输入位置与变量。',
    'Linear Block':'使用线性变换建模时间维度映射。',
    'MLP Block':'利用多层感知机表示非线性特征映射。',
    'CNN Block':'使用一维卷积提取局部时间模式。',
    'TCN Block':'通过膨胀因果卷积建模跨时间步的依赖。',
    'GRU Block':'使用门控循环单元更新时序隐状态。',
    'LSTM Block':'使用带记忆单元的门控递归更新时序状态。',
    'State Space Model Block':'通过状态空间递推表示序列动态变化。',
    'Transformer Encoder Block':'用自注意力与前馈层编码时间序列表征。',
    'Basis Block':'通过基函数组合表示时间序列变化。',
    'Linear Head':'将隐藏表征线性投影为指定预测长度的输出。',
    'MLP Head':'通过多层感知机将隐藏表征映射为预测序列。',
    'Basis Projection':'将系数投影到基函数上生成预测。',
    'Inverse Z-score':'恢复 Z-score 标准化之前的数值尺度。',
    'Inverse RevIN':'恢复 RevIN 归一化之前的数值尺度。',
    'Inverse Differencing':'从差分预测逐步累加，重建原始数值。',
    'Decomposition Recomposition':'将预测分量与趋势重新组合为最终输出。',
    'Single Block':'只使用一个骨干模块进行建模。',
    'Sequential Composition':'将两个不同骨干模块依次连接。',
    'Parallel Additive Branches':'两个骨干并行处理，再将分支输出相加。',
    'Residual Modeling':'主分支与校正分支分别处理输入，再将输出相加。',
    'Gated Fusion':'使用可学习门控融合两个骨干的输出。',
  };
  const state = {step:0, horizon:3, file:null, sample:null, result:null, code:null, tuning:false, busy:false, job:null, online:false, view:'workbench'};
  let deepseekKey = '';
  let keyPrompt = null;

  function setTuning(enabled) {
    state.tuning = enabled;
    state.code = null;
    if (!enabled) deepseekKey = '';
    $$('[name=tuning]').forEach(input => input.checked = (input.value === 'on') === enabled);
    $$('.tuning-choice').forEach(choice => choice.classList.toggle('selected', $('input', choice).checked));
    renderSettings();
  }

  function askDeepSeekKey() {
    if (keyPrompt) return keyPrompt;
    const dialog = $('#api-key-dialog');
    const input = $('#deepseek-api-key');
    input.value = '';
    input.setCustomValidity('');
    dialog.returnValue = '';
    keyPrompt = new Promise(resolve => {
      dialog.addEventListener('close', () => {
        const confirmed = dialog.returnValue === 'confirm';
        input.value = '';
        if (!confirmed) setTuning(false);
        keyPrompt = null;
        resolve(confirmed);
      }, {once: true});
    });
    dialog.showModal();
    return keyPrompt;
  }

  let toastTimer;
  function toast(message){$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,4500);}
  function setView(view){if(!['workbench','history','library','evidence','guide'].includes(view))return;state.view=view;$$('.view').forEach(el=>el.hidden=el.id!==`view-${view}`);$$('.main-nav button').forEach(el=>el.classList.toggle('active',el.dataset.view===view));$('#page-name').textContent={workbench:'模型工作台',history:'历史分析',library:'模块库',evidence:'规律与证据',guide:'使用指南'}[view];if(location.protocol!=='file:')history.replaceState(null,'',location.pathname+location.search+(view==='workbench'?'':'#'+view));if(view==='history')loadHistory();window.scrollTo({top:0,behavior:'instant'});}
  let archive=null, historyLoading=false;
  async function loadHistory(){
    if(historyLoading)return;
    historyLoading=true;$('#refresh-history').disabled=true;
    if(!archive)$('#history-list').innerHTML='<div class="empty-state">正在读取本地记录…</div>';
    try{
      archive=await api('/api/history');
      $('#history-count').textContent=archive.total_runs;
      const evaluated=archive.datasets.reduce((sum,group)=>sum+group.runs.filter(run=>run.has_evaluation&&run.status==='completed').length,0);
      $('#history-summary').innerHTML=[['数据集',archive.datasets.length],['运行记录',archive.total_runs],['已评估',evaluated]].map(([label,count])=>`<div><span>${label}</span><strong>${count}</strong></div>`).join('');
      renderHistory();
    }catch(error){
      $('#history-list').innerHTML=`<div class="empty-state"><h3>无法读取本地历史记录</h3><p>请运行 start.py 后打开本机页面，再刷新记录。</p><small>${esc(error.message)}</small></div>`;
    }finally{historyLoading=false;$('#refresh-history').disabled=false;}
  }
  function renderHistory(){
    if(!archive)return;
    const query=$('#history-search').value.trim().toLowerCase(), status=$('#history-status').value;
    const statuses={completed:'已完成',running:'运行中',queued:'排队中',failed:'失败',interrupted:'已中断',cancelled:'已取消'};
    const metric=(value,percent=false)=>Number.isFinite(value)?(percent?`${(value*100).toFixed(2)}%`:Number(value.toPrecision(5)).toLocaleString('en-US',{maximumSignificantDigits:5})):'—';
    const groups=archive.datasets.map(group=>({...group,runs:group.runs.filter(run=>(!status||(status==='running'?['running','queued'].includes(run.status):run.status===status))&&(!query||[group.name,run.name,run.id].some(text=>text.toLowerCase().includes(query))))})).filter(group=>group.runs.length);
    if(!groups.length){$('#history-list').innerHTML=`<div class="empty-state"><h3>${archive.total_runs?'未找到匹配记录':'暂无本地分析记录'}</h3><p>${archive.total_runs?'试试其他数据集名称或状态。':'完成一次建模后，结果会自动出现在这里。'}</p></div>`;return;}
    $('#history-list').innerHTML=(archive.skipped?'<p class="history-warning">部分记录无法读取，其他记录已正常显示。</p>':'')+groups.map(group=>`<details class="card history-group" open><summary><span class="dataset-icon">${icon('file-chart')}</span><span class="dataset-name" translate="no">${esc(group.name)}</span><span class="chip neutral">${group.runs.length} 个记录</span></summary><div class="history-table-wrap"><table class="history-table"><thead><tr><th>更新时间</th><th>预测步数</th><th>参数设置</th><th>状态</th><th>RMSE</th><th>MAE</th><th>WMAPE</th><th>操作</th></tr></thead><tbody>${group.runs.map(run=>{
      const date=new Date(run.updated_at), timestamp=date.toLocaleString('sv-SE',{hour12:false});
      const available=run.has_result||['running','queued'].includes(run.status);
      return `<tr><td><time datetime="${esc(run.updated_at)}">${esc(timestamp)}</time><small class="run-id" title="${esc(run.id)}">${run.id.slice(0,8)}</small></td><td>${esc(run.horizon??'—')}</td><td>${run.tuned?'自动调参':'默认参数'}</td><td><span class="run-status ${esc(run.status)}">${statuses[run.status]||esc(run.status)}</span>${run.has_result&&!run.has_evaluation?'<small class="run-note">仅架构</small>':''}</td><td class="numeric">${metric(run.metrics.rmse)}</td><td class="numeric">${metric(run.metrics.mae)}</td><td class="numeric">${metric(run.metrics.wmape,true)}</td><td>${available?`<a class="history-open" data-history-job="${run.id}" href="?job=${run.id}">${run.has_result?'打开结果':'查看进度'} ${icon('arrow-right')}</a>`:'<span class="muted" title="此记录未完成，已生成的文件仍保存在本机。">—</span>'}</td></tr>`;
    }).join('')}</tbody></table></div></details>`).join('');
  }
  function renderWorkflow(){
    const active=state.step===4?2:state.busy?1:0;
    $('#workflow').innerHTML=['数据与设置','自动建模','模型结果'].map((title,i)=>`${i?'<span class="workflow-line"></span>':''}<div class="workflow-step ${i===active?'active':i<active?'done':''}" ${i===active?'aria-current="step"':''}><span class="step-number">${i<active?'✓':i+1}</span><strong>${title}</strong></div>`).join('');
  }
  function setStep(step){
    if(![0,4].includes(step))return;
    if(state.busy&&step!==state.step)return;
    if(step===4&&(!state.result||!state.code))return;
    state.step=step;setView('workbench');
    $$('.step-panel').forEach(panel=>panel.hidden=panel.id!==`step-${step}`);
    document.body.classList.toggle('has-results',step===4);
    renderWorkflow();if(step===4)renderExport();
  }
  function renderBanner(){
    const r=state.result, banner=$('#session-banner');
    banner.hidden=!r;
    if(!r)return;
    const example=r.source==='recorded';
    const name=example?window.ModelI18n.source(r.name):r.name;
    banner.innerHTML=`<span${example?'':' translate="no"'}>${esc(name)}</span> · ${r.rows.toLocaleString()} 个时间点 · ${r.channels} 个变量`;
  }
  function renderExamples(){$('#example-grid').innerHTML=data.examples.map((r,i)=>`<button class="example-card" data-example="${i}"><span class="example-icon">${icon(['zap','bike','leaf'][i])}</span><div><h4>${esc(window.ModelI18n.source(r.name))}</h4><p>${esc(window.ModelI18n.source(r.category))} · ${r.rows.toLocaleString()} × ${r.channels}</p></div>${icon('arrow-right')}</button>`).join('');}
  function clearDownstream(){window.ModelPerformance.reset();state.result=null;state.baseResult=null;state.code=null;state.job=null;$('#session-banner').hidden=true;$('#analysis-job').hidden=true;if(location.protocol!=='file:')history.replaceState(null,'',location.pathname);}
  function chooseExample(index){if(state.busy)return;clearDownstream();state.sample=data.examples[index];state.file=null;$('#file-input').value='';$('#upload-error').hidden=true;showFile(window.ModelI18n.source(state.sample.name),`${state.sample.rows.toLocaleString()} 个时间点 · ${state.sample.channels} 个变量`);$('#start-analysis').disabled=false;$('#start-analysis').innerHTML='开始建模 '+icon('arrow-right');}
  function showFile(name,subtitle){$('#file-summary').hidden=false;$('#file-summary').innerHTML=icon('file-chart')+`<div class="file-details"><b ${state.file?'translate="no"':''}>${esc(name)}</b><p>${esc(subtitle)}</p></div><button class="icon-button" id="remove-file" aria-label="移除文件">×</button>`;}
  function uploadError(message){$('#upload-error').textContent=message;$('#upload-error').hidden=false;}
  function acceptFile(file){if(state.busy)return toast('当前计算未结束，暂时不能替换文件。');clearDownstream();state.sample=null;state.file=null;$('#start-analysis').disabled=true;$('#file-summary').hidden=true;$('#upload-error').hidden=true;if(!file)return;if(!/\.(npy|csv)$/i.test(file.name))return uploadError('请选择 .npy 或 .csv 格式的时间序列文件。');if(!file.size)return uploadError('文件为空，请检查后重新选择。');if(file.size>50*1024*1024)return uploadError('文件超过 50 MB，请选择符合大小限制的数据。');state.file=file;showFile(file.name,`${(file.size/1024).toFixed(1)} KB · 等待发送至计算服务校验`);$('#start-analysis').innerHTML='开始建模 '+icon('arrow-right');$('#start-analysis').disabled=false;}
  function sampleResult(sample){const v=sample.variants[String(state.horizon)];return {...sample,...v,selection_status:'Archived L architecture',law_table_sha256:data.law_sha256,tuned:false};}
  function setBusy(busy){
    state.busy=busy;$('#setup-fields').hidden=busy;
    $('#start-analysis').disabled=busy||(!state.file&&!state.sample);
    document.body.classList.toggle('is-modeling',busy);renderWorkflow();
  }
  async function analyze(){
    if(state.busy||(!state.file&&!state.sample))return;
    if (state.tuning && !deepseekKey && !await askDeepSeekKey()) return;
    clearDownstream();
    if(location.protocol==='file:'&&state.sample&&!state.tuning){state.result=sampleResult(state.sample);state.code=state.result.model_code;renderBanner();setStep(4);return;}
    setBusy(true);$('#upload-error').hidden=true;
    showJob('analysis-job',{status:'queued',message:'正在准备本次建模'});
    try{
      const health=await checkConnection();
      if(!health.compute_ready)throw Error(health.message);
      if(state.tuning&&!health.tuning_ready)throw Error(health.tuning_message);
      const headers = state.tuning ? {'X-DeepSeek-API-Key': deepseekKey} : {};
      let job;
      if(state.sample){
        job=await api('/api/modeling',{method:'POST',headers:{...headers,'Content-Type':'application/json'},body:JSON.stringify({sample_id:state.sample.id,horizon:state.horizon,tuning:state.tuning})});
      }else{
        const body=new FormData();body.append('file',state.file);body.append('horizon',state.horizon);body.append('run_model','true');body.append('tuning',String(state.tuning));
        job=await api('/api/jobs',{method:'POST',headers,body});
      }
      deepseekKey = '';
      delete headers['X-DeepSeek-API-Key'];
      state.job=job.id;history.replaceState(null,'',`?job=${encodeURIComponent(job.id)}`);
      await pollJob(job.id,'analysis-job',4);
    }catch(error){uploadError(error.message);if(!state.job)$('#analysis-job').hidden=true;}
    finally{setBusy(false);}
  }
  function metaValue(key){const value=state.result?.meta?.[key];return Number.isFinite(Number(value))?Number(value):null;}
  function renderFeatures(){
    const spec=state.result.spec;
    const selectedRules=['A','B','C','E'].flatMap(stage=>(Array.isArray(spec[stage])?spec[stage]:[spec[stage]]).map(module=>bestRule(stage,module))).filter(Boolean);
    const keys=[...new Set(selectedRules.flatMap(rule=>features.filter(([key])=>rule.condition.includes(key)).map(([key])=>key)))];
    const selected=[...new Set([...keys,'trend_strength_score','periodicity_score','stationarity_score'])].slice(0,3);
    $('#feature-grid').innerHTML=selected.map(key=>{const value=metaValue(key);return `<div><span>${featureNames[key]}</span><b>${value===null?'—':value.toFixed(3)}</b></div>`;}).join('');
  }
  function hitRules(){
  const digest = state.result?.law_table_sha256;
  if(digest!==data.law_sha256 && !data.equivalent_law_sha256?.includes(digest))return [];
  return (state.result.hits||[]).map(i=>data.rules[i]).filter(Boolean);
}
  function bestRule(stage,module){return hitRules().filter(r=>r.stage===stage&&r.module===module).sort((a,b)=>({'S':0,'A':1}[a.grade]-{'S':0,'A':1}[b.grade])||Number(b.lift_ci_lo||b.lift)-Number(a.lift_ci_lo||a.lift))[0];}
  function prettyCondition(condition){let text=String(condition);for(const[key,value]of Object.entries(featureNames))text=text.replaceAll(key,value);return text.replaceAll('config=12_3','预测长度 = 3 步').replaceAll('config=12_12','预测长度 = 12 步');}
  function renderRecommendations(){
    const spec=state.result.spec;
    $('#recommendation-grid').innerHTML=['A','B','C','E'].flatMap(stage=>{
      const modules=Array.isArray(spec[stage])?spec[stage]:[spec[stage]];
      return modules.map((module,index)=>recommendCard(stage,module,index));
    }).join('');
  }
  function recommendCard(stage,module,index){
    const rule=bestRule(stage,module);
    const names=rule?features.filter(([key])=>rule.condition.includes(key)).map(([,name])=>name):[];
    const reason=rule?(names.length?`依据${names.slice(0,2).join('、')}${names.length>2?'等':''}特征匹配推荐规律。`:'匹配当前预测长度的推荐规律。'):
      stage==='B'&&index>0?'根据组合需要补充该模块。':stage==='E'?'沿用项目默认组合。':'由项目默认配置或候选选优确定。';
    return `<article><h4><span>${stages[stage]}</span>${esc(module)}</h4><p>${esc(reason)}${esc(descriptions[module]||'')}</p></article>`;
  }
  function renderSettings(){$('#tuning-help').hidden=!state.tuning;$('#edit-api-key').hidden=!state.tuning;}

  function report(){const r=state.result;return window.ModelI18n.output({application:'affinity law',output_language:'en',created_at:new Date().toISOString(),dataset:r.name,source:r.source,horizon:state.horizon,shape:[r.rows,r.channels],meta_features:r.meta,spec:r.spec,matched_rules:hitRules(),law_table_sha256:r.law_table_sha256||null,selection_status:r.selection_status,model_io:r.model_io||{input_steps:r.tuning_result?.time_lag||12,output_steps:state.horizon,variables:r.channels},output_folder:r.job_id?`runs/${r.job_id}`:null,tuning_completed:!!r.tuned,evaluation:r.evaluation||null,tuning_settings:r.tuning_settings||null,tuning_comparison:r.tuning_comparison||null,recommendation_source:r.recommendation_source||null,model_configuration:r.tuning_result||{time_lag:12,lr:.001,batch_size:16,weight_decay:0},notes:['Original algorithms and research protocols are preserved.','Lift indicates module preference, not a gain in forecast accuracy.','Source code excludes trained weights. Use the original project environment.']});}
  function renderExport(){const r=state.result;state.code=window.ModelI18n.english(state.code);window.ModelPerformance.render(r,{api,evaluate:evaluateModel,download});$('#code-preview').textContent=state.code;$('#code-filename').textContent=r.tuned?'recommended_model.py':'candidate_model.py';$('#export-subtitle').textContent=r.evaluation?'模型与评估结果已保存，可以直接下载使用。':'模型代码已生成，尚未完成训练评估。';$('#research-details').open=false;$('#export-facts').innerHTML=[['输入步数',`${r.tuning_result?.time_lag||12} 步`],['输出步数',`${state.horizon} 步`],['变量数量',r.channels],['参数状态',r.tuned?'完成调参':'默认参数'],['源码行数',state.code.split('\n').length]].map(([k,v])=>`<div class="protocol-row"><span>${k}</span><b>${v}</b></div>`).join('');$('#download-note').textContent=r.tuned?'下载为调参后推荐源码。训练权重与完整记录保存在本机 4html/runs 目录。':(r.evaluation?'已完成默认参数评估。源码不包含权重；权重与预测数据保存在本机 4html/runs。':'默认导出为未训练架构源码，不包含训练权重。');const lag=r.tuning_result?.time_lag||12;$('#usage-example').textContent=`from ${r.tuned?'recommended_model':'candidate_model'} import CandidateModel → CandidateModel(time_lag=${lag}, pre_len=${state.horizon}, station_num=${r.channels})；输入张量 [B, ${lag}, ${r.channels}]。` ;}
  function modelRequest(){const r=state.result;const sample=r.sample_id||(r.source==='recorded'?r.id:null);return sample?{sample_id:sample,horizon:state.horizon}:{job_id:r.job_id,horizon:state.horizon};}

  async function evaluateModel(){
    if(state.busy||!state.result||state.result.evaluation)return;
    state.busy=true;
    $('#evaluate-model').disabled=true;
    try{
      const health=await checkConnection();
      if(!health.compute_ready)throw Error(health.message);
      const job=await api('/api/evaluation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(modelRequest())});
      state.job=job.id;
      await pollJob(job.id,'evaluation-job',4);
    }catch(error){
      $('#evaluation-job').hidden=false;
      $('#evaluation-job').innerHTML=`<div class="job-card"><h3>评估尚未完成</h3><p>${esc(error.message)}</p></div>`;
    }finally{
      state.busy=false;
      if($('#evaluate-model'))$('#evaluate-model').disabled=false;
    }
  }
  function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  function renderLibrary(stage='all'){$('#library-filters').innerHTML=[['all','全部模块'],...Object.entries(stages).map(([a,b])=>[a,`${a} · ${b}`])].map(([a,b])=>`<button class="filter-button ${stage===a?'active':''}" data-library-stage="${a}">${b}</button>`).join('');const modules=Object.entries(data.library).flatMap(([key,ms])=>ms.map(m=>({...m,stage:key[0]}))).filter(m=>stage==='all'||stage===m.stage);$('#module-grid').innerHTML=modules.map(m=>`<article class="card module-card"><span class="chip lilac">${m.stage} · ${stages[m.stage]}</span><h3>${esc(m.module_name)}</h3><p>${esc(descriptions[m.module_name]||m.role)}</p><p class="module-original">${esc(m.role)}</p>${m.inverse_required?'<p class="module-original">↩ 输出端配套执行逆变换</p>':''}</article>`).join('');}
  function qualitativeSearch(condition){
  return String(condition||'').split(', ').map(part=>{
    const [feature,level]=part.split('=');
    return window.ModelI18n.source(feature)+window.ModelI18n.source('='+level);
  }).join(', ');
}
function renderRules(){const q=$('#rule-search').value.toLowerCase().trim(),stage=$('#rule-stage').value,grade=$('#rule-grade').value;const rules=data.rules.filter(r=>(!stage||r.stage===stage)&&(!grade||r.grade===grade)&&(!q||[r.module,r.condition,r.condition_qual,qualitativeSearch(r.condition_qual),prettyCondition(r.condition),window.ModelI18n.english(prettyCondition(r.condition))].join(' ').toLowerCase().includes(q)));$('#rule-count').textContent=`${rules.length} 条规律`;$('#rules-list').innerHTML=rules.length?rules.map(r=>`<details class="rule-row"><summary><span class="stage-badge">${r.stage}</span><span><strong>${esc(r.rule_id ? r.rule_id+' · ' : '')}${esc(r.module)}</strong><small>${esc(prettyCondition(r.condition))}</small></span><span class="chip lilac">${r.grade} 级</span><span class="chip neutral">${Number(r.lift).toFixed(2)}× Lift</span></summary><div class="rule-content"><div class="evidence-metrics"><div><span>覆盖数据集</span><b>${r.n_dataset_confirm}</b></div><div><span>Lift 区间</span><b>${Number(r.lift_ci_lo).toFixed(2)}–${Number(r.lift_ci_hi).toFixed(2)}</b></div><div><span>q 值</span><b>${Number(r.q_rule).toPrecision(3)}</b></div><div><span>覆盖率</span><b>${(Number(r.coverage)*100).toFixed(1)}%</b></div></div><p>${esc(window.ModelI18n.source(r.evidence_tier))} · ${esc(window.ModelI18n.source(r.scope))} · ${esc(window.ModelI18n.source(r.purity))}</p><p>反偏好角：${esc(r.impure_detail||'无')}</p><p>原始条件：${esc(r.conditions_str_exact||r.condition)}</p><p>模块作用：${esc(descriptions[r.module]||'沿用原项目定义')}</p></div></details>`).join(''):'<div class="empty-state">没有匹配的规律，试试其他关键词或筛选条件。</div>';}
  async function api(path,options={}){if(location.protocol==='file:')throw Error('请先在项目环境中运行 python -B 4html/start.py，再使用自动打开的本地网页计算数据。');const response=await fetch(path,{...options,signal:options.signal||AbortSignal.timeout(30000)});const content=await response.text();let result;try{result=JSON.parse(content);}catch{throw Error('本地计算服务未返回有效结果，请确认是从 start.py 打开的网页。');}if(!response.ok)throw Error(result.error||`服务请求失败（${response.status}）`);return result;}
  async function checkConnection(){try{const health=await api('/api/health');if(health.application!=='tslab-project-bridge')throw Error('该地址不是时序工坊计算服务。');state.online=true;$('#connection-dot').classList.add('online');$('#connection-label').textContent=health.compute_ready?'本地计算已就绪':'本地环境 · 依赖待安装';return health;}catch(error){state.online=false;$('#connection-dot').classList.remove('online');$('#connection-label').textContent='仅预览 · 未启动计算';throw error;}}
  function showJob(target,job){
    const el=$('#'+target), expanded=el.querySelector('details')?.open;
    const message=window.ModelI18n.source(job.message)||'正在准备本次建模';
    const title=job.status==='failed'?'本次建模未完成':job.status==='cancelled'?'本次建模已取消':/保存/.test(message)?'正在保存结果':/调参/.test(message)?'正在优化模型':/种子|训练|测试/.test(message)?'正在训练与评估':/特征|候选|规律/.test(message)?'正在分析数据与选择模型':'正在准备建模';
    const running=['running','queued'].includes(job.status);
    el.hidden=false;
    el.innerHTML=`<div class="job-card"><h3>${icon('activity')}${title}</h3>${running?'<div class="progress-track indeterminate" aria-label="正在自动建模"></div><p>完成后自动展示结果。关闭页面后，计算仍会在本机继续。</p>':''}<details ${expanded?'open':''}><summary>查看运行详情</summary><p>${esc(message)}</p>${job.log?`<pre>${esc(job.log)}</pre>`:''}</details>${running&&job.id?`<button class="button secondary compact" data-cancel-job="${esc(job.id)}">取消本次建模</button>`:''}</div>`;
  }
  async function pollJob(id,target,completionStep){let failures=0;while(state.job===id){let job;try{job=await api(`/api/jobs/${encodeURIComponent(id)}`);failures=0;}catch(error){failures++;if(failures>=3)throw Error(`任务状态暂时无法读取：${error.message} 运行编号 ${id}。请检查连接后重试，避免重复创建训练任务。`);await new Promise(resolve=>setTimeout(resolve,2000));continue;}showJob(target,job);if(['failed','cancelled','interrupted'].includes(job.status))throw Error(job.error||job.message||'任务已停止。');if(job.status==='completed'){if(!job.result?.spec||!job.result?.model_code)throw Error('服务返回的模型结果不完整。');state.result=job.result;state.horizon=job.result.horizon;const nextStep=completionStep??4;state.code=nextStep===4?job.result.model_code:null;if(target==='evaluation-job')state.baseResult=job.result;setBusy(false);$('#'+target).hidden=true;renderBanner();setStep(nextStep);return;}await new Promise(resolve=>setTimeout(resolve,1400));}}
  function openConnection(){$('#connection-dialog').showModal();testConnection();}
  async function testConnection(){try{$('#connection-result').textContent='正在检查本机运行环境…';const h=await checkConnection();$('#connection-result').textContent=`${h.message} ${h.tuning_message}`;}catch(e){$('#connection-result').textContent=e.message;}}
  document.addEventListener('click',event=>{const b=event.target.closest('button,a');if(!b)return;if(b.dataset.historyJob&&state.busy){event.preventDefault();return toast('请先等待或取消当前任务，再打开历史结果。');}if(b.dataset.view){event.preventDefault();setView(b.dataset.view);}if(b.dataset.step!==undefined)setStep(Number(b.dataset.step));if(b.dataset.example!==undefined)chooseExample(Number(b.dataset.example));if(b.dataset.horizon){if(state.busy)return toast('计算期间无法更改预测长度。');const horizon=Number(b.dataset.horizon);if(state.horizon===horizon)return;state.horizon=horizon;clearDownstream();$$('[data-horizon]').forEach(el=>{el.classList.toggle('selected',Number(el.dataset.horizon)===horizon);el.setAttribute('aria-pressed',String(Number(el.dataset.horizon)===horizon));});}if(b.dataset.libraryStage)renderLibrary(b.dataset.libraryStage);if(b.id==='remove-file'){if(state.busy)return toast('计算中暂时不能移除文件。');state.file=null;state.sample=null;clearDownstream();$('#file-input').value='';$('#file-summary').hidden=true;$('#start-analysis').disabled=true;$('#upload-error').hidden=true;}if(b.dataset.cancelJob){b.disabled=true;api(`/api/jobs/${encodeURIComponent(b.dataset.cancelJob)}/cancel`,{method:'POST'}).catch(e=>{toast(e.message);b.disabled=false;});}});
  $('#choose-file').addEventListener('click',()=>$('#file-input').click());$('#file-input').addEventListener('change',e=>acceptFile(e.target.files[0]));const drop=$('#drop-zone');['dragenter','dragover'].forEach(e=>drop.addEventListener(e,event=>{event.preventDefault();drop.classList.add('dragover');}));['dragleave','drop'].forEach(e=>drop.addEventListener(e,event=>{event.preventDefault();drop.classList.remove('dragover');}));drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length!==1)return uploadError('每次研究请选择一个时间序列文件。');acceptFile(e.dataTransfer.files[0]);});window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
  $('#start-analysis').addEventListener('click', analyze);
  $$('[name=tuning]').forEach(input => input.addEventListener('change', async () => {
    if (state.busy) {
      input.checked = state.tuning === (input.value === 'on');
      return toast('调参运行中无法切换模式。');
    }
    setTuning(input.value === 'on');
    if (state.tuning) await askDeepSeekKey();
  }));
  $('#edit-api-key').addEventListener('click', askDeepSeekKey);
  $('#api-key-close').addEventListener('click', () => $('#api-key-dialog').close('cancel'));
  $('#api-key-cancel').addEventListener('click', () => $('#api-key-dialog').close('cancel'));
  $('#deepseek-api-key').addEventListener('input', event => event.target.setCustomValidity(''));
  $('#api-key-form').addEventListener('submit', event => {
    event.preventDefault();
    const input = $('#deepseek-api-key');
    const key = input.value.trim();
    if (!key || key.length > 512 || /[^\x21-\x7e]/.test(key)) {
      const message = '请输入有效的 DeepSeek API key，不要包含空格或换行。';
      input.setCustomValidity(window.ModelI18n.language === 'en' ? window.ModelI18n.english(message) : message);
      input.reportValidity();
      return;
    }
    deepseekKey = key;
    input.value = '';
    $('#api-key-dialog').close('confirm');
  });
  $('#download-model').addEventListener('click',()=>{if(state.code)download(state.code,state.result.tuned?'recommended_model.py':'candidate_model.py','text/x-python;charset=utf-8');});$('#download-report').addEventListener('click',async()=>{try{const value=state.result.job_id?await api(`/api/jobs/${state.result.job_id}/report`):report();download(JSON.stringify(value,null,2),'research_report.json','application/json;charset=utf-8');}catch(error){toast(error.message);}});$('#copy-code').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(state.code);toast('完整模型代码已复制。');}catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents($('#code-preview'));selection.removeAllRanges();selection.addRange(range);toast('已选中代码，请按 Ctrl+C 复制。');}});$('#new-study').addEventListener('click',()=>{if(state.busy)return toast('请先等待或取消当前评估任务。');clearDownstream();state.sample=null;state.file=null;state.tuning=false;deepseekKey='';$('#file-input').value='';$('#file-summary').hidden=true;$('#upload-error').hidden=true;$('#start-analysis').disabled=true;$$('[name=tuning]').forEach(el=>el.checked=el.value==='off');$$('.tuning-choice').forEach(c=>c.classList.toggle('selected',$('input',c).checked));renderSettings();setStep(0);});
  $('#rule-search').addEventListener('input',renderRules);$('#rule-stage').addEventListener('change',renderRules);$('#rule-grade').addEventListener('change',renderRules);$('#connection-open').addEventListener('click',openConnection);$('#guide-connect').addEventListener('click',openConnection);$('#connection-close').addEventListener('click',()=>$('#connection-dialog').close());$('#connection-test').addEventListener('click',testConnection);
  $('#research-details').addEventListener('toggle',()=>{if($('#research-details').open&&state.result){renderFeatures();renderRecommendations();}});
  $('#module-count').textContent=Object.values(data.library).flat().length;hydrateIcons();renderWorkflow();renderExamples();renderLibrary();renderRules();if(location.protocol!=='file:')checkConnection().catch(()=>{});
  $('#refresh-history').addEventListener('click',loadHistory);
  $('#history-search').addEventListener('input',renderHistory);
  $('#history-status').addEventListener('change',renderHistory);
  if(location.hash==='#history')setView('history');
  else if(location.protocol!=='file:')loadHistory();
  const savedJob=new URLSearchParams(location.search).get('job');
  if(savedJob&&/^[a-f0-9]{32}$/.test(savedJob))api(`/api/jobs/${savedJob}`).then(async job=>{
    if(state.busy||state.sample||state.file||state.result)return;
    if(['running','queued'].includes(job.status)){
      setBusy(true);state.job=savedJob;showJob('analysis-job',job);
      try{await pollJob(savedJob,'analysis-job',4);}finally{setBusy(false);}
      return;
    }
    if(job.status!=='completed'||!job.result?.model_code||!job.result?.spec)throw Error(job.error||'该任务尚未生成完整模型结果。');
    state.result=job.result;state.horizon=job.result.horizon;state.code=job.result.model_code;state.tuning=!!job.result.tuned;
    state.baseResult=job.result.base_result||null;
    $$('[name=tuning]').forEach(el=>el.checked=(el.value==='on')===state.tuning);
    $$('.tuning-choice').forEach(c=>c.classList.toggle('selected',$('input',c).checked));
    $$('[data-horizon]').forEach(el=>{el.classList.toggle('selected',Number(el.dataset.horizon)===state.horizon);el.setAttribute('aria-pressed',String(Number(el.dataset.horizon)===state.horizon));});
    renderSettings();renderBanner();setStep(4);
  }).catch(error=>{uploadError(error.message);toast(error.message);});
})();
