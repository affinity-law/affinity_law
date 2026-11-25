'use strict';

window.ModelPerformance = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const format = (value, percent = false) => Number.isFinite(value) ? (percent ? `${Number((value * 100).toPrecision(4)).toLocaleString('en-US',{maximumSignificantDigits:4})}%` : Number(value.toPrecision(5)).toLocaleString('en-US', {maximumSignificantDigits:5})) : '—';
  let revision = 0;
  function reset() { revision++; }

  function renderArtifacts(result, hooks) {
    let card = document.querySelector('#model-artifacts');
    if (!card) {
      card = document.createElement('section'); card.id = 'model-artifacts'; card.className = 'card model-artifacts';
      document.querySelector('#step-4 .export-layout').after(card);
    }
    const io = result.model_io || {input_steps:result.tuning_result?.time_lag||12, output_steps:result.horizon, variables:result.channels};
    card.innerHTML = `<div class="performance-heading"><h2>模型结构</h2>${result.topology_svg?'<button class="button secondary compact" id="download-topology">下载结构图</button>':''}</div>
      <div class="model-shape"><span>输入 <b>${io.input_steps}</b> 步</span><i>→</i><span>输出 <b>${io.output_steps}</b> 步</span><span>变量数 <b>${io.variables}</b></span></div>
      <div class="topology-image">${(result.topology_svg?window.ModelI18n.svg(result.topology_svg):'')||'<p>此历史结果未保存结构图。</p>'}</div>
      <div class="result-folder">${result.output_folder?`<div class="saved-result-row"><span>✓ 结果已保存到独立文件夹</span><button class="button secondary compact" id="copy-result-folder">复制文件夹路径</button></div><details><summary>查看保存位置与文件</summary><div class="folder-path"><input readonly aria-label="本次结果文件夹路径" value="${esc(result.output_folder)}"></div><ul>${(result.output_files||[]).map(name=>`<li translate="no">${esc(window.ModelI18n.english(name))}</li>`).join('')}</ul><p>完整训练与预测文件保留在同一目录的子文件夹中。</p><p>输入 [batch, ${io.input_steps}, ${io.variables}] → 输出 [batch, ${io.output_steps}, ${io.variables}]。</p></details>`:'<p>当前仅预览模型。启动本地服务后可自动计算并保存结果。</p>'}</div>`;
    card.querySelector('#download-topology')?.addEventListener('click',()=>hooks.download(window.ModelI18n.svg(result.topology_svg),'model_topology.svg','image/svg+xml;charset=utf-8'));
    card.querySelector('#copy-result-folder')?.addEventListener('click', async event => {
      const button = event.currentTarget;
      try { await navigator.clipboard.writeText(result.output_folder); button.textContent='已复制'; }
      catch { card.querySelector('.result-folder details').open=true; card.querySelector('input').select(); card.querySelector('#copy-result-folder').textContent='请按 Ctrl+C'; }
    });
  }

  function chartSVG(curve) {
    const w = 1000, h = 340, p = {left:76, right:20, top:30, bottom:52};
    const all = [...curve.actual, ...curve.predicted];
    let low = Math.min(...all), high = Math.max(...all);
    const margin = (high - low || Math.abs(high) || 1) * .08;
    low -= margin; high += margin;
    const x = i => p.left + i * (w-p.left-p.right) / Math.max(1, curve.actual.length-1);
    const y = value => p.top + (high-value) * (h-p.top-p.bottom) / (high-low);
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="测试集真实值与预测值对比，变量 ${curve.channel+1}，种子 ${curve.seed}"><rect width="100%" height="100%" fill="#fff"/>`;
    for (let i = 0; i < 5; i++) {
      const v = high - i * (high-low)/4, yy = y(v);
      svg += `<line x1="${p.left}" x2="${w-p.right}" y1="${yy}" y2="${yy}" stroke="#e8edf2" stroke-dasharray="4 5"/><text x="${p.left-10}" y="${yy+4}" text-anchor="end" fill="#687585" font-size="13">${esc(format(v))}</text>`;
    }
    for (let i = 0; i < Math.min(5, curve.actual.length); i++) {
      const n = Math.round(i*(curve.actual.length-1)/Math.max(1, Math.min(5,curve.actual.length)-1));
      svg += `<text x="${x(n)}" y="${h-30}" text-anchor="middle" fill="#687585" font-size="13">${curve.indices[n]}</text>`;
    }
    for (const [key, color, dash] of [['actual','#268c87',''],['predicted','#ee8833',' stroke-dasharray="6 3"']]) {
      const points = curve[key].map((v,i) => `${x(i)},${y(v)}`).join(' ');
      svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"${dash} stroke-linejoin="round"/>`;
      if (curve[key].length === 1) svg += `<circle cx="${x(0)}" cy="${y(curve[key][0])}" r="4" fill="${color}"/>`;
    }
    return svg + `<text x="${w/2}" y="${h-6}" text-anchor="middle" fill="#687585" font-size="13">原序列行号（从 1 开始）</text><text x="${p.left}" y="17" fill="#687585" font-size="13">原始数值</text><line id="prediction-cursor" y1="${p.top}" y2="${h-p.bottom}" stroke="#aaa" hidden/></svg>`;
  }

  function render(result, hooks) {
    const version = ++revision;
    renderArtifacts(result, hooks);
    let panel = document.querySelector('#performance-panel');
    if (!panel) {
      panel = document.createElement('section');
      panel.id = 'performance-panel'; panel.className = 'card performance-card';
      panel.setAttribute('aria-labelledby', 'performance-title');
      document.querySelector('#step-4 .export-layout').before(panel);
    }
    const e = result.evaluation;
    const comparison = result.tuning_comparison;
    panel.innerHTML = `<div class="performance-heading"><h2 id="performance-title">模型效果</h2><span class="chip ${e?'lilac':'neutral'}">${e?'测试集评估':'仅模型代码'}</span></div><div id="evaluation-job" hidden aria-live="polite"></div>`;
    if (!e) {
      panel.insertAdjacentHTML('beforeend', `<div class="performance-empty"><div class="performance-placeholder"><span>RMSE <b>—</b></span><span>MAE <b>—</b></span><span>WMAPE <b>—</b></span></div><p>${esc(result.evaluation_error || '此结果尚未完成训练评估。')}</p>${result.tuned?'':'<button class="button primary" id="evaluate-model">训练并查看效果</button>'}</div>`);
      panel.querySelector('#evaluate-model')?.addEventListener('click', hooks.evaluate);
      return;
    }
    panel.insertAdjacentHTML('beforeend', `<div class="performance-metrics">${[['rmse','均方根误差'],['mae','平均绝对误差'],['wmape','加权绝对百分比误差']].map(([key,name])=>`<article><span>${key.toUpperCase()} <small>${name}</small></span><strong>${format(e.metrics[key].mean,key==='wmape')}</strong></article>`).join('')}</div>
      <details class="seed-details"><summary>查看评估详情</summary><p>${esc(window.ModelI18n.source(e.label))}。${e.seed_count===1?'指标来自一次训练评估':`指标为 ${e.seed_count} 个随机种子的均值`}，使用完整测试集的 ${e.point_count.toLocaleString()} 个时间点、${e.channels} 个变量。</p><p>原始数据尺度，重叠窗口预测先平均后评分。WMAPE 显示百分比，报告保留比例。图表的变量或范围不影响上方指标。${result.tuning_comparison?.historical_l_rmse != null ? '最终推荐按 3par 协议与历史 L 的测试结果比较。' : result.tuning_comparison ? '本次配置按验证集 RMSE 选出。' : ''}</p><table><thead><tr><th>随机种子</th><th>RMSE</th><th>MAE</th><th>WMAPE</th></tr></thead><tbody>${e.per_seed.map(s=>`<tr><td>${s.seed}</td>${['rmse','mae','wmape'].map(k=>`<td>${format(s.metrics[k],k==='wmape')}</td>`).join('')}</tr>`).join('')}${e.seed_count>1?`<tr><td>标准差</td>${['rmse','mae','wmape'].map(k=>`<td>${format(e.metrics[k].std,k==='wmape')}</td>`).join('')}</tr>`:''}</tbody></table><a class="performance-link" href="?job=${encodeURIComponent(result.job_id)}">本次结果链接（可收藏）</a></details>
      <div class="prediction-heading"><h3>真实值与预测值</h3><div class="prediction-legend"><span><i class="actual-line"></i>真实值</span><span><i class="predicted-line"></i>预测值</span></div></div>
      <div class="prediction-controls"><label ${e.channels===1?'hidden':''}>变量 <select id="prediction-channel" aria-label="预测图变量">${Array.from({length:e.channels},(_,i)=>`<option value="${i}">变量 ${i+1}</option>`).join('')}</select></label><details class="chart-options" ${e.seed_count===1?'hidden':''}><summary>图表设置</summary><label>随机种子 <select id="prediction-seed" aria-label="预测图随机种子">${e.per_seed.map(s=>`<option>${s.seed}</option>`).join('')}</select></label><p>每段显示最多 600 个连续测试点。</p></details><button class="button secondary compact" id="download-prediction" disabled>下载对比图</button></div>
      <div id="prediction-chart" aria-live="polite"></div><p class="prediction-readout" id="prediction-readout">正在读取预测曲线…</p>
      <div class="prediction-pager"><button class="button secondary compact" id="prediction-prev" ${e.point_count<=600?'hidden':''} disabled>上一段</button><span id="prediction-range"></span><button class="button secondary compact" id="prediction-next" ${e.point_count<=600?'hidden':''} disabled>下一段</button></div>`);
    if (Number.isFinite(comparison?.historical_l_rmse) && Number.isFinite(comparison?.selected_rmse)) {
      panel.querySelector('.seed-details table').insertAdjacentHTML('beforebegin',
        `<p>历史 L 默认配置 RMSE <strong>${format(comparison.historical_l_rmse)}</strong> · 验证选中配置 RMSE <strong>${format(comparison.selected_rmse)}</strong> · RMSE 相对降幅 <strong>${format(comparison.rmse_gain,true)}</strong></p>`);
    }
    let curve, start = 0, serial = 0;
    const query = id => panel.querySelector('#'+id);
    async function load() {
      const request = ++serial;
      const seed = Number(query('prediction-seed').value), channel = Number(query('prediction-channel').value);
      const path = `/api/jobs/${encodeURIComponent(result.job_id)}/curve?seed=${seed}&channel=${channel}&start=${start}`;
      query('download-prediction').disabled = true;
      query('prediction-prev').disabled = query('prediction-next').disabled = true;
      query('prediction-chart').innerHTML = '<p class="performance-description">正在读取该种子与变量的实际预测…</p>';
      try {
        const loaded = await hooks.api(path);
        if (revision !== version || serial !== request) return;
        curve = loaded;
        query('prediction-chart').innerHTML = chartSVG(curve);
        query('prediction-readout').textContent = '移动到曲线上查看对应行的真实值与预测值。';
        query('prediction-range').textContent = `测试点 ${curve.start+1}–${curve.stop} / ${curve.total} · 原序列行 ${curve.indices[0]}–${curve.indices.at(-1)}`;
        query('prediction-prev').disabled = start === 0;
        query('prediction-next').disabled = curve.stop >= curve.total;
        query('download-prediction').disabled = false;
        query('download-prediction').onclick = async () => {
          const button = query('download-prediction'); button.disabled = true;
          try {
            const response = await fetch(path.replace('/curve?', '/curve.png?'), {signal:AbortSignal.timeout(30000)});
            if (!response.ok) throw Error('图片生成失败，请重试。');
            const blob = await response.blob();
            if (revision === version && serial === request) hooks.download(blob,`prediction_seed${seed}_variable${channel+1}_${curve.start+1}-${curve.stop}.png`,'image/png');
          } catch(error) { if (revision === version && serial === request) query('prediction-readout').textContent = error.message; }
          finally { if (revision === version && serial === request) button.disabled = false; }
        };
        query('prediction-chart').querySelector('svg').addEventListener('pointermove', event => {
          const svg = event.currentTarget, bounds = svg.getBoundingClientRect();
          const index = Math.max(0, Math.min(curve.actual.length-1, Math.round((((event.clientX-bounds.left)/bounds.width)*1000-76)/904*(curve.actual.length-1))));
          query('prediction-readout').textContent = `原序列第 ${curve.indices[index]} 行 · 真实值 ${format(curve.actual[index])} · 预测值 ${format(curve.predicted[index])} · 误差 ${format(curve.predicted[index]-curve.actual[index])}`;
          const cursor = query('prediction-cursor'), x = 76+index*904/Math.max(1,curve.actual.length-1);
          cursor.removeAttribute('hidden'); cursor.setAttribute('x1',x); cursor.setAttribute('x2',x);
        });
      } catch(error) {
        if (revision !== version || serial !== request) return;
        query('prediction-chart').textContent = error.message;
        query('prediction-readout').textContent = '未能读取曲线。指标已保留，可重新进入此步骤重试。';
      }
    }
    for (const id of ['prediction-seed','prediction-channel']) query(id).addEventListener('change',()=>{start=0;load();});
    query('prediction-prev').addEventListener('click',()=>{start=Math.max(0,start-600);load();});
    query('prediction-next').addEventListener('click',()=>{start+=600;load();});
    load();
  }
  return {render, reset};
})();
