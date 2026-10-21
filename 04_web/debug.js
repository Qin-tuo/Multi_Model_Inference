// SPDX-License-Identifier: Apache-2.0
// One existing WebSocket; bounded browser history, all inference timing from board.
(() => {
  'use strict';
  const modelNames={detection_segmentation:'检测分割',body_hand_pose:'人体/手部关键点',face:'人脸检测',face_identity:'人脸特征',expression:'表情',head_pose:'头姿',gaze_landmarks:'视线辅助关键点',gaze:'视线',person_reid:'行人特征',object_embedding:'物体特征'};
  const panel = document.getElementById('debugPanel');
  const toggle = document.getElementById('debugToggle');
  const labels = {vad:'VAD 分段',asr:'ASR 调用',llm:'LLM 迭代',tool:'工具执行',tts:'TTS 合成',release:'资源释放'};
  const card = (label,id,unit='') => `<div class="debug-card"><label>${label}</label><strong id="${id}">—</strong><small>${unit}</small></div>`;
  panel.innerHTML = `
    <div class="debug-top"><strong>板端实时状态</strong></div>
    <div class="debug-live stale" id="debugFresh" role="status">等待板端采样</div>
    <div class="debug-note">板端约 2 秒采样一次 · 趋势保留最近 60 个样本 · 每帧耗时展示最新有效结果</div>
    <div class="debug-section">视觉链路</div>
    <div class="debug-cards">
      ${card('板端采集','debugCapture','FPS · 相邻采样之间的实际增量')}
      ${card('板端识别','debugPerception','FPS · 完成人体／脸／手一整轮')}
      ${card('整轮处理','debugProcessing','ms · 最新有效帧')}
      ${card('结果帧龄','debugAge','ms · 主机收帧 → 板端采样')}
    </div>
    <div class="debug-note" id="debugVisionMeta"></div>
    <div class="debug-error debug-note" id="debugVisionError"></div>
    <div id="debugModelPanel">
    <div class="debug-note">各模型：最近 120 次成功调用的板端墙钟耗时，包含预处理／推理／后处理；未触发时显示 —。拒绝/失败包含输入质量检查，不等于 GPU 故障次数。</div>
    <table class="debug-table"><thead><tr><th>模型模块</th><th>成功调用</th><th>最新 ms</th><th>P50 / P95 ms</th><th>拒绝/失败</th></tr></thead><tbody id="debugModels"></tbody></table>
    </div>
    <canvas id="debugRateChart" class="debug-chart" aria-label="板端实际帧率趋势"></canvas>
    <div class="debug-legend"><span class="blue">● 采集 FPS</span><span class="green">● 识别 FPS</span></div>
    <canvas id="debugLatencyChart" class="debug-chart" aria-label="板端处理时间与结果帧龄趋势"></canvas>
    <div class="debug-legend"><span class="green">● 整轮处理 ms</span><span class="orange">● 结果帧龄 ms</span></div>
    <div class="debug-section">板端资源</div>
    <div class="debug-cards">
      ${card('CPU / GPU','debugLoad','% · GPU 为采样值，可能漏掉短任务')}
      ${card('系统可用内存','debugMemory','GiB · CPU / GPU 共享板载内存')}
      ${card('CPU / GPU 温度','debugTemp','°C · 板端 thermal 传感器')}
      ${card('GPU / EMC 频率','debugClocks','MHz · 当前频率')}
      ${card('模块输入功率','debugPower','W · VDD_IN；不含整机外设')}
      ${card('已用 Swap','debugSwap','GiB · 存量不等于正在换页')}
    </div>
    <div class="debug-section">当前／最近轮次</div>
    <div class="debug-note" id="debugTurn">尚无请求</div>
    <div class="debug-cards">
      ${card('请求接纳 → 首个回答内容','debugFirst','ms · 包含前置处理／工具轮次')}
      ${card('请求接纳 → 输出发布','debugPublished','ms · 文件完成或设备回执')}
      ${card('取消 → 资源释放','debugCancel','ms · 不代表实际扬声器静音')}
      ${card('最近完成轮首段 PCM','debugPCM','ms · 首个 TTS 调用 → 首个非空 PCM 块')}
    </div>
    <table class="debug-table"><thead><tr><th>阶段</th><th>调用数</th><th>累计 / ms</th></tr></thead><tbody id="debugStages"></tbody></table>
    <div class="debug-timeline" id="debugTimeline"></div>
    <div class="debug-note">时间线来自板端单调时钟；LLM 迭代含接口、缓冲及消费等待，TTS 可与它重叠。累计值不能相加当作总延迟，也不是 CUDA kernel 时间。</div>
    <div class="debug-note" id="debugLast"></div>
    <div class="debug-section">传输与采样</div>
    <div class="debug-note" id="debugTransport"></div>
    <div class="debug-note">浏览器接收 FPS 仅反映预览传输；板端帧龄不含曝光时间，也不含网络与屏幕显示延迟。缺失指标显示 —。</div>
    `;
  let enabled = location.pathname === '/debug', latest = null, history = [];
  let receivedAt = 0, connected = false, session = null, sequence = -1;
  let frameCount = 0, frameSince = performance.now(), browserFPS = null;
  const $ = id => document.getElementById(id);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const f = (x,n=1) => finite(x) ? x.toFixed(n) : '—';
  const put = (id,text) => { $(id).textContent = text; };
  const row = values => {
    const tr = document.createElement('tr');
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    return tr;
  };
  function freshness(){
    const age = receivedAt ? (performance.now()-receivedAt)/1000 : null;
    const stale = !connected || age === null || age + (latest?.cache_age_seconds||0) > 6;
    panel.dataset.stale = String(stale);
    $('debugFresh').classList.toggle('stale',stale);
    put('debugFresh',!connected ? '连接未就绪 · 历史读数不代表当前状态' : !latest ? '等待板端采样' :
      `${stale?'采样已过期':'板端数据已接收'} · 收到样本后 ${f(age)} s · #${latest.sequence}`);
  }
  function chart(id, series){
    const canvas=$(id), width=Math.max(200,canvas.clientWidth), height=100, dpr=window.devicePixelRatio||1;
    canvas.width=width*dpr;canvas.height=height*dpr;
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const values=history.flatMap(s=>series.map(x=>s[x.key])).filter(finite);
    const max=Math.max(1,...values)*1.15, left=38, bottom=height-18;
    ctx.font='10px sans-serif';ctx.fillStyle='#8b949e';ctx.fillText(f(max),2,12);ctx.fillText('0',16,bottom);
    ctx.strokeStyle='#283545';ctx.beginPath();ctx.moveTo(left,10);ctx.lineTo(left,bottom);ctx.lineTo(width-5,bottom);ctx.stroke();
    if(!history.length)return;
    const end=history[history.length-1].time, start=Math.min(end-1,history[0].time);
    ctx.fillText(`-${f(end-start,0)} s`,left,height-3);ctx.fillText('最新',width-30,height-3);
    for(const s of series){
      ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();let drawing=false;
      for(const point of history){
        if(!finite(point[s.key])){drawing=false;continue;}
        const x=left+(point.time-start)/(end-start)*(width-left-7),y=bottom-point[s.key]/max*(bottom-10);
        if(drawing)ctx.lineTo(x,y);else ctx.moveTo(x,y);drawing=true;
      }ctx.stroke();
    }
  }
  function render(){
    if(!enabled||!latest)return;
    const {vision:v,resources:r,turn:t}=latest, tr=t.trace, marks=tr?.marks_ms||{};
    put('debugCapture',f(v.capture_fps));put('debugPerception',f(v.perception_fps));
    put('debugProcessing',f(v.processing_ms));put('debugAge',f(v.frame_age_ms));
    put('debugVisionMeta',`目标：采集 ${f(v.capture_target_fps,0)} / 识别 ${f(v.perception_target_fps,0)} FPS · 帧 ${v.frame_id??'—'} · 已处理 ${v.processed??'—'} · 跳过采集帧 ${v.skipped_capture_frames??'—'} · 失败 ${v.failures??'—'} · 重连 ${v.reconnects??'—'}`);
    put('debugVisionError',v.ready?'':v.error||'暂无新鲜感知结果');
    const models=$('debugModels');models.replaceChildren();
    for(const [name,data] of Object.entries(v.model_metrics||{}))models.append(row([modelNames[name]||name,data.calls,f(data.last_ms,2),f(data.p50_ms,2)+' / '+f(data.p95_ms,2),data.errors||0]));
    put('debugLoad',`${f(r.cpu_percent)} / ${f(r.gpu_percent)}`);
    put('debugMemory',`${f(r.ram_available_gib,2)} / ${f(r.ram_total_gib,2)}`);
    put('debugTemp',`${f(r.temperatures_c['cpu-thermal'])} / ${f(r.temperatures_c['gpu-thermal'])}`);
    put('debugClocks',`${f(r.gpu_mhz,0)} / ${f(r.emc_mhz,0)}`);
    put('debugPower',f(r.module_input_w,2));put('debugSwap',f(r.swap_used_gib,2));
    const active=(tr?.spans||[]).filter(s=>s.outcome==='running').map(s=>labels[s.name]||s.name);
    put('debugTurn',`状态 ${t.phase||'—'}${active.length?' · '+active.join(' + '):''} · 待处理 ${t.pending??'—'}${tr?' · 轮次 '+tr.turn_id.slice(0,8)+' · '+tr.status+' · '+f(tr.elapsed_ms/1000)+' s · 排队 '+f(marks.dequeued,2)+' ms':''}`);
    put('debugFirst',f(marks.first_content));put('debugPublished',f(marks.output_published));
    put('debugCancel',f(finite(marks.cancel_requested)&&tr?.status!=='running'&&tr?.status!=='release_unconfirmed'?tr.elapsed_ms-marks.cancel_requested:null));
    put('debugPCM',f(t.last.first_tts_chunk_pcm_ms));
    const table=$('debugStages');table.replaceChildren();
    for(const [key,label] of Object.entries(labels)){
      const stage=tr?.stages[key], running=(tr?.spans||[]).filter(s=>s.name===key&&s.outcome==='running');
      table.append(row([label,stage||running.length?`${stage?.count||0}${running.length?' + '+running.length+' 进行中':''}`:'—',
        stage||running.length?f((stage?.duration_ms||0)+running.reduce((a,s)=>a+s.duration_ms,0)):'—']));
    }
    const timeline=$('debugTimeline');timeline.replaceChildren();
    for(const span of tr?.spans||[]){
      const line=document.createElement('div');line.className='debug-span';
      const label=document.createElement('span');label.textContent=labels[span.name]||span.name;
      const track=document.createElement('div');track.className='debug-span-track';
      const bar=document.createElement('div');bar.className='debug-span-bar '+span.outcome;
      bar.style.left=`${Math.min(100,span.start_ms/Math.max(1,tr.elapsed_ms)*100)}%`;
      bar.style.width=`${Math.min(100,span.duration_ms/Math.max(1,tr.elapsed_ms)*100)}%`;
      bar.title=`${f(span.start_ms)} ms → ${f(span.start_ms+span.duration_ms)} ms`;track.append(bar);
      const duration=document.createElement('span');duration.textContent=f(span.duration_ms)+' ms';
      line.append(label,track,duration);timeline.append(line);
    }
    put('debugLast',t.last.turn_id?`最近完成轮 ${t.last.turn_id.slice(0,8)}：${t.last.status} · 输出 ${t.last.output_kind||'无音频回执'} · 音频 ${f(t.last.audio_seconds)} s${t.last.error?' · '+t.last.error:''}${tr?.omitted_spans?' · 时间线省略较早 '+tr.omitted_spans+' 段，累计耗时保留':''}`:'最近完成轮：暂无');
    const w=latest.transport;
    put('debugTransport',`浏览器接收 ${f(browserFPS)} FPS · JPEG 缓存 ${f(v.buffer_mib,2)} MiB · 采样间隔 ${f(latest.interval_seconds,2)} s · 附加指标组装 ${f(latest.collection_ms,2)} ms · 查看端 ${w.clients??'—'} · 消息桥积压 ${w.bridge_events??'—'} · 丢弃 ${w.dropped??'—'} · 溢出 ${w.overflows??'—'}`);
    chart('debugRateChart',[{key:'capture',color:'#69b5ff'},{key:'perception',color:'#9bdd63'}]);
    chart('debugLatencyChart',[{key:'processing',color:'#9bdd63'},{key:'age',color:'#ffbc76'}]);
    freshness();
  }
  function setEnabled(value){
    enabled=value;panel.hidden=!value;document.body.classList.toggle('debug-mode',value);
    toggle.setAttribute('aria-pressed',String(value));toggle.textContent=value?'返回对话视图':'板端调试';
    requestAnimationFrame(render);
  }
  toggle.addEventListener('click',()=>setEnabled(!enabled));
  window.boardDebug={
    disconnect(){connected=false;frameCount=0;frameSince=performance.now();browserFPS=null;freshness();},
    receive(msg){
      if(msg.type==='frame')frameCount++;
      const data=msg.type==='diagnostics'?msg:msg.type==='capabilities'?msg.diagnostics:null;
      if(!data?.ready)return;
      if(data.session!==session){session=data.session;sequence=-1;history=[];frameCount=0;frameSince=performance.now();browserFPS=null;}
      if(data.sequence<=sequence)return;
      connected=true;sequence=data.sequence;latest=data;receivedAt=performance.now();
      const elapsed=(receivedAt-frameSince)/1000;
      if(elapsed>=1){browserFPS=frameCount/elapsed;frameCount=0;frameSince=receivedAt;}
      history.push({time:data.sampled_monotonic,capture:data.vision.capture_fps,perception:data.vision.perception_fps,
        processing:data.vision.processing_ms,age:data.vision.frame_age_ms,cpu:data.resources.cpu_percent,
        gpu:data.resources.gpu_percent,available_gib:data.resources.ram_available_gib,power_w:data.resources.module_input_w});
      if(history.length>60)history.shift();render();
    }
  };
  window.addEventListener('resize',()=>{if(enabled)render();});
  setInterval(()=>{if(enabled)freshness();},1000);
  setEnabled(enabled);
})();
