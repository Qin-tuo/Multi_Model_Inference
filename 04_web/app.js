/*
 SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

(function(){
  const $ = id => document.getElementById(id);

  let speechReady = false;
  let activeTurn = null, cancelledTurn = null, pendingVoice = null;
  let sessionId = null;
  let ws = null, frameCount = 0, msgIdx = 0;
  const answers=new Map(), transcripts=new Map(), receipts=new Map(), finishedTurns=new Set();
  let reconnectTimer = null, pttActive = null, pipelineStage = 'connecting', faceBoxTimer = null;
  const camNatural = {w:640, h:480};
  let lastFaceBox = null, lastCentered = false, trackingEnabled = false;
  let faceHoldTimer = null;
  const FACE_HOLD_MS = 0;

  /* ── WebSocket ──────────────────── */
  function connect(){
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => {
      $('connStatus').className = 'conn-status ok';
      $('connStatus').innerHTML = '<span class="conn-dot"></span> Connected';
      if(reconnectTimer){ clearTimeout(reconnectTimer); reconnectTimer = null; }
    };
    ws.onclose = () => {
      window.boardDebug?.disconnect();
      $('connStatus').className = 'conn-status err';
      $('connStatus').innerHTML = '<span class="conn-dot"></span> Disconnected';
      $('liveBadge').className = 'live-badge off';
      pttActive = null;
      pipelineStage = 'connecting';
      renderPipelineStatus();
      $('pttBtn').disabled = true;
      $('pttBtn').className = 'ptt-btn live';
      $('pttBtn').title = 'Waiting for microphone state';
      $('pttLabel').textContent = 'Connecting...';
      $('micSlash').style.display = 'none';
      $('speakerSelect').disabled = true;
      setVolumeControls(false);
      $('volumeValue').textContent = '--%';
      $('speakerStatus').textContent = 'Offline';
      $('clearSession').disabled=true;
      $('sendText').disabled=true; $('sendAudio').disabled=true; $('cancelTurn').disabled=true;
      reconnectTimer = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      let msg;
      try{ msg = JSON.parse(ev.data); } catch(e){ return; }
      window.boardDebug?.receive(msg);
      if((msg.turn_id === cancelledTurn || finishedTurns.has(msg.turn_id)) && ['token','transcript'].includes(msg.type)) return;
      switch(msg.type){
        case 'frame':        onFrame(msg); break;
        case 'status':       onStatus(msg); break;
        case 'transcript':   onTranscript(msg); break;
        case 'token':        onToken(msg); break;
        case 'done':         onDone(msg); break;
        case 'session_cleared':onSessionCleared(msg); break;
        case 'session_state':onSession(msg.session); break;
        case 'action_receipt':onActionReceipt(msg.receipt,msg.turn_id); break;
        case 'actions_state':onActionsState(msg); break;
        case 'voice_state':onVoiceState(msg); break;
        case 'audio_state':onLiveAudio(msg); break;
        case 'cancelled':    cancelledTurn=msg.turn_id; onStatus({...msg,stage:'cancelling'}); break;
        case 'cancel_ack':   if(!msg.accepted && cancelledTurn===msg.turn_id) cancelledTurn=null; break;
        case 'stats':        onStats(msg); break;
        case 'perception':   onPerception(msg); break;
        case 'scene':        onScene(msg); break;
        case 'gesture':      onGesture(msg); break;
        case 'gesture_state':onGestureState(msg); break;
        case 'capabilities':onCapabilities(msg); break;
        case 'camera_state':cameraUnavailable(msg); break;
        case 'error': case 'overflow': $('inputError').textContent=msg.error || '请求失败'; break;
        case 'info':         onInfo(msg); break;
        case 'ptt_state':    onPttState(msg); break;
        case 'speaker_state':onSpeakerState(msg); break;
      }
    };
  }

  /* ── Frame handler ─────────────── */
  let pendingFaceBox = null;

  function onFrame(msg){
    const img = $('camImg');
    $('cameraState').textContent='就绪';

    const hasFace = msg.face_detected && msg.face_box;
    pendingFaceBox = hasFace ? msg.face_box : null;
    const faceLost = msg.face_detected === false;
    if(msg.centered != null) lastCentered = msg.centered;

    img.onload = function(){
      if(img.naturalWidth){ camNatural.w = img.naturalWidth; camNatural.h = img.naturalHeight; }
      const cv = $('faceOverlay');
      cv.width = cv.offsetWidth;
      cv.height = cv.offsetHeight;

      if(pendingFaceBox){
        lastFaceBox = pendingFaceBox;
        drawFaceBox(pendingFaceBox, lastCentered);
        cropFace(pendingFaceBox);
        showFaceCrop(true);
        updateTrackPanel(pendingFaceBox, lastCentered);
        resetFaceHold();
      } else if(faceLost){
        clearFaceBox();
        scheduleFaceHide();
      }
      drawScene(msg.persons || [], msg.hands || [], msg.objects || []);
    };

    img.src = 'data:image/jpeg;base64,' + msg.data;
    img.style.display = 'block';
    $('camPlaceholder').style.display = 'none';
    $('liveBadge').className = 'live-badge on';
    frameCount++;
    $('camInfo').textContent = `Frame #${frameCount}`;
  }

  /* ── Face crop visibility with hold timer ── */
  function resetFaceHold(){
    if(faceHoldTimer){ clearTimeout(faceHoldTimer); faceHoldTimer = null; }
  }

  function scheduleFaceHide(){
    if(faceHoldTimer) return;
    faceHoldTimer = setTimeout(() => {
      faceHoldTimer = null;
      showFaceCrop(false);
      setTrackIdle();
    }, FACE_HOLD_MS);
  }

  function showFaceCrop(visible){
    $('faceCropPlaceholder').style.display = visible ? 'none' : '';
    $('faceCropCanvas').style.display = visible ? '' : 'none';
  }

  /* ── Face tracking panel ───────── */
  function setTrackIdle(){
    const dot = $('trackDot'), lbl = $('trackLabel');
    dot.className = 'track-dot' + (trackingEnabled ? ' searching' : '');
    lbl.textContent = trackingEnabled ? 'Searching\u2026' : 'No face';
    lbl.style.color = 'var(--dim)';
    $('trackStats').textContent = trackingEnabled ? '等待新的有效人脸结果' : '人脸检测未启用';
    setAxis('axisX', 0, true); setAxis('axisY', 0, true);
  }

  function updateTrackPanel(box, centered){
    const cx = camNatural.w/2, cy = camNatural.h/2;
    const fx = (box[0]+box[2])/2, fy = (box[1]+box[3])/2;
    const ox = (fx - cx)/cx, oy = (fy - cy)/cy;   // -1..1

    const dot = $('trackDot'), lbl = $('trackLabel');
    if(centered){
      dot.className = 'track-dot centered';
      lbl.textContent = 'Centered'; lbl.style.color = 'var(--accent)';
    } else {
      dot.className = 'track-dot tracking';
      lbl.textContent = 'Tracking'; lbl.style.color = '#58a6ff';
    }
    $('trackStats').textContent = 'offset ' + signed(ox) + ' \u00b7 ' + signed(oy);
    setAxis('axisX', ox, centered);
    setAxis('axisY', oy, centered);
  }

  function signed(v){ const p = Math.round(Math.abs(v)*100); return (v>=0?'+':'\u2212')+p+'%'; }

  function setAxis(id, off, centered){
    const fill = $(id);
    const v = Math.max(-1, Math.min(1, off));
    const half = Math.abs(v) * 50;          // % of half-width
    fill.style.left = (v >= 0 ? 50 : 50 - half) + '%';
    fill.style.width = half + '%';
    fill.className = 'axis-fill' + (centered ? ' centered' : '');
  }

  function cropFace(box){
    const img = $('camImg');
    if(!img.naturalWidth) return;
    const cv = $('faceCropCanvas');
    const ctx = cv.getContext('2d');

    const x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    const fw = x2 - x1, fh = y2 - y1;
    if(fw < 5 || fh < 5) return;

    const pad = Math.max(fw, fh) * 0.25;
    const sx = Math.max(0, x1 - pad);
    const sy = Math.max(0, y1 - pad);
    const sw = Math.min(img.naturalWidth - sx, fw + pad * 2);
    const sh = Math.min(img.naturalHeight - sy, fh + pad * 2);

    const size = Math.max(sw, sh);
    cv.width = 120;
    cv.height = 120;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 120, 120);
    const scale = 120 / size;
    const dx = (120 - sw * scale) / 2;
    const dy = (120 - sh * scale) / 2;
    ctx.drawImage(img, sx, sy, sw, sh, dx, dy, sw * scale, sh * scale);
  }

  /* ── Pipeline / transcript / chat ── */
  function renderPipelineStatus(){
    let stage = pipelineStage;
    if(pttActive === false && (stage === 'connecting' || stage === 'muted' || stage === 'listening' || stage === 'recording')){
      stage = 'muted';
    } else if(pttActive === true && (stage === 'connecting' || stage === 'muted')){
      stage = 'listening';
    }

    const labels = {
      idle:'等待输入', connecting:'Connecting...', muted:'Mic muted', listening:'Listening...',
      recording:'Recording...', transcribing:'Transcribing...',
      thinking:'Thinking...', synthesizing:'生成语音文件...', speaking:'Speaking...',
      running:'处理中', cancelling:'正在停止', releasing:'等待资源释放', release_unconfirmed:'资源释放待确认',executing_tool:'执行本地操作'
    };
    const status = $('pipeStatus');
    status.textContent = labels[stage] || stage;
    status.className = 'pipe-status ' + (
      stage === 'muted' ? 'muted' : stage === 'listening' ? 'live' :
      stage === 'connecting' ? '' : 'busy'
    );
  }

  function onStatus(msg){
    if(msg.turn_id && finishedTurns.has(msg.turn_id) && msg.turn_id!==activeTurn) return;
    if(msg.stage==='idle' && msg.turn_id && activeTurn && msg.turn_id!==activeTurn) return;
    if(msg.turn_id && msg.stage !== 'idle') activeTurn=msg.turn_id;
    if(msg.stage === 'idle') activeTurn=null;
    const stages = $('pipeStages').querySelectorAll('.stage');
    for(let s of stages) s.classList.toggle('active', s.dataset.stage === msg.stage);
    pipelineStage = msg.stage;
    const busy=['running','thinking','transcribing','synthesizing','speaking','executing_tool','cancelling','releasing','release_unconfirmed'].includes(msg.stage);
    $('sendText').disabled=busy || !!pendingVoice; $('sendAudio').disabled=busy || !!pendingVoice || !speechReady;
    $('cancelTurn').disabled=!pendingVoice && (!activeTurn || ['cancelling','releasing','release_unconfirmed'].includes(msg.stage));
    $('clearSession').disabled=busy || !!pendingVoice || !sessionId || ws?.readyState!==WebSocket.OPEN;
    renderPipelineStatus();
  }

  function onTranscript(msg){
    if(!msg.turn_id || transcripts.has(msg.turn_id) || finishedTurns.has(msg.turn_id)) return;
    hideEmpty();
    const div = document.createElement('div');
    div.className = 'msg user';
    div.innerHTML = `<div class="content">${esc(msg.text)}</div>`
      + `<div class="meta">${msg.stt_time ? 'STT '+msg.stt_time.toFixed(1)+'s' : ''}`
      + `${msg.duration ? ' \u00b7 '+msg.duration.toFixed(1)+'s' : ''}</div>`;
    $('messages').appendChild(div);
    transcripts.set(msg.turn_id,div);
    scrollChat();
    msgIdx++;
    updateMsgCount();
  }

  function onToken(msg){
    if(!msg.turn_id || finishedTurns.has(msg.turn_id)) return;
    let currentAssistant=answers.get(msg.turn_id);
    if(!currentAssistant){
      hideEmpty();
      currentAssistant = document.createElement('div');
      currentAssistant.className = 'msg assistant';
      currentAssistant.innerHTML = '<div class="content cursor-blink"></div><div class="meta"></div>';
      $('messages').appendChild(currentAssistant);
      answers.set(msg.turn_id,currentAssistant);
    }
    currentAssistant.querySelector('.content').textContent += msg.text;
    scrollChat();
  }

  function onDone(msg){
    if(!msg.turn_id) return;
    for(const receipt of msg.actions_committed || []) onActionReceipt({...receipt,committed:true},msg.turn_id);
    if(!answers.has(msg.turn_id) && !finishedTurns.has(msg.turn_id)) onToken({...msg,text:msg.text || ''});
    const currentAssistant=answers.get(msg.turn_id);
    if(currentAssistant){
      if(typeof msg.text === 'string') currentAssistant.querySelector('.content').textContent=msg.text;
      currentAssistant.querySelector('.content').classList.remove('cursor-blink');
      const meta = currentAssistant.querySelector('.meta');
      const parts = [];
      if(msg.ttft != null) parts.push('TTFT '+msg.ttft.toFixed(1)+'s');
      if(msg.vlm_time != null) parts.push('VLM '+msg.vlm_time.toFixed(1)+'s');
      if(msg.total_time != null) parts.push('总计 '+msg.total_time.toFixed(1)+'s');
      if(msg.tokens != null) parts.push(msg.tokens+' tok');
      if(msg.status === 'failed') parts.push('未完成');
      if(msg.status === 'cancelled') parts.push('已取消');
      meta.textContent = parts.join(' \u00b7 ');
      if(msg.audio_url){
        const link=document.createElement('a'); link.href=msg.audio_url; link.textContent=' 下载语音文件';
        link.setAttribute('download','reply.wav'); meta.appendChild(link);
      } else if(msg.output_kind === 'device_completed'){
        meta.appendChild(document.createTextNode(' 设备输出完成'));
      }
      if(!finishedTurns.has(msg.turn_id)) msgIdx++;
      updateMsgCount();
    }
    finishedTurns.add(msg.turn_id);
    while(finishedTurns.size>100) finishedTurns.delete(finishedTurns.values().next().value);
    scrollChat();
  }

  /* ── Stats / Robot / Info ────── */
  function onStats(msg){
    if(Number.isFinite(msg.cpu)) setBar('cpu', msg.cpu);
    if(Number.isFinite(msg.gpu)) setBar('gpu', msg.gpu);
    if(Number.isFinite(msg.ram_used) && Number.isFinite(msg.ram_total) && msg.ram_total > 0){
      const ramPercent=msg.ram_used/msg.ram_total*100;
      $('ramBar').style.width=Math.min(100, ramPercent).toFixed(0)+'%';
      barColor($('ramBar'), ramPercent);
      $('ramVal').textContent=msg.ram_used.toFixed(1)+' / '+msg.ram_total.toFixed(1)+' GB';
      $('ramVal').title=$('ramVal').textContent;
    }
    if(msg.models) updateModels(msg.models);
    if(msg.platform){
      $('platformBadge').textContent=msg.platform;
      $('platformFooter').textContent=msg.platform;
      $('platformFooter').title=msg.platform;
      $('headerPlatform').textContent=msg.platform;
    }
    if(msg.clients != null) $('clientCount').textContent = msg.clients+' viewer'+(msg.clients!==1?'s':'');
  }

  function cameraUnavailable(msg){
    $('cameraState').textContent='不可用';
    $('camImg').style.display='none';
    $('camImg').onload=null;
    $('camImg').removeAttribute('src');
    $('camPlaceholder').style.display='';
    $('camPlaceholder').textContent=msg.error || '摄像头未启用';
    $('liveBadge').className='live-badge off';
    pendingFaceBox=null; lastFaceBox=null;
    clearTimeout(faceHoldTimer); faceHoldTimer=null;
    showFaceCrop(false);
    setTrackIdle();
    const c=$('faceOverlay'); c.getContext('2d').clearRect(0,0,c.width,c.height);
  }

  function onScene(msg){
    $('sceneInfo').textContent=msg.ready ? '人体 '+msg.persons.length+' · 人脸 '+(msg.face && msg.face.face_detected ? 1 : 0)+' · 手 '+msg.hands.length : (msg.error || '感知尚未就绪');
    $('visionTools').hidden=msg.backend!=='tensorrt_b1';
    if(msg.backend!=='tensorrt_b1') return;
    const update=(id,rows,empty)=>{const select=$(id),old=select.value;select.replaceChildren();
      if(empty){const o=document.createElement('option');o.value='';o.textContent='未选择';select.append(o);}
      for(const row of rows.filter(r=>r.track_id)){const o=document.createElement('option');o.value=row.track_id;
        o.textContent=(row.identity?.label || row.class_name || '人脸')+' #'+row.track_id.split(':').at(-1)+(row.range?.available ? ' · '+row.range.meters.toFixed(2)+' m':'');select.append(o);}
      if([...select.options].some(o=>o.value===old))select.value=old;};
    update('visionTarget',[...(msg.persons||[]),...(msg.objects||[])],true);
    update('visionFace',msg.faces||[],false);
    $('visionEnroll').disabled=!msg.ready || !$('visionFace').value;
    $('visionAttributes').textContent=(msg.attributes||[]).map(a=>{
      const h=a.head_pose?.angles_degrees?.map(v=>v.toFixed(1)).join(' / ') || '--';
      const g=a.gaze?.gaze_yaw_deg!=null ? a.gaze.gaze_yaw_deg.toFixed(1)+' / '+a.gaze.gaze_pitch_deg.toFixed(1):'不可用';
      return '人脸 #'+a.target_id.split(':').at(-1)+' · 头姿 '+h+'° · 视线 '+g+'° · 表情 '+(a.expression?.label || '--');
    }).join('；');
  }

  async function visionRequest(path,method,body){
    const response=await fetch(path,{method,headers:{'Content-Type':'application/json'},body:body ? JSON.stringify(body):undefined});
    const value=await response.json();if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail));return value;
  }
  $('visionSelectTarget').onclick=async()=>{try{await visionRequest('/api/vision/target','POST',{track_id:$('visionTarget').value||null});$('visionAction').textContent='目标选择已更新';}catch(e){$('visionAction').textContent=e.message;}};
  $('visionEnroll').onclick=async()=>{try{const row=await visionRequest('/api/vision/identities','POST',{target_id:$('visionFace').value,label:$('visionLabel').value});$('visionAction').textContent='已登记：'+row.label;}catch(e){$('visionAction').textContent=e.message;}};
  $('visionList').onclick=async()=>{try{const rows=await visionRequest('/api/vision/identities','GET');const area=$('visionIdentities');area.replaceChildren();
    for(const row of rows){const line=document.createElement('div'),text=document.createElement('span'),button=document.createElement('button');text.textContent=row.label+' ';button.textContent='移除';
      button.onclick=async()=>{try{await visionRequest('/api/vision/identities','DELETE',{label:row.label});line.remove();}catch(e){$('visionAction').textContent=e.message;}};line.append(text,button);area.append(line);}
    if(!rows.length)area.textContent='尚无登记标签';}catch(e){$('visionAction').textContent=e.message;}};

  function drawScene(persons,hands,objects=[]){
    const canvas=$('faceOverlay'),ctx=canvas.getContext('2d');
    const scale=Math.min(canvas.width/camNatural.w,canvas.height/camNatural.h);
    const ox=(canvas.width-camNatural.w*scale)/2,oy=(canvas.height-camNatural.h*scale)/2;
    function box(row,color,label){
      const b=row.box_xyxy;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.setLineDash([]);
      ctx.strokeRect(ox+b[0]*scale,oy+b[1]*scale,(b[2]-b[0])*scale,(b[3]-b[1])*scale);
      ctx.font='13px sans-serif';ctx.fillText(label,ox+b[0]*scale,Math.max(14,oy+b[1]*scale-4));
    }
    const label=(row,name)=>name+(row.track_id?' #'+row.track_id.split(':').at(-1):'')+(row.range?.available?' '+row.range.meters.toFixed(2)+'m':'');
    for(const person of persons) box(person,'#76b900',label(person,'人体'));
    for(const object of objects) box(object,'#64b5f6',label(object,object.class_name));
    const edges=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
    for(const hand of hands){
      box(hand,'#f0a64b','手');const points=hand.landmarks;
      ctx.strokeStyle='#f0a64b';ctx.lineWidth=2;ctx.beginPath();
      for(const [a,b] of edges){ctx.moveTo(ox+points[a][0]*scale,oy+points[a][1]*scale);ctx.lineTo(ox+points[b][0]*scale,oy+points[b][1]*scale);}
      ctx.stroke();ctx.fillStyle='#ffd180';
      for(const point of points){ctx.beginPath();ctx.arc(ox+point[0]*scale,oy+point[1]*scale,2,0,2*Math.PI);ctx.fill();}
    }
  }

  function onPerception(msg){
    trackingEnabled=!!msg.enabled;
    $('trackBadge').className='track-badge ' + (trackingEnabled ? 'on' : 'off');
    $('trackBadge').textContent=trackingEnabled ? 'ON' : 'OFF';
    if(!msg.face_detected) setTrackIdle();
  }

  function onGesture(msg){
    const name=msg.event?.kind==='wave' ? '挥手' : '举掌';
    const reasons={greeting_started:'已开始问候',turn_cancelled:'已停止本轮',busy:'当前忙碌，本次问候已忽略',
      cooldown:'问候间隔中',greeting_muted:'自动问候已静音',idle:'当前没有需要停止的任务',
      expired_or_wrong_source:'画面已失效',frame_precedes_turn:'画面早于本轮',duplicate:'已处理',not_ready:'服务未就绪'};
    $('gestureAction').textContent=name+'：'+(reasons[msg.reason] || msg.reason || '已处理');
  }

  function onGestureState(msg){
    const reasons={disabled:'手势未启用',awaiting_frames:'手势等待画面',stale_frame:'手势画面已过期',
      perception_unavailable:'手势感知暂不可用',out_of_order_frame:'手势等待连续画面',too_many_hands:'手部数量超出处理范围'};
    if(msg.error) $('gestureInfo').textContent='手势处理异常：'+msg.error;
    else if(!msg.ready) $('gestureInfo').textContent=reasons[msg.reason] || '手势未就绪';
    else{
      const hands=msg.tracks || [];
      const holding=hands.filter(h=>h.open_palm && !h.latched);
      $('gestureInfo').textContent=holding.length ? '正在观察张掌，静止保持片刻可停止' :
        hands.some(h=>h.latched) ? '本次举手已处理，收手后可再次触发' :
        msg.rejected_hands?.clipped_hand ? '请将手腕和五指完整放入画面' :
        msg.rejected_hands?.small_hand ? '手部太小，请适当靠近摄像头' : '手势已就绪';
    }
    if(msg.last_action) onGesture(msg.last_action);
  }

  function onCapabilities(msg){
    if(msg.face) onPerception(msg.face);
    if(msg.voice) onVoiceState(msg.voice);
    if(msg.live_audio) onLiveAudio(msg.live_audio);
    if(msg.session) onSession(msg.session);
    if(msg.actions) onActionsState(msg.actions);
    updateModels(msg.models || {});
    if(msg.perception) onScene(msg.perception);
    if(msg.gestures) onGestureState({...msg.gestures,last_action:msg.last_gesture});
    $('useCamera').disabled=!msg.camera_enabled;
    if(!msg.camera_enabled) $('useCamera').checked=false;
    $('cameraState').textContent=msg.camera_ready ? '就绪' : '未启用/不可用';
    if(!msg.camera_ready) cameraUnavailable({error:msg.camera_error});
    speechReady=!!msg.speech_ready;
    $('pttBtn').style.display=msg.audio_enabled ? '' : 'none';
    if(msg.audio_enabled) $('pttBtn').style.display='flex';
    $('audioFile').disabled=!speechReady;
    $('audioMode').textContent=speechReady ? '16kHz 单声道 PCM WAV；输出：'+(msg.audio_output_mode==='file' ? '文件' : '设备') : '语音未启用';
    $('inputMode').textContent=msg.audio_enabled ? '麦克风与文字' : speechReady ? '语音文件与文字' : '文字';
    activeTurn=msg.active_turn;
    onStatus({stage:msg.phase || (msg.active_turn ? 'thinking' : 'idle'),turn_id:msg.active_turn});
    if(msg.last_result){
      onDone({...msg.last_result,ttft:msg.last_result.first_content_seconds,
        total_time:msg.last_result.elapsed_seconds,output_kind:msg.last_result.speech?.receipt?.kind});
    }
  }

  $('audioForm').addEventListener('submit',async function(event){
    event.preventDefault(); $('inputError').textContent='';
    const file=$('audioFile').files[0];
    if(!file){ $('inputError').textContent='请选择 WAV 文件'; return; }
    $('sendAudio').disabled=true; $('sendText').disabled=true;
    try{
      const response=await fetch('/api/audio?use_image='+$('useCamera').checked+'&request_id='+crypto.randomUUID().replaceAll('-',''),
        {method:'POST',headers:{'Content-Type':'audio/wav'},body:file});
      const data=await response.json();
      if(!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : '语音输入无效');
    }catch(error){
      $('inputError').textContent=error.message; $('sendAudio').disabled=!speechReady; $('sendText').disabled=false;
    }
  });

  $('cancelTurn').addEventListener('click',async function(){
    const turn=pendingVoice || activeTurn;
    if(!turn) return;
    cancelledTurn=turn;
    $('cancelTurn').disabled=true;
    try{
      if(ws?.readyState===WebSocket.OPEN){
        ws.send(JSON.stringify({type:'cancel',turn_id:turn}));
        return;
      }
      const response=await fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({turn_id:turn})});
      const result=await response.json();
      if(!response.ok) throw new Error(result.detail || '停止请求失败');
      // WebSocket carries the ordered lifecycle. A delayed HTTP response must
      // not rewind an already-idle/new turn back to cancelling.
      if(!result.accepted && cancelledTurn===turn) cancelledTurn=null;
    }catch(error){
      if(cancelledTurn===turn) cancelledTurn=null;
      $('inputError').textContent=error.message; $('cancelTurn').disabled=false;
    }
  });

  function onActionsState(msg){
    $('actionsInfo').textContent=!msg.enabled ? '本地笔记与提醒未启用' : msg.error ? '本地操作异常：'+msg.error :
      '已保存 '+msg.notes+' 条笔记；最近提醒：'+(msg.reminders || []).map(r=>r.text+'（'+({scheduled:'待到时',due:'等待输出',claimed:'等待输出',delivering:'输出中',delivered:'已输出',stopped:'已停止本轮输出',cancelled:'已取消',expired:'已过期',uncertain:'输出结果待确认',failed:'失败'}[r.status] || r.status)+'）').join('、');
  }

  function onVoiceState(msg){
    pendingVoice=msg.pending?.turn_id || null;
    const reasons={input_epoch_changed:'输入已切换或静音，请重新讲话',pending_voice_expired:'等待超时，请重新讲话',
      utterance_too_long:'语音过长，请分句重说',utterance_rejected:'未取得有效语音',input_stopped:'语音输入已停止',
      user:'已停止待处理语音',stationary_palm:'举掌已停止待处理语音',superseded_by_new_voice:'已改为听取新语音'};
    $('voiceInfo').textContent=msg.pending ? (msg.pending.phase==='capturing' ? '正在听取新指令' : '语音已收到，等待上一轮资源释放') :
      msg.last?.outcome==='discarded' ? (reasons[msg.last.reason] || '语音未提交：'+msg.last.reason) : '';
    onStatus({stage:pipelineStage,turn_id:activeTurn});
  }

  function onLiveAudio(msg){
    const input=msg.input;
    $('liveAudioInfo').textContent=!msg.enabled ? '' : !input?.alive ? '麦克风输入已停止：'+(input?.error || '请检查设备并重启应用') :
      input.suppressed_reason ? '半双工：本轮结束后恢复听取' :
      msg.interruptions_available ? '可以讲话打断当前轮次' : '半双工：语音打断未启用，或回声消除尚未就绪';
    if(msg.enabled && !input?.alive) $('pttBtn').style.display='none';
  }

  function onActionReceipt(receipt,turn){
    const key=turn+':'+receipt.position;
    if(receipts.has(key)) return;
    hideEmpty();const div=document.createElement('div');div.className='msg assistant';
    const body=document.createElement('div');body.className='content';
    const value=receipt.result || {};
    const labels={create_note:'笔记',list_notes:'查询笔记',create_reminder:'提醒',list_reminders:'查询提醒',cancel_reminder:'取消提醒'};
    body.textContent=receipt.committed ? (labels[receipt.name] || '本地操作')+'已执行：'+
      (value.title || value.text || ({cancelled:'已取消',not_found:'未找到',already_delivered:'已经完成输出，原回执保留',already_expired:'已经过期，记录保留'}[value.status]) || '查询完成') : '本地操作未执行：'+(value.error || '参数无效');
    div.appendChild(body);$('messages').appendChild(div);receipts.set(key,div);scrollChat();
  }

  function onSession(session){
    sessionId=session.id;
    $('sessionInfo').textContent=session.enabled ? '本次会话保留 '+session.retained_turns+' 轮文字，关闭应用后清空' : '会话上下文已关闭';
  }

  function onSessionCleared(msg){
    onSession(msg.session);
    $('messages').replaceChildren();answers.clear();transcripts.clear();receipts.clear();finishedTurns.clear();msgIdx=0;updateMsgCount();
    $('inputError').textContent='';cancelledTurn=null;
    onStatus({stage:'idle'});
  }

  $('clearSession').addEventListener('click',async function(){
    if(!sessionId) return;
    $('clearSession').disabled=true;
    $('sendText').disabled=true; $('sendAudio').disabled=true;
    try{
      const response=await fetch('/api/session/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId})});
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail || '清空失败');
      // Ordered WebSocket event owns clearing; a late HTTP reply must not erase a new turn.
    }catch(error){$('inputError').textContent=error.message; onStatus({stage:pipelineStage});}
  });

  $('textForm').addEventListener('submit', async function(event){
    event.preventDefault();
    $('inputError').textContent=''; $('sendText').disabled=true;
    try{
      const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:$('textInput').value,use_image:$('useCamera').checked,request_id:crypto.randomUUID().replaceAll('-','')})});
      const data=await response.json();
      if(!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : '输入无效');
      $('textInput').value='';
    }catch(error){
      $('inputError').textContent=error.message; $('sendText').disabled=false;
    }
  });

  function onInfo(msg){
    if(msg.models) updateModels(msg.models);
    if(msg.platform){ $('platformBadge').textContent=msg.platform; $('platformFooter').textContent=msg.platform; $('platformFooter').title=msg.platform; $('headerPlatform').textContent=msg.platform; }
  }

  function updateModels(m){
    if(m.stt) $('modelSTT').textContent=m.stt;
    if(m.vlm) $('modelVLM').textContent=m.vlm;
    if(m.tts) $('modelTTS').textContent=m.tts;
    if(m.vad) $('modelVAD').textContent=m.vad;
  }

  function onPttState(msg){
    pttActive = Boolean(msg.active);
    const btn=$('pttBtn'), lbl=$('pttLabel'), slash=$('micSlash');
    btn.disabled=false;
    if(pttActive){
      btn.className='ptt-btn live';
      btn.title='Mute the microphone';
      btn.setAttribute('aria-label', 'Mute microphone');
      btn.setAttribute('aria-pressed', 'false');
      lbl.textContent='Mute mic';
      slash.style.display='none';
    } else {
      btn.className='ptt-btn muted';
      btn.title='Unmute the microphone';
      btn.setAttribute('aria-label', 'Unmute microphone');
      btn.setAttribute('aria-pressed', 'true');
      lbl.textContent='Unmute mic';
      slash.style.display='';
    }
    renderPipelineStatus();
  }

  function onSpeakerState(msg){
    const select=$('speakerSelect'), status=$('speakerStatus'), volume=$('speakerVolume');
    const speakers=Array.isArray(msg.speakers) ? msg.speakers : [];
    select.replaceChildren();

    if(!speakers.length){
      const option=document.createElement('option');
      option.textContent='No speaker outputs found';
      select.appendChild(option);
      select.disabled=true;
      setVolumeControls(false);
      $('volumeValue').textContent='--%';
      status.textContent=msg.error || '';
      status.className='speaker-status err';
      return;
    }

    for(const speaker of speakers){
      const option=document.createElement('option');
      option.value=speaker.id;
      option.textContent=speaker.label || speaker.id;
      option.selected=speaker.id===msg.selected;
      select.appendChild(option);
    }
    select.disabled=false;
    if(Number.isFinite(msg.volume)){
      const value=clampVolume(msg.volume);
      volume.value=value;
      $('volumeValue').textContent=value+'%';
      setVolumeControls(true);
    } else {
      setVolumeControls(false);
      $('volumeValue').textContent='--%';
    }
    status.textContent=msg.error || (msg.aec_enabled ? 'AEC Active' : 'Direct');
    status.className='speaker-status' + (msg.error ? ' err' : '');
  }

  $('pttBtn').addEventListener('click', function(){
    if(!ws || ws.readyState !== WebSocket.OPEN || pttActive === null) return;
    this.disabled=true;
    ws.send(JSON.stringify({type:'ptt', active:!pttActive}));
  });

  $('speakerSelect').addEventListener('change', function(){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    this.disabled=true;
    setVolumeControls(false);
    $('speakerStatus').textContent='Switching...';
    $('speakerStatus').className='speaker-status';
    ws.send(JSON.stringify({type:'set_speaker', sink:this.value}));
  });

  let volumeTimer=null;
  function clampVolume(value){ return Math.max(0, Math.min(100, Math.round(Number(value)||0))); }
  function setVolumeControls(enabled){
    $('speakerVolume').disabled=!enabled;
    $('volumeDown').disabled=!enabled;
    $('volumeUp').disabled=!enabled;
  }
  function sendVolume(value){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({type:'set_speaker_volume', volume:clampVolume(value)}));
  }
  function setVolumeFromUi(value, sendNow){
    const volume=clampVolume(value);
    $('speakerVolume').value=volume;
    $('volumeValue').textContent=volume+'%';
    if(volumeTimer){ clearTimeout(volumeTimer); volumeTimer=null; }
    if(sendNow) sendVolume(volume);
    else volumeTimer=setTimeout(()=>{ volumeTimer=null; sendVolume(volume); }, 100);
  }
  $('speakerVolume').addEventListener('input', function(){ setVolumeFromUi(this.value, false); });
  $('speakerVolume').addEventListener('change', function(){ setVolumeFromUi(this.value, true); });
  $('volumeDown').addEventListener('click', ()=>setVolumeFromUi(Number($('speakerVolume').value)-5, true));
  $('volumeUp').addEventListener('click', ()=>setVolumeFromUi(Number($('speakerVolume').value)+5, true));

  /* ── Resizable camera/control split ── */
  const splitKey='edge-assistant-layout-split';
  const splitter=$('layoutSplitter'), leftPane=document.querySelector('.left'), main=document.querySelector('.main');
  function setSplit(percent, persist){
    if(window.innerWidth <= 768) return;
    const width=main.getBoundingClientRect().width;
    const splitterWidth=splitter.getBoundingClientRect().width;
    const minPercent=Math.max(25, 260/width*100);
    const maxPercent=Math.min(70, (width-340-splitterWidth)/width*100);
    const value=Math.max(minPercent, Math.min(maxPercent, Number(percent)||48));
    leftPane.style.flexBasis=value+'%';
    splitter.setAttribute('aria-valuenow', Math.round(value));
    if(persist) localStorage.setItem(splitKey, value.toFixed(2));
  }
  function splitFromPointer(event){
    const bounds=main.getBoundingClientRect();
    setSplit((event.clientX-bounds.left)/bounds.width*100, true);
  }
  splitter.addEventListener('pointerdown', event=>{
    splitter.setPointerCapture(event.pointerId);
    splitter.classList.add('dragging');
    document.body.classList.add('resizing');
    splitFromPointer(event);
  });
  splitter.addEventListener('pointermove', event=>{
    if(splitter.hasPointerCapture(event.pointerId)) splitFromPointer(event);
  });
  function endSplit(event){
    if(splitter.hasPointerCapture(event.pointerId)) splitter.releasePointerCapture(event.pointerId);
    splitter.classList.remove('dragging');
    document.body.classList.remove('resizing');
  }
  splitter.addEventListener('pointerup', endSplit);
  splitter.addEventListener('pointercancel', endSplit);
  splitter.addEventListener('dblclick', ()=>setSplit(48, true));
  splitter.addEventListener('keydown', event=>{
    if(event.key!=='ArrowLeft' && event.key!=='ArrowRight') return;
    event.preventDefault();
    const current=parseFloat(leftPane.style.flexBasis)||48;
    setSplit(current+(event.key==='ArrowRight'?2:-2), true);
  });
  const savedSplit=parseFloat(localStorage.getItem(splitKey));
  setSplit(Number.isFinite(savedSplit) ? savedSplit : 48, false);
  window.addEventListener('resize', ()=>setSplit(parseFloat(leftPane.style.flexBasis)||48, false));

  /* ── Drawing helpers ───────────── */
  function drawFaceBox(box, centered){
    const cv=$('faceOverlay'), ctx=cv.getContext('2d');
    const cw=cv.width, ch=cv.height;
    if(!cw||!ch) return;
    const nw=camNatural.w, nh=camNatural.h, s=Math.min(cw/nw,ch/nh);
    const ox=(cw-nw*s)/2, oy=(ch-nh*s)/2;
    ctx.clearRect(0,0,cw,ch);

    // Center target crosshair
    const tcx=ox+nw*s/2, tcy=oy+nh*s/2, r=10;
    ctx.strokeStyle='rgba(255,255,255,.35)'; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(tcx-r,tcy); ctx.lineTo(tcx+r,tcy);
    ctx.moveTo(tcx,tcy-r); ctx.lineTo(tcx,tcy+r);
    ctx.stroke();

    // Face box (green when centered, blue while tracking)
    const color = centered ? '#76b900' : '#58a6ff';
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash([6,3]);
    ctx.strokeRect(ox+box[0]*s, oy+box[1]*s, (box[2]-box[0])*s, (box[3]-box[1])*s);
    ctx.setLineDash([]);

    // Line from face center toward target center
    const fcx=ox+(box[0]+box[2])/2*s, fcy=oy+(box[1]+box[3])/2*s;
    ctx.strokeStyle=color; ctx.globalAlpha=.5; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(fcx,fcy); ctx.lineTo(tcx,tcy); ctx.stroke();
    ctx.globalAlpha=1;

    if(faceBoxTimer) clearTimeout(faceBoxTimer);
    faceBoxTimer = setTimeout(()=>ctx.clearRect(0,0,cw,ch), 600);
  }

  function clearFaceBox(){
    const cv=$('faceOverlay'), ctx=cv.getContext('2d');
    ctx.clearRect(0,0,cv.width,cv.height);
    if(faceBoxTimer){ clearTimeout(faceBoxTimer); faceBoxTimer=null; }
  }

  function scrollChat(){
    const m=$('messages');
    while(m.children.length>100) m.firstElementChild.remove();
    for(const map of [answers,transcripts,receipts]) for(const [key,node] of map) if(!node.isConnected) map.delete(key);
    m.scrollTop=m.scrollHeight;
  }
  function hideEmpty(){ const e=$('emptyState'); if(e) e.remove(); }
  function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
  function setBar(prefix, percent){
    const value=Math.max(0, Math.min(100, Number(percent)));
    const bar=$(prefix+'Bar');
    bar.style.width=value.toFixed(0)+'%';
    barColor(bar, value);
    $(prefix+'Val').textContent=value.toFixed(0)+'%';
  }
  function barColor(element, percent){
    element.className='stat-bar-fill '+(percent>90?'red':percent>70?'amber':'green');
  }
  function updateMsgCount(){ $('msgCount').textContent=msgIdx+' message'+(msgIdx!==1?'s':''); }

  connect();
})();
