// Missing DBH is a visible No, never zero or a copied neighbour's diameter.
window.ManualDbh = (() => {
  const drafts=new Map(),statuses=new Map();let saving=false;
  const missing=value=>value==null||['','no','未获取','未测得'].includes(String(value).trim().toLowerCase());
  const geometry=[['height','树高','m'],['crown_width','冠幅','m'],['horizontal_distance','水平距离','m'],['vertical_distance','垂直距离','m'],['clearance_distance','三维最小净空','m']];
  function number(value){
    if(missing(value))return null;
    const n=Number(value);
    if(!Number.isFinite(n)||n<=0||n>500)throw Error('胸径请填 No 或大于 0、不超过 500 的现场实测值（cm）');
    return n;
  }
  function display(value){return missing(value)?'No':Number(value).toFixed(2)}
  function label(result){return result?.dbh_input?.source==='MANUAL_FIELD_CORRECTION'?'人工修正（点云原值另存）':result?.dbh_input?.source==='MANUAL_FIELD_MEASUREMENT'?'人工实测':result?.dbh_input?.source==='ORIGINAL_LAS_1P30M'?'原始点云 1.30 m':result?.dbh_input?.value_cm==null?'No（未测量）':'输入数据'}
  function rootKey(key){return`${integratedContext?.project_root||''}:${integratedContext?.analysis_run_id||''}:${key}`}
  function valueFor(tree){return drafts.get(rootKey(tree.tree_key))??tree.manual_diameter_cm??'未获取'}
  function original(tree){return Object.hasOwn(tree,'point_cloud_diameter')?tree.point_cloud_diameter:tree.diameter}
  function field(tree,index){
    const auto=original(tree),known=!missing(auto);
    return`<label class="manual-dbh-field">第 ${index} 棵胸径 (cm)<input type="text" inputmode="decimal" data-manual-dbh-key="${esc(tree.tree_key)}" value="${esc(drafts.get(rootKey(tree.tree_key))??(known?tree.diameter:valueFor(tree)))}" aria-label="第 ${index} 棵胸径"><small>${known?'原始点云 1.30 m，原值 '+display(auto)+' cm；可人工修正用于预测，不覆盖点云；清空恢复原值':'未获取：请在此直接填写现场 1.30 m 实测胸径；无实测值可保留未获取'}</small></label>`;
  }
  function geometryFields(tree,index){
    const raw=tree.point_cloud_measurements||tree;
    return geometry.map(([key,label,unit])=>`<label class="manual-geometry-field">第 ${index} 棵${label} (${unit})<input type="text" inputmode="decimal" data-manual-geometry-key="${esc(tree.tree_key)}" data-measurement-field="${key}" value="${esc(drafts.get(rootKey(tree.tree_key)+':'+key)??tree[key]??'未获取')}" aria-label="第 ${index} 棵${label}"><small>${missing(raw[key])?'点云未获取，可补填现场实测值；不是自动测量':`点云原值 ${esc(raw[key])} ${unit}；可人工修正用于预测；清空恢复原值`}</small></label>`).join('');
  }
  function profileFields(tree,index){
    const speciesKey=rootKey(tree.tree_key)+':species',ageKey=rootKey(tree.tree_key)+':age';
    const species=drafts.get(speciesKey)??tree.species??'其他',age=drafts.get(ageKey)??tree.age??20;
    return `<div class="grid integrated-tree-profile"><label>第 ${index} 棵树种（可选择或输入）<input data-integrated-tree-species="${esc(tree.tree_key)}" list="integratedSpeciesOptions" value="${esc(species)}"></label><label>第 ${index} 棵树龄（年，可修改）<input data-integrated-tree-age="${esc(tree.tree_key)}" type="number" min="0" max="1000" step="1" value="${esc(age)}"></label><div><button class="secondary" type="button" data-estimate-integrated-age="${esc(tree.tree_key)}">按树高和胸径估算树龄</button><small data-integrated-age-status="${esc(tree.tree_key)}">公式：A≈0.60×D/gD + 0.40×H/gH；树种参数按本地公式库匹配。冠幅仅作为预测参数，不强行代入未校准树龄公式。</small></div></div>`;
  }
  function selectedTrees(){
    const rows=[];
    const first=(integratedContext?.trees||[]).find(t=>t.tree_key===$('integratedTree').value);
    if(first)rows.push(first);
    document.querySelectorAll('[data-integrated-extra-tree]').forEach(select=>{
      const tree=(integratedContext?.trees||[]).find(t=>t.tree_key===select.value);
      if(tree&&!rows.some(item=>item.tree_key===tree.tree_key))rows.push(tree);
    });
    return rows;
  }
  function measurementCard(tree,index){
    const raw=tree.point_cloud_measurements||tree;
    return `<section class="selected-tree-measurement-card" data-selected-tree-measurement="${esc(tree.tree_key)}"><h4>第 ${index} 棵 · ${esc(tree.tree_code)} · ${esc(tree.persistent_tree_id||tree.tree_key)}</h4><div class="integrated-tree-metrics"><div><span>树高（点云）</span><b>${integratedNumber(raw.height)} m</b></div><div><span>胸径（点云）</span><b>${original(tree)==null?'未获取':display(original(tree))+' cm'}</b></div><div><span>冠幅（点云）</span><b>${integratedNumber(raw.crown_width)} m</b></div><div><span>水平 / 垂直距离</span><b>${integratedNumber(raw.horizontal_distance)} / ${integratedNumber(raw.vertical_distance)} m</b></div><div><span>三维最小净空</span><b>${integratedNumber(raw.clearance_distance)} m</b></div></div>${profileFields(tree,index)}<div class="grid">${field(tree,index)}${geometryFields(tree,index)}</div></section>`;
  }
  function render(){
    if(!integratedContext)return;
    const trees=selectedTrees(),first=trees[0];
    const holder=$('integratedDbhFields');
    if(holder&&first)holder.innerHTML=`<h3>已选树木的点云参数与人工补填（按选择顺序）</h3><p>每棵树的树高、胸径、冠幅、水平距离、垂直距离和三维最小净空都直接列在对应树号下方。未获取项可人工填写；人工值会标明来源，不伪装成点云值。</p>${trees.map((tree,index)=>measurementCard(tree,index+1)).join('')}<button class="secondary" type="button" id="saveManualDbh" ${saving?'disabled':''}>保存全部人工补填与修正</button><span id="manualDbhStatus" role="status">${esc(statuses.get(rootKey(first.tree_key))||'')}</span>`;
    if($('saveManualDbh'))$('saveManualDbh').onclick=async()=>{
      if(saving)return;saving=true;const key=rootKey(first.tree_key);
      const update=text=>{statuses.set(key,text);if(rootKey($('integratedTree').value)===key&&$('manualDbhStatus'))$('manualDbhStatus').textContent=text};
      update('正在保存…');$('saveManualDbh').disabled=true;
      try{await saveAll();update('已保存在当前项目，重新打开仍可使用。');renderProcessedData();render()}catch(error){update(error.message)}
      finally{saving=false;if($('saveManualDbh'))$('saveManualDbh').disabled=false}
    };
  }
  async function saveAll(){
    const context=integratedContext;
    if(!context)throw Error('尚未连接当前项目');
    const scope={expected_project_root:context.project_root,expected_analysis_run_id:context.analysis_run_id};
    // Context polling replaces the object without changing the project/run.
    // Compare stable identity, not JavaScript object identity, and write into
    // the latest context after the response so its refreshed data stays current.
    const ensureCurrent=()=>{
      if(!integratedContext||integratedContext.project_root!==scope.expected_project_root||integratedContext.analysis_run_id!==scope.expected_analysis_run_id)
        throw Error('项目或数据已切换，请在当前页面重新保存补填数据');
      return integratedContext;
    };
    const inputs=[...document.querySelectorAll('[data-manual-dbh-key]:not([readonly])')];
    const values=inputs.map(input=>({tree_key:input.dataset.manualDbhKey,manual_diameter_cm:number(input.value)})).filter(body=>{
      const tree=(context.trees||[]).find(t=>t.tree_key===body.tree_key);
      return missing(original(tree)) || body.manual_diameter_cm!==tree.diameter;
    });
    const supplemental=new Map();
    document.querySelectorAll('[data-manual-geometry-key]').forEach(input=>{
      const key=input.dataset.manualGeometryKey,field=input.dataset.measurementField,value=missing(input.value)?null:Number(input.value);
      if(value!==null&&(!Number.isFinite(value)||value<0||(['height','crown_width'].includes(field)&&value<=0)))throw Error('树高和冠幅必须大于 0，距离必须大于等于 0；未获取可留空');
      const tree=(context.trees||[]).find(t=>t.tree_key===key);
      if(!missing(tree?.[field])&&value===Number(tree[field]))return;
      if(!supplemental.has(key))supplemental.set(key,{});supplemental.get(key)[field]=value;
    });
    for(const body of values){
      ensureCurrent();
      const saved=await readResponse(await fetch('/integrated/measurements',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,...scope,correct_existing_measurements:true})}));
      const current=ensureCurrent();
      const tree=(current.trees||[]).find(t=>t.tree_key===body.tree_key);
      if(tree)Object.assign(tree,{manual_diameter_cm:saved.manual_diameter_cm,diameter:saved.effective_diameter_cm,diameter_source:saved.diameter_source});
      drafts.set(rootKey(body.tree_key),saved.effective_diameter_cm??'No');
    }
    for(const [tree_key,manual_fields] of supplemental){
      ensureCurrent();
      const saved=await readResponse(await fetch('/integrated/measurements',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tree_key,manual_fields,...scope,correct_existing_measurements:true})}));
      const current=ensureCurrent();
      const tree=(current.trees||[]).find(t=>t.tree_key===tree_key);
      if(tree){Object.assign(tree,saved.effective_fields);tree.prediction_ready=!missing(tree.height)&&!missing(tree.clearance_distance);tree.measurement_sources??={};Object.assign(tree.measurement_sources,saved.measurement_sources);for(const [key,value] of Object.entries(saved.effective_fields))drafts.set(rootKey(tree_key)+':'+key,value??'No')}
    }
  }
  document.addEventListener('input',event=>{
    if(event.target.matches('[data-manual-dbh-key]'))drafts.set(rootKey(event.target.dataset.manualDbhKey),event.target.value);
    if(event.target.matches('[data-manual-geometry-key]'))drafts.set(rootKey(event.target.dataset.manualGeometryKey)+':'+event.target.dataset.measurementField,event.target.value);
    if(event.target.matches('[data-integrated-tree-species]'))drafts.set(rootKey(event.target.dataset.integratedTreeSpecies)+':species',event.target.value);
    if(event.target.matches('[data-integrated-tree-age]'))drafts.set(rootKey(event.target.dataset.integratedTreeAge)+':age',event.target.value);
    if(event.target.id==='diameter'){
      const toggle=$('use_diameter');if(toggle&&!toggle.checked){toggle.checked=true;applyFieldSelection('diameter')}
    }
  });
  document.addEventListener('click',async event=>{
    const button=event.target.closest('[data-estimate-integrated-age]');if(!button)return;
    const key=button.dataset.estimateIntegratedAge,tree=(integratedContext?.trees||[]).find(item=>item.tree_key===key),card=button.closest('[data-selected-tree-measurement]');
    if(!tree||!card)return;
    const species=card.querySelector('[data-integrated-tree-species]')?.value.trim()||'其他',height=Number(card.querySelector('[data-measurement-field="height"]')?.value||tree.height),diameter=number(card.querySelector('[data-manual-dbh-key]')?.value),status=card.querySelector('[data-integrated-age-status]');
    if(!diameter||!Number.isFinite(height)||height<=0){status.textContent='树龄估算需要有效树高和胸径；胸径为 No 时请先填现场 1.30 m 实测值。';return}
    button.disabled=true;status.textContent='正在按本地树种公式估算…';
    try{const result=await readResponse(await fetch('/predict/tree-age',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({species,diameter_cm:diameter,height_m:height,region:$('formulaRegion')?.value||'北京丰台'})})),estimate=Number(result.estimate_age_years),ageInput=card.querySelector('[data-integrated-tree-age]');if(!Number.isFinite(estimate)||estimate<=0)throw Error(result.message||'没有得到有效树龄');ageInput.value=String(Math.round(estimate));drafts.set(rootKey(key)+':age',ageInput.value);status.textContent=`估算约 ${Math.round(estimate)} 年；${result.results?.[0]?.formula||'树高与胸径经验融合公式'}。这是经验估计，可按现场档案修改。`}
    catch(error){status.textContent=`估算失败：${error.message}`}
    finally{button.disabled=false}
  });
  document.addEventListener('change',event=>{if(event.target.matches('[data-integrated-extra-tree]'))render()});
  return{number,display,label,render,saveAll,original};
})();
