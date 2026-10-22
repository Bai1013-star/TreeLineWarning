// One dated row per day; no network or invented meteorological forecasts.
window.DailyWeather = (() => {
  const states = new Map();
  const columns = [['wind_force','风力等级 (0–12级)','0','12'],['rainfall','降雨量 (mm)','0','3000'],['humidity','湿度 (%)','0','100'],['temperature','温度 (℃)','-90','70'],['sunlight_hours','日照时长 (h)','0','24']];
  const forceLimits = [.3,1.6,3.4,5.5,8,10.8,13.9,17.2,20.8,24.5,28.5,32.7];
  const forceSpeeds = [.15,.95,2.45,4.4,6.75,9.4,12.3,15.5,19,22.65,26.5,30.6,34.7];
  let editing = null;
  const scope = () => `${integratedContext?.project_root||location.origin}:${activeEntryMode}`;
  const state = () => states.get(scope()) || {days:7, rows:[]};
  const observed = () => activeEntryMode==='integrated' ? $('integratedObservationDate').value : activeEntryMode==='batch' ? $('externalObservationDate').value : localIsoDate(selectedObservationDate());
  function dates(base, days) {
    const start = new Date(`${base}T00:00:00Z`);
    if (!Number.isFinite(start.getTime()) || start.toISOString().slice(0,10)!==base) throw Error('请先填写有效观测日期');
    return Array.from({length:days}, (_,i) => new Date(start.getTime()+(i+1)*86400000).toISOString().slice(0,10));
  }
  function capture() {
    if (!editing) return;
    editing.rows = [...$('dailyWeatherRows').querySelectorAll('tr')].map(row => {
      const result = {enabled: row.querySelector('[data-weather-enabled]').checked};
      row.querySelectorAll('[data-daily-field]').forEach(input => { result[input.dataset.dailyField] = input.value; });
      return result;
    });
  }
  function render() {
    const days = Number($('dailyWeatherDays').value);
    const calendar = dates(observed(), days);
    $('dailyWeatherRows').innerHTML = calendar.map((day,i) => {
      const saved = editing.rows[i]||{};
      return `<tr><td><input type="checkbox" data-weather-enabled ${saved.enabled===false?'':'checked'} aria-label="使用第${i+1}天天气"></td><td>第 ${i+1} 天<br><small>${day}</small></td>${columns.map(([field,label,min,max])=>`<td><input type="number" data-daily-field="${field}" min="${min}" max="${max}" step="any" value="${esc(saved[field]??'')}" aria-label="第${i+1}天${label}" placeholder="未提供"></td>`).join('')}</tr>`;
    }).join('');
  }
  function request(base) {
    const saved = state();
    if (!saved.days) return {};
    const calendar = dates(base, saved.days);
    const rows = calendar.flatMap((day,i) => {
      const input = saved.rows[i]||{};
      if (input.enabled===false) return [];
      const result = {date:day,source:'人工逐日天气情景'};
      for (const [field,label,min,max] of columns) {
        if (input[field]==null || input[field]==='') continue;
        const value=Number(input[field]);
        if (!Number.isFinite(value) || value<Number(min) || value>Number(max) || (field==='wind_force'&&!Number.isInteger(value))) throw Error(`${day} ${label}超出有效范围`);
        result[field]=value;
      }
      return [result];
    });
    return {future_weather:rows,short_horizons_days:Array.from({length:saved.days},(_,i)=>i+1)};
  }
  function body(payload) {
    const base=payload.tree_data?.observation_date||payload.observation_date;
    const plan=request(base),days=plan.short_horizons_days?.length||7;
    return JSON.stringify({...payload,...plan,forecast_scope:'short',target_date:dates(base,days).at(-1)});
  }
  function html(result) {
    if (!result.daily_weather_enabled) return '';
    const force = value => forceLimits.findIndex(limit=>Number(value)<limit);
    const rows = (result.short_term_forecast||[]).map(day => `<tr><td>${esc(day.target_date)}</td><td>${force(day.predicted_wind_speed)<0?12:force(day.predicted_wind_speed)}级</td><td>${day.predicted_temperature}</td><td>${day.predicted_rainfall}</td><td>${day.predicted_humidity}</td><td>${day.predicted_sunlight_hours}</td><td>${esc(day.weather_source)}</td><td>${esc(day.hazard_level||day.risk_level)}</td><td>${day.predicted_effective_clearance}</td></tr>`).join('');
    return `<section class="work-section daily-weather-result"><h3>未来逐日天气与短时风险</h3><p>${esc(result.future_weather_note)} 风级采用区间代表速度计算；未填写的天气字段沿用观测情景，不是实测值。</p><div class="forecast-table-wrap"><table class="forecast-table"><thead><tr><th>日期</th><th>风级</th><th>温度 ℃</th><th>降雨 mm</th><th>湿度 %</th><th>日照 h</th><th>天气来源</th><th>风险等级</th><th>有效净空 m</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
  }
  document.addEventListener('DOMContentLoaded', () => {
    const section=document.createElement('section');section.className='input-section';section.id='dailyWeatherSection';section.hidden=typeof activeEntryMode!=='undefined'&&activeEntryMode==='batch';
    section.innerHTML='<h3>短时逐日天气（手动、当前点云、外部表格共用）</h3><button type="button" class="secondary" id="openDailyWeather">选择天数并逐日填写天气</button><span id="dailyWeatherStatus">未启用：短时计算采用观测天气情景。</span>';
    $('predictionStandardPanel').after(section);
    const dialog=document.createElement('dialog');dialog.id='dailyWeatherDialog';dialog.className='settings-modal';
    const dayOptions=Array.from({length:15},(_,index)=>`<option value="${index+1}">${index+1} 天</option>`).join('');
    dialog.innerHTML='<div class="modal-head"><h2>未来逐日天气输入</h2><button type="button" class="secondary" id="closeDailyWeather">取消</button></div><div class="modal-body"><p>选择未来天数后，系统严格生成相同数量的日期行：1 天显示 1 行，3 天显示 3 行，5 天显示 5 行，15 天显示 15 行。每行分别填写风力等级、降雨量、湿度、温度和日照时长；风级用区间代表速度参与计算。</p><h3>当前观测天气（也作为逐日空项的基准）</h3><div id="dailyWeatherBaseline"></div><div class="daily-weather-days"><label>预测未来天数（0–15 天）<select id="dailyWeatherDays">'+dayOptions+'</select></label><div class="daily-weather-shortcuts"><button type="button" class="secondary" data-weather-days="1">1 天</button><button type="button" class="secondary" data-weather-days="3">3 天</button><button type="button" class="secondary" data-weather-days="5">5 天</button><button type="button" class="secondary" data-weather-days="7">7 天</button><button type="button" class="secondary" data-weather-days="15">15 天</button></div></div><div class="forecast-table-wrap"><table class="forecast-table"><thead><tr><th>使用</th><th>日期</th>'+columns.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr></thead><tbody id="dailyWeatherRows"></tbody></table></div><button type="button" id="saveIntegratedWeather">保存手动与逐日天气</button><span id="dailyWeatherError" role="status"></span></div>';
    document.body.append(dialog);
    section.querySelector('h3').textContent='短时逐日天气（1～多棵树共用）';
    dialog.querySelector('.daily-weather-days label').firstChild.textContent='预测未来天数（1–15天）';
    $('dailyWeatherBaseline').insertAdjacentHTML('beforebegin','<section class="weather-history"><h3>天气输入历史记录</h3><p>保存后下次可一键复用。按第1天、第2天顺序套用到本次日期，导入后仍可修改。</p><label>选择已保存天气<select id="weatherTemplateList"><option value="">正在读取…</option></select></label><div class="actions"><button type="button" class="secondary" id="loadWeatherTemplate">一键导入所选天气</button><button type="button" class="secondary" id="deleteWeatherTemplate">删除所选记录</button></div><label>本次记录名称（可选）<input id="weatherTemplateName" placeholder="例如：北京秋季7天，方便下次查找"></label><label><input id="rememberWeatherTemplate" type="checkbox" checked> 保存天气时同时记入历史</label><p id="weatherTemplateStatus" role="status"></p></section>');
    let templates=[];
    const baselineIds={wind_force:'integratedWind',rainfall:'integratedRain',humidity:'integratedHumidity',temperature:'integratedTemperature',sunlight_hours:'integratedSunlight'};
    const baseline=()=>{const values={selected_fields:[]};for(const [field,id]of Object.entries(baselineIds)){values[field]=$(id).value;const key=field==='wind_force'?'wind_speed':field;if(document.querySelector(`[data-integrated-weather-use="${key}"]`)?.checked)values.selected_fields.push(field)}return values};
    const applyBaseline=values=>{for(const [field,id]of Object.entries(baselineIds)){if(values[field]!=null)$(id).value=values[field];const key=field==='wind_force'?'wind_speed':field,used=(values.selected_fields||[]).includes(field),toggle=document.querySelector(`[data-integrated-weather-use="${key}"]`);if(toggle)toggle.checked=used;if(activeEntryMode==='single'&&$(key)){if(values[field]!=null)$(key).value=values[field];if($('use_'+key)){$('use_'+key).checked=used;applyFieldSelection(key)}}}};
    const refreshTemplates=async()=>{try{const response=await readResponse(await fetch('/weather/templates',{cache:'no-store'}));templates=response.templates||[];$('weatherTemplateList').innerHTML='<option value="">'+(templates.length?'请选择一条天气记录':'暂无记录；保存天气后即可复用')+'</option>'+templates.map(item=>`<option value="${item.id}">${esc(item.name)} · ${item.plan.days}天 · ${esc(item.created_at)}</option>`).join('')}catch(error){$('weatherTemplateStatus').textContent='读取历史失败：'+error.message}};
    const legacyGrid=$('integratedWeatherDialog')?.querySelector('.grid');
    if(legacyGrid)$('dailyWeatherBaseline').append(legacyGrid);
    const open=()=>{try {editing=structuredClone(state());editing.scope=scope();if(activeEntryMode==='single'&&!states.has(scope())){const values={selected_fields:[]};for(const[field]of columns){const key=field==='wind_force'?'wind_speed':field;values[field]=$(key)?.value;if(fieldSelected(key))values.selected_fields.push(field)}applyBaseline(values)}$('dailyWeatherDays').value=String(editing.days);$('dailyWeatherError').textContent='';$('weatherTemplateStatus').textContent='';render();dialog.showModal();refreshTemplates();} catch(error){alert(error.message)}};
    DailyWeather.open=open;
    $('openDailyWeather').onclick=open;
    $('dailyWeatherDays').onchange=()=>{capture();editing.days=Number($('dailyWeatherDays').value);render();};
    dialog.querySelectorAll('[data-weather-days]').forEach(button=>button.onclick=event=>{event.preventDefault();capture();$('dailyWeatherDays').value=button.dataset.weatherDays;editing.days=Number(button.dataset.weatherDays);render()});
    $('closeDailyWeather').onclick=()=>dialog.close();
    const commit=async(remember=true)=>{
      capture();editing.days=$('dailyWeatherRows').children.length;
      if (editing.scope!==scope()) {$('dailyWeatherError').textContent='数据来源或项目已改变，请重新打开天气窗口。';return;}
      const previous=states.get(scope());states.set(scope(),editing);
      try {request(observed());editing.baseline=baseline();if(remember&&$('rememberWeatherTemplate').checked){await readResponse(await fetch('/weather/templates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('weatherTemplateName').value.trim()||`${observed()} · ${editing.days}天天气`,plan:editing})}));}applyBaseline(editing.baseline);$('dailyWeatherStatus').textContent=`已启用未来 ${editing.days} 天：按第1天至第${editing.days}天计算；空项沿用观测情景。`;$('integratedTargetDate').value=dates(observed(),editing.days).at(-1);if(typeof integratedWeatherMode!=='undefined'){integratedWeatherMode='manual';integratedWeatherMeta={source:'人工逐日天气',observed_at:localIsoDate()};if($('integratedWeatherStatus'))$('integratedWeatherStatus').textContent=`已采用人工逐日天气：未来 ${editing.days} 天。`;}dialog.close();}
      catch(error){if(previous)states.set(scope(),previous);else states.delete(scope());$('dailyWeatherError').textContent=error.message;}
    };
    $('saveIntegratedWeather').onclick=async()=>{const button=$('saveIntegratedWeather');button.disabled=true;try{await commit()}finally{button.disabled=false}};
    $('loadWeatherTemplate').onclick=async()=>{const item=templates.find(item=>String(item.id)===$('weatherTemplateList').value);if(!item){$('weatherTemplateStatus').textContent='请先选择一条天气记录';return}editing=structuredClone(item.plan);editing.scope=scope();$('dailyWeatherDays').value=String(editing.days);applyBaseline(editing.baseline||{});render();await commit(false)};
    $('deleteWeatherTemplate').onclick=async()=>{const id=$('weatherTemplateList').value;if(!id)return;if(!confirm('删除选中的天气历史记录？'))return;try{await readResponse(await fetch('/weather/templates/'+encodeURIComponent(id),{method:'DELETE'}));await refreshTemplates()}catch(error){$('weatherTemplateStatus').textContent=error.message}};
    const renderForecasts=renderForecastSections;
    renderForecastSections=function(result){renderForecasts(result);$('forecastOutput').insertAdjacentHTML('beforeend',html(result));};
  });
  function speedFromForce(force){const value=Math.max(0,Math.min(12,Math.round(Number(force)||0)));return forceSpeeds[value]}
  function forceFromSpeed(speed){const index=forceLimits.findIndex(limit=>Number(speed)<limit);return index<0?12:index}
  function setDays(days){const count=Number(days);if(!Number.isInteger(count)||count<1||count>15)return;const saved=structuredClone(state());saved.days=count;saved.rows=saved.rows.slice(0,count);states.set(scope(),saved);if($('dailyWeatherStatus'))$('dailyWeatherStatus').textContent=`本次预测未来${count}天，可打开天气窗口逐日填写或导入历史。`}
  return {body,request,html,speedFromForce,forceFromSpeed,setDays};
})();
